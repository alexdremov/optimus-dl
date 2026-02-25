from __future__ import annotations

import contextlib
import logging
from abc import (
    ABC,
    abstractmethod,
)
from collections import (
    OrderedDict,
    defaultdict,
)
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
)

from optimus_dl.modules.distributed import Collective

logger = logging.getLogger(__name__)


@dataclass
class MeterEntry:
    """Container for a meter and its logging metadata.

    This dataclass holds an instance of a `BaseMeter` along with metadata
    that controls its behavior within a `MeterGroup` and during logging.

    Attributes:
        meter: The actual `BaseMeter` instance responsible for accumulating data.
        reset: If True, this meter will be reset (removed from the group)
               after each logging step in its `MeterGroup`. This is typically
               used for per-step or per-iteration meters. If False, the meter
               accumulates its state across multiple steps/iterations.
        priority: An integer representing the logging priority. Meters with
                  lower priority values will appear earlier in the logs.
    """

    meter: BaseMeter
    reset: bool = False
    priority: int = 0

    def state_dict(self) -> dict[str, Any]:
        """Return the current state of this `MeterEntry` for checkpointing.

        This method serializes the internal state of the `meter` itself,
        along with the `reset` and `priority` flags, and the class name
        of the meter, to allow for reconstruction.

        Returns:
            A dictionary containing the serializable state of the `MeterEntry`.
        """
        return {
            "meter": self.meter.state_dict(),
            "reset": self.reset,
            "priority": self.priority,
            "meter_class": self.meter.__class__.__name__,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the `MeterEntry`'s state from a checkpoint.

        This method reconstructs the `BaseMeter` instance and restores its
        internal state from the provided `state_dict`. It also updates
        the `reset` and `priority` flags.

        Args:
            state_dict: A dictionary containing the saved state of a `MeterEntry`.
                        Supports legacy checkpoints by looking for "metric_class"
                        and "metric" keys if "meter_class" and "meter" are not found.
        """
        import optimus_dl.modules.metrics as metrics

        # Fallback for legacy checkpoints where meters were named 'metrics'
        class_name = state_dict.get("meter_class") or state_dict.get("metric_class")
        if not class_name:
            logger.warning(
                "Could not find 'meter_class' or 'metric_class' in MeterEntry state_dict. "
                "Attempting to infer from available classes, this may lead to errors."
            )
            # Attempt to infer if class_name is missing for old checkpoints
            # This might require more sophisticated logic if class names have changed drastically
            raise NotImplementedError(
                "Dynamic class name inference not yet implemented for MeterEntry load_state_dict without 'class_name'."
            )

        meter_state = state_dict.get("meter") or state_dict.get("metric")
        if not meter_state:
            logger.warning(
                "Could not find 'meter' or 'metric' state in MeterEntry state_dict. "
                "Attempting to load with empty state, this may lead to errors."
            )
            meter_state = (
                {}
            )  # Allow to proceed with empty state, constructor should handle it

        # Use getattr to fetch the meter class, then reconstruct it
        meter_cls = getattr(metrics, class_name)
        legacy_class_name_map = {
            "AverageMetric": "AverageMeter",
            "SummedMetric": "SummedMeter",
            "FrequencyMetric": "FrequencyMeter",
            "BaseMetric": "BaseMeter",
        }
        mapped_class_name = legacy_class_name_map.get(class_name, class_name)
        if mapped_class_name != class_name:
            logger.info(
                "Mapping legacy metric class '%s' to meter class '%s' when loading MeterEntry.",
                class_name,
                mapped_class_name,
            )
        # Use getattr to fetch the meter class, then reconstruct it
        meter_cls = getattr(metrics, mapped_class_name)
        self.meter = meter_cls.from_state_dict(meter_state)
        self.reset = state_dict["reset"]
        self.priority = state_dict["priority"]


class MeterGroup:
    """A named collection of meters that are logged together.

    This class manages a group of related meters (e.g., 'train' or 'eval'). It
    handles:

    - **Sampling Frequency**: Only triggers logging every `log_freq` steps.
    - **Priority Sorting**: Ensures consistent ordering of meters in output.
    - **State Management**: Can reset meters after logging and serialize the
      entire group state for checkpointing.

    Args:
        name: Unique name for the group.
        log_freq: Frequency (in iterations) at which to trigger logging.
                  If None, defaults to 1 (log every iteration).
    """

    def __init__(self, name: str, log_freq: int | None = None):
        self.name = name
        self.log_freq = log_freq or 1
        self._meters: OrderedDict[str, MeterEntry] = OrderedDict()
        self._keys_sorted: list[str] = []
        self._iteration_counter: int = 0

    def compute(self) -> dict[str, float | int | dict[str, float | int]]:
        """Compute the current values for all meters in the group.

        Iterates through all meters currently in the group (sorted by priority)
        and calls their `compute()` method to get their current value.

        Returns:
            An `OrderedDict` mapping meter names to their computed values.
            The values can be floats, integers, or nested dictionaries
            (for meters emitting multiple sub-values).
        """
        return OrderedDict(
            (name, self._meters[name].meter.compute()) for name in self._keys_sorted
        )

    @property
    def meters(self) -> OrderedDict[str, MeterEntry]:
        """Returns the internal `OrderedDict` of `MeterEntry` objects.

        Note: The meters are returned in their natural insertion order,
              not sorted by priority. Use `_keys_sorted` for ordered iteration.
        """
        return self._meters

    def step(self) -> bool:
        """Increment the internal iteration counter for the group.

        This method should be called once per relevant step (e.g., per batch)
        to track progress and determine when logging should occur based on `log_freq`.

        Returns:
            True if the current step is a logging step (i.e., `_iteration_counter`
            is a multiple of `log_freq`), False otherwise.
        """
        self._iteration_counter += 1
        return (self._iteration_counter % self.log_freq) == 0

    def should_log(self) -> bool:
        """Check if the current iteration should trigger logging.

        This is a passive check that does not increment the iteration counter.

        Returns:
            True if logging should occur at the current iteration, False otherwise.
        """
        return (self._iteration_counter % self.log_freq) == 0

    def add_meter(self, name: str, meter_entry: MeterEntry):
        """Add a new meter entry to the group.

        If a meter with the same `name` already exists, it will be overwritten.
        After adding, the sorted list of keys is updated to reflect any priority changes.

        Args:
            name: The unique identifier for the meter within this group.
            meter_entry: The `MeterEntry` object containing the `BaseMeter` instance
                         and its metadata.
        """
        self._meters[name] = meter_entry
        self._update_keys_sorted()

    def _update_keys_sorted(self):
        """Update the sorted list of meter keys based on priorities.

        This internal helper method re-sorts `self._keys_sorted` whenever a meter
        is added or removed, ensuring that `compute()` and other operations
        process meters in the correct priority order.
        """
        self._keys_sorted = sorted(
            self._meters.keys(),
            key=lambda k: self._meters[k].priority,
        )

    def get_meter(self, name: str) -> MeterEntry | None:
        """Retrieve a specific `MeterEntry` by its name.

        Args:
            name: The name of the meter to retrieve.

        Returns:
            The `MeterEntry` if found, otherwise None.
        """
        return self._meters.get(name)

    def reset(self):
        """Reset all meters marked for reset after logging.

        This method iterates through all `MeterEntry` objects in the group.
        If an entry's `reset` flag is True, the corresponding meter is removed
        from the group. This is typically called after a logging event to
        prepare for the next accumulation cycle for per-step meters.
        """
        # Create a copy of keys to iterate over as we might modify _meters
        for key in list(self._meters.keys()):
            entry = self._meters[key]
            if entry.reset:
                self._meters.pop(key)
        self._update_keys_sorted()

    def state_dict(self) -> dict[str, Any]:
        """Return the entire `MeterGroup` state for checkpointing.

        Serializes the group's name, logging frequency, and the state of all
        contained `MeterEntry` objects.

        Returns:
            A dictionary containing the serializable state of the `MeterGroup`.
        """
        return {
            "name": self.name,
            "log_freq": self.log_freq,
            "meters": {
                name: entry.state_dict() for name, entry in self._meters.items()
            },
            "iteration_counter": self._iteration_counter,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the `MeterGroup` state from a checkpoint.

        Reconstructs the group's internal state, including all its meters
        and their individual states, from the provided `state_dict`.

        Args:
            state_dict: A dictionary containing the saved state of a `MeterGroup`.
                        Supports legacy checkpoints by looking for a "metrics"
                        key if "meters" is not found for the collection of meters.

        Raises:
            AssertionError: If the name in the `state_dict` does not match the
                            current group's name, indicating a mismatch.
        """
        assert (
            self.name == state_dict["name"]
        ), f"Name mismatch: expected {self.name}, got {state_dict['name']}"
        self.log_freq = state_dict["log_freq"]
        self._iteration_counter = state_dict.get(
            "iteration_counter", 0
        )  # Backward compatibility
        self._meters = OrderedDict()

        # Backward compatibility for 'metrics' key in state_dict
        meters_data = state_dict.get("meters") or state_dict.get("metrics", {})
        if not meters_data:
            logger.warning(
                f"No 'meters' or 'metrics' found in state_dict for MeterGroup '{self.name}'. Initializing with empty meters."
            )

        for name, entry_state in meters_data.items():
            entry = MeterEntry(meter=None)  # type: ignore # Meter will be set by load_state_dict
            try:
                entry.load_state_dict(entry_state)
                self._meters[name] = entry
            except Exception as e:
                logger.error(
                    f"Failed to load MeterEntry '{name}' for MeterGroup '{self.name}': {e}"
                )
                # Decide how to handle this error: skip, raise, or re-initialize
                # For now, we will skip the problematic meter but log the error.
        self._update_keys_sorted()


_meter_groups: OrderedDict[str, MeterGroup] = OrderedDict()
_active_meter_groups = defaultdict(lambda: 0)


class BaseMeter(ABC):
    """Abstract base class for all individual stateful meter implementations.

    Meters are responsible for accumulating raw data (via the `log` method)
    and processing it into a final, reportable value (via the `compute` method).
    A key feature of `BaseMeter` is its support for merging states from other
    meter instances, which is crucial for distributed aggregation across multiple
    workers or processes.

    Subclasses must implement:
    - `compute()`: To return the current aggregated value(s).
    - `log(**kwargs)`: To accumulate new data points.
    - `merge(other_state)`: To combine its state with that of another meter.
    """

    @abstractmethod
    def compute(self) -> float | int | dict[str, float | int]:
        """Compute the final meter value from accumulated data.

        This method should perform any necessary calculations on the internally
        accumulated data and return the result. It should not modify the meter's
        internal state.

        Returns:
            The computed value, which can be a float, integer, or a dictionary
            of sub-values (e.g., {'precision': 0.8, 'recall': 0.9}).
        """
        raise NotImplementedError

    @abstractmethod
    def log(self, **kwargs):
        """Accumulate new raw data points into the meter's internal state.

        This method is called for each data point or batch that needs to be
        processed by the meter. The specific arguments in `**kwargs` depend
        on the concrete meter implementation.

        Args:
            **kwargs: Arbitrary keyword arguments representing the data to be logged.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(self, other_state: dict[str, Any]):
        """Merge state from another instance of the same meter type.

        This method is critical for distributed training, allowing the states
        of meters from different processes/ranks to be combined into a single,
        globally consistent state. The `other_state` should be a dictionary
        representing the internal state of another meter.

        Args:
            other_state: A dictionary containing the internal state of another
                         `BaseMeter` instance, typically obtained via `state_dict()`.
        """
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> BaseMeter:
        """Create a new meter instance and restore its state from a dictionary.

        This factory method constructs an instance of the meter class (`cls`)
        and then calls its `load_state_dict` method to populate its internal state.

        Args:
            state_dict: A dictionary containing the saved internal state of a meter.

        Returns:
            A new instance of the `BaseMeter` subclass with its state restored.
        """
        instance = cls()
        instance.load_state_dict(state_dict)
        return instance

    def state_dict(self) -> dict[str, Any]:
        """Return the internal meter state as a dictionary for checkpointing.

        By default, this returns a shallow copy of `self.__dict__`. Subclasses
        may override this method if they need custom serialization logic (e.g.,
        to handle non-serializable attributes or specific data structures).

        Returns:
            A dictionary representing the internal, serializable state of the meter.
        """
        return self.__dict__

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the internal meter state from a dictionary.

        By default, this updates `self.__dict__` with the provided `state_dict`.
        Subclasses may override this method for custom deserialization,
        especially if `state_dict()` was also overridden.

        Args:
            state_dict: A dictionary containing the saved internal state of a meter.
        """
        self.__dict__.update(state_dict)


@contextlib.contextmanager
def metrics_group(name: str, log_freq: int | None = None, force_recreate: bool = False):
    """Context manager for activating a metrics group.

    While inside this context, any calls to `log_metric` will be directed to
    the `MeterGroup` identified by `name`. This allows for grouping related
    meters (e.g., "train" or "eval" metrics).

    Args:
        name: Name of the `MeterGroup` to activate.
        log_freq: Optional logging frequency (in iterations) to set or update
                  for this group. If the group already exists and `log_freq`
                  is provided, its frequency will be updated.
        force_recreate: If True, any existing `MeterGroup` with the given `name`
                        will be removed and a new one created, effectively clearing
                        its state.

    Yields:
        bool: True if the group should trigger logging at this step, based on
              its `log_freq` and internal iteration counter. False otherwise.
    """
    if force_recreate:
        _meter_groups.pop(name, None)
    _meter_groups.setdefault(name, MeterGroup(name, log_freq=log_freq))
    if log_freq is not None:
        _meter_groups[name].log_freq = log_freq
    _active_meter_groups[name] += 1

    # Return whether we should log at current iteration
    should_log = _meter_groups[name].should_log()

    try:
        yield should_log
    finally:
        _active_meter_groups[name] -= 1
        if _active_meter_groups[name] == 0:
            _active_meter_groups.pop(name)


def compute_metrics(
    group_name: str, aggregate: bool = False, collective: Collective | None = None
) -> dict[str, float | int | dict[str, float | int]]:
    """Compute final values for a named `MeterGroup`, with optional distributed aggregation.

    This function retrieves the specified `MeterGroup`, computes the current
    value for each of its meters, and optionally aggregates these values
    across distributed ranks.

    If `aggregate` is True, it performs an all-gather of meter states across
    all distributed ranks and merges them before computing final values. This
    ensures that metrics reflect a global view of the data.

    Args:
        group_name: Name of the `MeterGroup` to compute.
        aggregate: If True, meter states are aggregated from all ranks
                   using the provided `collective`. If False, only local
                   meter values are returned.
        collective: A `Collective` instance (from `optimus_dl.modules.distributed`)
                    required for distributed aggregation if `aggregate` is True.

    Returns:
        A dictionary mapping meter names (or metric names, as exposed) to
        their computed values. These values can be floats, integers, or nested
        dictionaries. Returns an empty dictionary if the group name is not found.
    """
    if group_name not in _meter_groups:
        logger.debug(f"MeterGroup '{group_name}' not found for computing metrics.")
        return {}

    group = _meter_groups[group_name]
    local_metrics = group.compute()  # These are actually meter outputs

    if not aggregate or collective is None:
        return local_metrics

    # Collect local meter states to send for aggregation
    local_meter_states = {
        name: entry.meter.state_dict()
        for name, entry in group.meters.items()
        if name in local_metrics  # Only consider meters that produced a local metric
    }

    # Gather all meter states from all ranks in one communication
    all_rank_states: list[dict[str, dict[str, Any]]] = collective.all_gather_objects(
        local_meter_states
    )

    # Aggregate meters across ranks using their merge functionality
    aggregated_metrics: dict[str, float | int | dict[str, float | int]] = {}

    for name in local_metrics.keys():  # Iterate over keys that were computed locally
        if name not in group.meters:
            continue

        entry = group.meters[name]
        # Create a fresh instance of the meter's class for aggregation
        aggregated_meter = entry.meter.__class__()

        # Merge states from all ranks
        for rank_states in all_rank_states:
            if name in rank_states:
                try:
                    aggregated_meter.merge(rank_states[name])
                except Exception as e:
                    logger.error(
                        f"Error merging state for meter '{name}' from rank states: {e}"
                    )
                    # Depending on error, could skip this rank's state or handle differently
                    continue

        # Compute final aggregated value
        try:
            aggregated_metrics[name] = aggregated_meter.compute()
        except Exception as e:
            logger.error(
                f"Error computing aggregated metric for '{name}': {e}. Falling back to local value."
            )
            # Fall back to local value if aggregation computation fails
            aggregated_metrics[name] = local_metrics[name]

    return aggregated_metrics


def step_metrics(name: str) -> None:
    """Explicitly step the iteration counter for a named `MeterGroup`.

    This function allows external components to manually advance the iteration
    counter of a specific `MeterGroup`, which can influence when `should_log`
    returns True.

    Args:
        name: The name of the `MeterGroup` to step.
    """
    if name in _meter_groups:
        _meter_groups[name].step()
    else:
        logger.debug(f"Attempted to step non-existent MeterGroup '{name}'.")


def reset_metrics(name: str) -> None:
    """Reset all resettable meters within a named `MeterGroup`.

    This function triggers the `reset()` method on the specified `MeterGroup`,
    which in turn removes all `MeterEntry` objects that have their `reset` flag
    set to True.

    Args:
        name: The name of the `MeterGroup` to reset.
    """
    if name in _meter_groups:
        _meter_groups[name].reset()
    else:
        logger.debug(f"Attempted to reset non-existent MeterGroup '{name}'.")


def state_dict() -> dict[str, Any]:
    """Return the combined state dictionary for all managed `MeterGroup`s.

    This function collects the `state_dict()` from each active `MeterGroup`,
    allowing the entire metrics system state to be checkpointed.

    Returns:
        A dictionary where keys are `MeterGroup` names and values are their
        respective state dictionaries.
    """
    return {
        group_name: group.state_dict() for group_name, group in _meter_groups.items()
    }


def load_state_dict(state_dict: dict[str, Any]) -> None:
    """Restore the state for all managed `MeterGroup`s from a state dictionary.

    This function iterates through the provided `state_dict`, recreating
    `MeterGroup`s as needed and loading their saved states.

    Args:
        state_dict: A dictionary containing the saved state of all `MeterGroup`s,
                    typically obtained from a previous call to `state_dict()`.
    """
    for group_name, group_data in state_dict.items():
        if group_name not in _meter_groups:
            # Recreate MeterGroup if it doesn't exist, using saved log_freq
            log_freq = group_data.get(
                "log_freq", 1
            )  # Default to 1 if not in state_dict
            _meter_groups[group_name] = MeterGroup(name=group_name, log_freq=log_freq)
        try:
            _meter_groups[group_name].load_state_dict(group_data)
        except Exception as e:
            logger.error(
                f"Failed to load state_dict for MeterGroup '{group_name}': {e}"
            )
            # Depending on error, could re-initialize or skip loading this group


def _evaluate_value(value_or_callable: Any) -> Any:
    """Helper to evaluate a value or callable lazily.

    If the input is a callable (e.g., a lambda function), it is executed
    and its return value is returned. Otherwise, the input value itself
    is returned. This is used to support lazy evaluation of metric inputs.

    Args:
        value_or_callable: A value or a callable that returns a value.

    Returns:
        The evaluated value.
    """
    if callable(value_or_callable):
        return value_or_callable()
    return value_or_callable


def log_metric(
    name: str,
    meter_factory: Callable[[], BaseMeter],
    reset: bool = True,
    priority: int = 100,
    force_log: bool = False,
    **kwargs: Any,
):
    """Log data point(s) to all currently active meter groups.

    This is the primary function for adding data to meters within active
    `MeterGroup`s. It ensures that data is only logged when the group's
    `should_log()` condition is met (unless `force_log` is True) and
    handles the creation of meters if they don't already exist in a group.

    Args:
        name: The name under which this metric's data will be stored and reported.
              This acts as the key for the `MeterEntry` within the `MeterGroup`.
        meter_factory: A callable (e.g., a lambda function) that, when called
                       with no arguments, returns a new instance of a `BaseMeter`
                       subclass. This factory is used only if a meter with the
                       given `name` does not already exist in the group.
        reset: If True, the meter created or used for this log will be removed
               from its `MeterGroup` after the group's `reset()` method is called.
               Defaults to True, suitable for per-iteration metrics.
        priority: An integer determining the order of this meter in logs.
                  Lower numbers mean higher priority (appear earlier). Defaults to 100.
        force_log: If True, the metric will be logged even if the current
                   `MeterGroup`'s `should_log()` method returns False. This is
                   useful for critical events or debugging that need to be logged
                   regardless of frequency settings. Defaults to False.
        **kwargs: Arbitrary keyword arguments that will be passed directly
                  to the `log()` method of the `BaseMeter` instance. These
                  typically represent the actual data points (e.g., `value`, `weight`).
    """
    for group_name in _active_meter_groups:
        group = _meter_groups[group_name]

        # Only evaluate expensive callables if we should log or are forcing a log
        if group.should_log() or force_log:
            # Evaluate any callable values in kwargs lazily, only if logging is active
            evaluated_kwargs = {k: _evaluate_value(v) for k, v in kwargs.items()}

            if name not in group.meters:
                # If meter doesn't exist, create it using the factory and add to group
                group.add_meter(
                    name,
                    MeterEntry(
                        meter=meter_factory(),
                        reset=reset,
                        priority=priority,
                    ),
                )
            # Log the evaluated data to the meter
            group.meters[name].meter.log(**evaluated_kwargs)
