import contextlib
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

from optimus_dl.modules.distributed import Collective


@dataclass
class MetricEntry:
    """Container for a metric and its logging metadata.

    Attributes:
        metric: The actual BaseMetric instance.
        reset: If True, the metric is reset after each step.
        priority: Priority for ordering metrics in logs (lower is higher priority).
    """

    metric: "BaseMetric"
    reset: bool = False
    priority: int = 0

    def state_dict(self) -> dict[str, Any]:
        """Return entry state for checkpointing."""
        return {
            "metric": self.metric.state_dict(),
            "reset": self.reset,
            "priority": self.priority,
            "metric_class": self.metric.__class__.__name__,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restore entry state."""
        import optimus_dl.modules.metrics as metrics

        self.metric = getattr(metrics, state_dict["metric_class"]).from_state_dict(
            state_dict["metric"]
        )
        self.reset = state_dict["reset"]
        self.priority = state_dict["priority"]


class MetricGroup:
    """A named collection of metrics that are logged together.

    This class manages a group of related metrics (e.g., 'train' or 'eval'). It
    handles:
    - **Sampling Frequency**: Only triggers logging every `log_freq` steps.
    - **Priority Sorting**: Ensures consistent ordering of metrics in output.
    - **State Management**: Can reset metrics after logging and serialize the
      entire group state.

    Args:
        name: Unique name for the group.
        log_freq: Frequency (in iterations) at which to trigger logging.
    """

    def __init__(self, name: str, log_freq: int | None = None):
        self.name = name
        self.log_freq = log_freq or 1
        self._metrics: OrderedDict[str, MetricEntry] = OrderedDict()
        self._keys_sorted = []
        self._iteration_counter = 0

    def compute(self) -> dict[str, float | int | dict[str, float | int]]:
        """Compute the current values for all metrics in the group."""
        return OrderedDict(
            (name, self._metrics[name].metric.compute()) for name in self._keys_sorted
        )

    @property
    def metrics(self) -> OrderedDict[str, MetricEntry]:
        """Return the metrics in sorted order by priority."""
        return self._metrics

    def step(self) -> bool:
        """Increment iteration counter and return whether to log at this step."""
        self._iteration_counter += 1
        return (self._iteration_counter % self.log_freq) == 0

    def should_log(self) -> bool:
        """Check if the current iteration should trigger logging."""
        return (self._iteration_counter % self.log_freq) == 0

    def add_metric(self, name: str, metric_entry: MetricEntry):
        """Add a new metric entry to the group."""
        self._metrics[name] = metric_entry
        self._update_keys_sorted()

    def _update_keys_sorted(self):
        """Update the sorted list of metric keys based on priorities."""
        self._keys_sorted = sorted(
            self._metrics.keys(),
            key=lambda k: self._metrics[k].priority,
        )

    def get_metric(self, name: str) -> MetricEntry | None:
        """Retrieve a specific metric entry by name."""
        return self._metrics.get(name)

    def reset(self):
        """Reset all metrics marked for reset after logging."""
        for key in list(self._metrics.keys()):
            entry = self._metrics[key]
            if entry.reset:
                self._metrics.pop(key)
        self._update_keys_sorted()

    def state_dict(self) -> dict[str, Any]:
        """Return the entire group state for checkpointing."""
        return {
            "name": self.name,
            "log_freq": self.log_freq,
            "metrics": {
                name: entry.state_dict() for name, entry in self._metrics.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restore the group state."""
        assert self.name == state_dict["name"]
        self.log_freq = state_dict["log_freq"]
        self._metrics = OrderedDict()
        for name, entry_state in state_dict["metrics"].items():
            entry = MetricEntry(metric=None)  # type: ignore
            entry.load_state_dict(entry_state)
            self._metrics[name] = entry
        self._update_keys_sorted()


_metric_groups: OrderedDict[str, MetricGroup] = OrderedDict()
_active_metric_groups = defaultdict(lambda: 0)


class BaseMetric(ABC):
    """Abstract base class for all individual metric implementations.

    Metrics are responsible for accumulating raw data (log) and processing it
    into a final value (compute). They must support merging states for
    distributed aggregation.
    """

    @abstractmethod
    def compute(self) -> float | int | dict[str, float | int]:
        """Compute the final metric value from accumulated data."""
        raise NotImplementedError

    @abstractmethod
    def log(self, **kwargs):
        """Accumulate new raw data points."""
        raise NotImplementedError

    @abstractmethod
    def merge(self, other_state):
        """Merge state from another instance of the same metric type."""
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]):
        """Create a new metric instance restored from state."""
        instance = cls()
        instance.load_state_dict(state_dict)
        return instance

    def state_dict(self) -> dict[str, Any]:
        """Return internal metric state."""
        return self.__dict__

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restore internal metric state."""
        self.__dict__.update(state_dict)


@contextlib.contextmanager
def metrics_group(name: str, log_freq: int | None = None, force_recreate: bool = False):
    """Context manager for activating a metrics group.

    While inside this context, any calls to `log_metric` will be recorded in
    this group.

    Args:
        name: Name of the group to activate.
        log_freq: Optional logging frequency to set/update.
        force_recreate: If True, clears existing group state.

    Yields:
        bool: True if the group should trigger logging at this step.
    """
    if force_recreate:
        _metric_groups.pop(name, None)
    _metric_groups.setdefault(name, MetricGroup(name, log_freq=log_freq))
    if log_freq is not None:
        _metric_groups[name].log_freq = log_freq
    _active_metric_groups[name] += 1

    # Return whether we should log at current iteration
    should_log = _metric_groups[name].should_log()

    try:
        yield should_log
    finally:
        _active_metric_groups[name] -= 1
        if _active_metric_groups[name] == 0:
            _active_metric_groups.pop(name)


def compute_metrics(
    group_name: str, aggregate: bool = False, collective: Collective | None = None
) -> dict[str, float | int | dict[str, float | int]]:
    """Compute final values for a group, with optional distributed aggregation.

    If `aggregate` is True, it performs an all-gather of metric states across
    all distributed ranks and merges them before computing final values.

    Args:
        group_name: Name of the group to compute.
        aggregate: If True, merges metrics from all ranks.
        collective: Distributed collective for aggregation.

    Returns:
        Dictionary mapping metric names to computed values.
    """
    if group_name not in _metric_groups:
        return {}

    group = _metric_groups[group_name]
    local_metrics = group.compute()
    if not aggregate or collective is None:
        return local_metrics

    # Single all_gather for all metric states to minimize communication
    local_metric_states = {
        name: entry.metric.state_dict()
        for name, entry in group.metrics.items()
        if name in local_metrics
    }

    # Gather all metric states from all ranks in one communication
    all_rank_states = collective.all_gather_objects(local_metric_states)

    # Aggregate metrics across ranks using merge functionality
    aggregated_metrics = {}

    for name in local_metrics.keys():
        if name not in group.metrics:
            continue

        entry = group.metrics[name]

        # Create a new metric instance for aggregation
        aggregated_metric = entry.metric.__class__()

        # Merge states from all ranks
        for rank_states in all_rank_states:
            if name in rank_states:
                aggregated_metric.merge(rank_states[name])

        # Compute final aggregated value
        try:
            aggregated_metrics[name] = aggregated_metric.compute()
        except Exception:
            # Fall back to local value if aggregation fails
            aggregated_metrics[name] = local_metrics[name]

    return aggregated_metrics


def step_metrics(name: str):
    """Explicitly step the iteration counter for a metric group."""
    if name in _metric_groups:
        _metric_groups[name].step()


def reset_metrics(name: str):
    """Reset all resettable metrics in a group."""
    if name in _metric_groups:
        _metric_groups[name].reset()


def state_dict():
    """Return state for all managed metric groups."""
    return {
        group_name: group.state_dict() for group_name, group in _metric_groups.items()
    }


def load_state_dict(state_dict: dict[str, Any]):
    """Restore state for all managed metric groups."""
    for group_name, group in state_dict.items():
        if group_name not in _metric_groups:
            _metric_groups[group_name] = MetricGroup(
                name=group_name, log_freq=group.get("log_freq", None)
            )
        _metric_groups[group_name].load_state_dict(group)


def _evaluate_value(value_or_callable):
    """Helper to evaluate a value or callable lazily."""
    if callable(value_or_callable):
        return value_or_callable()
    return value_or_callable


def log_metric(
    name: str,
    metric_factory,
    reset: bool = True,
    priority: int = 100,
    force_log: bool = False,
    **kwargs,
):
    """Log data point(s) to all currently active metric groups.

    Args:
        name: Name of the metric.
        metric_factory: Callable that creates a new metric instance if needed.
        reset: Whether to reset the metric after making a step or preserving it across steps.
        priority: Ordering priority.
        force_log: If True, logs even if `should_log` is False for the group.
        **kwargs: Data to pass to the metric's `log` method.
    """
    for group_name in _active_metric_groups:
        group = _metric_groups[group_name]

        # Only evaluate expensive callables if we should log
        if group.should_log() or force_log:
            # Evaluate any callable values in kwargs lazily
            evaluated_kwargs = {k: _evaluate_value(v) for k, v in kwargs.items()}

            if name not in group.metrics:
                group.add_metric(
                    name,
                    MetricEntry(
                        metric=metric_factory(),
                        reset=reset,
                        priority=priority,
                    ),
                )
            group.metrics[name].metric.log(**evaluated_kwargs)
