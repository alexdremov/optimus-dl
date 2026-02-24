from __future__ import annotations

import logging
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)
from collections.abc import Callable

import torch

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.metrics.base import (
    BaseMeter,  # Changed from BaseMetric to BaseMeter
)
from optimus_dl.modules.metrics.base import (
    log_metric,  # Ensure log_metric is imported from base
)
from optimus_dl.modules.metrics.base import (
    metrics_group,
)
from optimus_dl.modules.metrics.common import AverageMeter  # Changed from AverageMetric
from optimus_dl.modules.metrics.common import SummedMeter  # Changed from SummedMetric
from optimus_dl.modules.metrics.common import (  # Re-importing specific meters for local scope
    AveragedExponentMeter,
)
from optimus_dl.modules.metrics.definitions import (
    Metric,
    build_metric,
)
from optimus_dl.modules.metrics.source import (
    MetricSource,
    build_source,
)

logger = logging.getLogger(__name__)


class GatherMetric(BaseMeter):  # Changed from BaseMetric to BaseMeter
    """Accumulator that gathers all raw values across the entire dataset.

    Use this for meters that require full dataset context (e.g., BLEU, ROC-AUC).
    """

    def __init__(self):
        """Initializes the GatherMetric with an empty list to store values."""
        self.values: list[Any] = []

    def log(self, value: Any):
        """Logs a single value to be gathered.

        Args:
            value: The value to append to the internal list.
        """
        self.values.append(value)

    def compute(self) -> list[Any]:
        """Returns the list of all gathered values.

        Returns:
            A list containing all values that have been logged.
        """
        return self.values

    def merge(self, other_state: dict[str, Any]):
        """Merges the state from another GatherMetric instance.

        Extends the current list of values with the values from `other_state`.

        Args:
            other_state: A dictionary containing the 'values' key from another
                         GatherMetric instance.
        """
        self.values.extend(other_state["values"])

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the GatherMetric for checkpointing.

        Returns:
            A dictionary containing the 'values' list.
        """
        return {"values": self.values}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restores the state of the GatherMetric from a state dictionary.

        Args:
            state_dict: A dictionary containing the 'values' key to restore from.
        """
        self.values = state_dict["values"]


@dataclass
class EngineMetricConfig(RegistryConfigStrict):
    """Configuration for a single Metric in the Engine.

    This dataclass defines how a specific `Metric` (stateless logic) should be
    configured within the `MetricEngine`. It specifies the `Metric` type,
    how its sub-metrics should be accumulated, and optional filtering rules.

    Attributes:
        _name: The registered name of the component to instantiate. This field is
               explicitly defined here as non-optional to satisfy dataclass
               ordering rules, as `type` is also a non-default field.
        type: Configuration for the `Metric` itself (e.g., AccuracyMetricConfig).
              This should be a dictionary that can be used by `build_metric`.
        accumulators: A mapping from sub-metric names (as emitted by `Metric.batch_call`)
                      to accumulator types (e.g., 'average', 'sum', 'gather', 'perplexity').
                      If a sub-metric is not specified here, it defaults to 'average'.
        reset: If True, the meters associated with this metric will be reset
               (removed from the group) after each logging step. Defaults to True.
        priority: An integer priority for logging order. Lower numbers mean
                  higher priority. Defaults to 100.
        slice_filter: An optional Python expression (string) that is evaluated
                      against the current `batch`. If the expression evaluates
                      to False, the metric is skipped for that batch. This allows
                      conditional metric computation.
    """

    _name: str = field(
        metadata={"help": "The registered name of the component to instantiate."}
    )  # Make it explicitly non-default and redefine to ensure proper order
    type: dict[str, Any] = field(
        metadata={"help": "Config for the Metric (e.g., AccuracyMetricConfig)."}
    )
    accumulators: dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Mapping from sub-metric names to accumulator types. Supported: 'average', 'sum', 'gather', 'perplexity'. Defaults to 'average' if not specified for a sub-metric."
        },
    )
    reset: bool = field(
        default=True,
        metadata={
            "help": "Whether to reset the meters for this metric after each logging step."
        },
    )
    priority: int = field(
        default=100,
        metadata={
            "help": "Priority for ordering meters in logs (lower is higher priority)."
        },
    )
    slice_filter: str | None = field(
        default=None,
        metadata={
            "help": "Optional Python expression to filter batches before computing this metric."
        },
    )


class MetricEngine:
    """Orchestrator for complex metric calculation and aggregation.

    The `MetricEngine` is responsible for coordinating the entire metrics
    pipeline. It manages:
    - `MetricSource`s: Data producers that extract relevant information from
      model outputs and batch inputs.
    - `Metric`s: Stateless logic that defines how raw data from sources is
      transformed into batch-level results.
    - `BaseMeter`s (Accumulators): Stateful components that store and aggregate
      these batch-level results until final computation.

    It ensures lazy execution of sources, handles conditional metric computation
    via `slice_filter`, and orchestrates distributed aggregation and finalization.
    """

    def __init__(
        self,
        group_name: str,
        metrics_configs: list[EngineMetricConfig],
        source_configs: dict[str, dict[str, Any]],
    ):
        """Initializes the MetricEngine.

        Args:
            group_name: The name of the `MeterGroup` that this engine will operate within.
                        All meters managed by this engine will belong to this group.
            metrics_configs: A list of `EngineMetricConfig` objects, each defining
                             a specific `Metric` to be managed by the engine.
            source_configs: A dictionary mapping source names (strings) to their
                            configuration dictionaries. These are used to build
                            `MetricSource` instances.
        """
        self.group_name = group_name
        self._sources: dict[str, MetricSource] = {
            name: build_source(cfg) for name, cfg in source_configs.items()
        }

        self._metrics: dict[str, Metric] = {}
        self._metric_configs: dict[str, EngineMetricConfig] = {}

        # Ensure a 'forward' source exists as it's commonly used.
        if "forward" not in self._sources:
            logger.debug(
                "No 'forward' source configured. Adding default ForwardSource."
            )
            from optimus_dl.modules.metrics.source import ForwardSourceConfig

            self._sources["forward"] = build_source(ForwardSourceConfig())

        # Build all Metric instances based on configurations
        for m_cfg in metrics_configs:
            metric_impl = build_metric(m_cfg.type)
            # The name for the metric logic is taken from its config's _name
            name = m_cfg.type.get("_name") or metric_impl.__class__.__name__
            if name in self._metrics:
                logger.warning(
                    f"Duplicate metric name '{name}' detected. Overwriting existing metric definition."
                )
            self._metrics[name] = metric_impl
            self._metric_configs[name] = m_cfg

        self._source_cache: dict[str, Any] = (
            {}
        )  # Cache for lazy source execution results

    def update(self, model: torch.nn.Module, batch: dict[str, Any]):
        """Runs sources and metrics for a given batch, updating internal accumulators.

        This method is typically called once per training/evaluation step.
        It orchestrates the flow:
        1. Clears the source cache.
        2. Checks if logging is due for the current `MeterGroup`. If not, skips.
        3. For each configured `Metric`:
           a. Applies `slice_filter` if defined.
           b. Lazily executes required `MetricSource`s and caches their results.
           c. Calls the `Metric`'s stateless logic to get batch-level results.
           d. Logs these results to the appropriate `BaseMeter` (accumulator).

        Args:
            model: The PyTorch model, typically in evaluation mode during metric computation.
            batch: The current batch of data (e.g., from a DataLoader).
        """
        self._source_cache.clear()

        # Use the context manager to check if logging should occur
        with metrics_group(self.group_name) as should_log:
            if not should_log:
                return  # Skip if not a logging step

            for name, metric in self._metrics.items():
                m_cfg = self._metric_configs[name]

                # 1. Filter Check: Skip metric computation for this batch if filter fails
                if m_cfg.slice_filter:
                    try:
                        # eval() can be dangerous. Ensure filter expressions are trusted.
                        if not eval(
                            m_cfg.slice_filter,
                            {"batch": batch, "torch": torch, "model": model},
                        ):
                            logger.debug(
                                f"Metric '{name}' skipped for batch due to slice_filter."
                            )
                            continue
                    except Exception as e:
                        logger.error(
                            f"Error evaluating slice_filter for metric '{name}': {e}. Skipping metric for this batch."
                        )
                        continue

                # 2. Gather Data (Lazy execution of sources)
                sources_data: dict[str, Any] = {}
                for source_name in metric.required_sources:
                    if source_name not in self._sources:
                        logger.error(
                            f"Metric '{name}' requires source '{source_name}' but it is not configured. Skipping metric for this batch."
                        )
                        # This should ideally be caught during initialization or validation
                        continue

                    if source_name not in self._source_cache:
                        try:
                            # Execute source and cache result
                            self._source_cache[source_name] = self._sources[
                                source_name
                            ](model, batch)
                        except Exception as e:
                            logger.error(
                                f"Error executing source '{source_name}' for metric '{name}': {e}. Skipping metric for this batch."
                            )
                            # Prevent further processing for this metric if its source fails
                            self._source_cache[source_name] = None  # Mark as failed
                            continue

                    if self._source_cache[source_name] is not None:
                        sources_data[source_name] = self._source_cache[source_name]
                    else:
                        # Source failed in previous step, skip this metric
                        continue

                # If any required source failed or was skipped, skip this metric
                if len(sources_data) != len(metric.required_sources):
                    logger.debug(
                        f"Not all required sources available for metric '{name}'. Skipping for this batch."
                    )
                    continue

                # 3. Compute raw results using the Metric's stateless logic
                try:
                    batch_results = metric(batch, sources_data)
                except Exception as e:
                    logger.error(
                        f"Error computing batch results for metric '{name}': {e}. Skipping logging for this metric."
                    )
                    continue

                # 4. Log to stateful accumulators (Meters)
                # Removed redundant imports here as they are now at the top of the file
                for sub_name, log_kwargs in batch_results.items():
                    # Construct full name: e.g., "Accuracy/top1" or just "Loss"
                    full_name = f"{name}/{sub_name}" if sub_name != name else name

                    acc_type = m_cfg.accumulators.get(sub_name, "average")
                    factory = self._get_accumulator_factory(acc_type)

                    log_metric(
                        name=full_name,
                        meter_factory=factory,  # Changed to meter_factory
                        reset=m_cfg.reset,
                        priority=m_cfg.priority,
                        **log_kwargs,
                    )

    def _get_accumulator_factory(
        self, acc_type: str
    ) -> Callable[[], BaseMeter]:  # Return type changed to BaseMeter
        """Returns a factory function for creating a specific type of accumulator (Meter)."""
        # Re-importing specific meters for local scope
        # (already imported at file-level, but explicit here for clarity in this method's context)
        # from optimus_dl.modules.metrics.common import (
        #     AverageMeter,  # Changed from AverageMetric
        #     SummedMeter,   # Changed from SummedMetric
        #     AveragedExponentMeter,
        # )
        if acc_type == "average":
            return lambda: AverageMeter()  # Changed to AverageMeter
        if acc_type == "sum":
            return lambda: SummedMeter()  # Changed to SummedMeter
        if acc_type == "gather":
            return lambda: GatherMetric()
        if acc_type == "perplexity":
            return lambda: AveragedExponentMeter()
        raise ValueError(f"Unknown accumulator type: {acc_type}")

    def compute(
        self, aggregate: bool = False, collective: Any | None = None
    ) -> dict[str, Any]:
        """Aggregates meter data and runs finalization logic for each metric.

        This method is called to get the final, reportable metric values
        after accumulating data over several batches or an entire epoch.
        It orchestrates:
        1. Global aggregation of meter states (if `aggregate` is True).
        2. Finalization of each `Metric` definition using the aggregated data.

        Args:
            aggregate: If True, meter states are aggregated across distributed
                       ranks before finalization.
            collective: The distributed collective for aggregation, required if
                        `aggregate` is True.

        Returns:
            A dictionary where keys are finalized metric names (e.g., "Accuracy/top1")
            and values are their computed results.
        """
        from optimus_dl.modules.metrics.base import (  # Ensure compute_metrics is imported from base
            compute_metrics,
        )

        # First, compute results from all meters (locally or aggregated)
        raw_results = compute_metrics(
            self.group_name, aggregate=aggregate, collective=collective
        )

        final_report: dict[str, Any] = {}
        for name, metric in self._metrics.items():
            m_cfg = self._metric_configs[name]

            # Collect all relevant meter data for this specific Metric definition
            acc_data: dict[str, Any] = {}
            # Iterate through the expected sub-metrics from the config, or assume the metric itself
            # for simpler metrics that don't emit sub_names
            configured_sub_metrics = m_cfg.accumulators.keys()
            if (
                not configured_sub_metrics and name in raw_results
            ):  # Handle simple metrics without explicit sub_names
                configured_sub_metrics = [name]

            for sub_name in configured_sub_metrics:
                full_name = f"{name}/{sub_name}" if sub_name != name else name
                if full_name in raw_results:
                    acc_data[sub_name] = raw_results[full_name]
                else:
                    logger.warning(
                        f"Accumulated data for '{full_name}' not found for metric '{name}'. Skipping its finalization if required."
                    )

            # Finalize the metric using the collected accumulator data
            try:
                finalized = metric.finalize(acc_data)
            except Exception as e:
                logger.error(
                    f"Error finalizing metric '{name}' with data {acc_data}: {e}. Skipping its report."
                )
                continue

            # Map finalized results back to the report
            for k, v in finalized.items():
                final_report[f"{name}/{k}" if k != name else name] = v

        return final_report
