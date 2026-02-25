from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import torch

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.metrics.base import (
    BaseMeter,
    log_metric,
    metrics_group,
)
from optimus_dl.modules.metrics.common import (
    AveragedExponentMeter,
    AverageMeter,
    SummedMeter,
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


class GatherMeter(BaseMeter):
    """Accumulator that gathers all raw values across the entire dataset.

    Use this for meters that require full dataset context (e.g., BLEU, ROC-AUC).
    """

    def __init__(self):
        """Initializes the GatherMeter with an empty list to store values."""
        self.values: list[Any] = []

    def log(self, value: Any):
        """Logs a single value to be gathered."""
        self.values.append(value)

    def compute(self) -> list[Any]:
        """Returns the list of all gathered values."""
        return self.values

    def merge(self, other_state: dict[str, Any]):
        """Merges the state from another GatherMeter instance."""
        self.values.extend(other_state["values"])

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the GatherMeter for checkpointing."""
        return {"values": self.values}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restores the state of the GatherMeter from a state dictionary."""
        self.values = state_dict["values"]


@dataclass
class EngineMetricConfig(RegistryConfigStrict):
    """Configuration for a single Metric within a group."""

    type: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Config for the Metric (e.g., AccuracyMetricConfig)."},
    )
    role_mapping: dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Maps metric role requirements to source names in the group."
        },
    )
    reset: bool = field(
        default=True,
        metadata={"help": "Whether to reset the meters after each logging step."},
    )
    priority: int = field(
        default=100,
        metadata={"help": "Priority for ordering meters in logs (lower is higher)."},
    )
    slice_filter: str | None = field(
        default=None,
        metadata={"help": "Optional Python expression to filter batches."},
    )


class ParsedGroup:
    """Internal representation of a validated metric group."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.sources: dict[str, MetricSource] = {}
        self.metrics: list[tuple[Metric, EngineMetricConfig]] = []


class MetricEngine:
    """Orchestrator for complex metric calculation and aggregation.

    The `MetricEngine` coordinates `MetricSource`s (data producers) and
    `Metric`s (stateless compute logic). It handles:
    - Protocol Handshake: Ensuring sources provide the data metrics require.
    - Lazy Evaluation: Executing each source at most once per batch across all groups via hash caching.
    - Grouping: Allowing the same metric to run over different sources.
    - Source Dependencies: Allowing sources to depend on other sources in the group.
    """

    def __init__(
        self,
        group_name: str,
        configs: list[dict[str, Any]],
    ):
        """Initializes the MetricEngine.

        Args:
            group_name: The name of the `MeterGroup` (logging namespace).
            configs: A list of configuration dicts.
        """
        self.group_name = group_name
        self.groups: list[ParsedGroup] = []

        self._parse_and_validate_configs(configs)

    def _parse_and_validate_configs(self, configs: list[dict[str, Any]]):
        """Parses configurations, builds sources/metrics, and performs handshakes."""
        for idx, cfg in enumerate(configs):
            if cfg.get("_name") == "source_group":
                prefix = cfg.get("prefix", f"group_{idx}")
                sources_dict = cfg.get("sources", {"default": {"_name": "causal_lm"}})
                metrics_list = cfg.get("metrics", [])
            else:
                prefix = "default"
                sources_dict = {"default": {"_name": "causal_lm"}}
                metrics_list = [cfg]

            group = ParsedGroup(prefix=prefix)

            # 1. Build Sources
            for role, s_cfg in sources_dict.items():
                group.sources[role] = build_source(s_cfg)

            # 2. Validate Source Dependencies (Internal Handshake)
            for s_name, source in group.sources.items():
                self._validate_handshake(
                    component_name=s_name,
                    component_requires=source.requires,
                    available_sources=group.sources,
                    role_mapping=getattr(source.cfg, "dependencies", {}),
                    group_prefix=prefix,
                    component_type="Source",
                )

            # 3. Build Metrics and perform Protocol Handshake
            for m_dict in metrics_list:
                if (
                    "_name" not in m_dict
                    and "type" in m_dict
                    and "_name" in m_dict["type"]
                ):
                    m_dict["_name"] = m_dict["type"]["_name"]
                elif "_name" not in m_dict:
                    m_dict["_name"] = "unnamed_metric"

                m_cfg = EngineMetricConfig(**m_dict)
                metric_impl = build_metric(m_cfg.type)

                self._validate_handshake(
                    component_name=m_cfg.type.get("_name", "unnamed_metric"),
                    component_requires=metric_impl.requires,
                    available_sources=group.sources,
                    role_mapping=m_cfg.role_mapping,
                    group_prefix=prefix,
                    component_type="Metric",
                )

                group.metrics.append((metric_impl, m_cfg))

            self.groups.append(group)

    def _validate_handshake(
        self,
        component_name: str,
        component_requires: dict[str, set[str]],
        available_sources: dict[str, MetricSource],
        role_mapping: dict[str, str],
        group_prefix: str,
        component_type: str,
    ):
        """Validates that all dependencies and protocols for a component are met."""
        for req_role, req_protocols in component_requires.items():
            mapped_source_name = role_mapping.get(req_role, req_role)

            if mapped_source_name not in available_sources:
                raise ValueError(
                    f"{component_type} '{component_name}' requires role '{req_role}' mapped to "
                    f"'{mapped_source_name}', but it is not provided in group '{group_prefix}'."
                )

            provided_protocols = available_sources[mapped_source_name].provides
            missing = req_protocols - provided_protocols
            if missing:
                raise ValueError(
                    f"{component_type} handshake failed in group '{group_prefix}'. Source '{mapped_source_name}' "
                    f"does not provide required protocols {missing} for {component_type.lower()} '{component_name}'."
                )

    def _eval_source(
        self,
        group: ParsedGroup,
        source_name: str,
        model: torch.nn.Module,
        batch: dict[str, Any],
        global_cache: dict[str, dict[str, Any]],
        _evaluating: set[str] | None = None,
    ) -> dict[str, Any]:
        """Recursively evaluates a source and its dependencies, using a global cross-group cache.

        Args:
            group: The ParsedGroup containing the source.
            source_name: The name of the source to evaluate within the group.
            model: The PyTorch model being evaluated.
            batch: The current batch of data.
            global_cache: A dictionary caching results across all groups to prevent redundant computation.
            _evaluating: Internal set used for cycle detection during recursive evaluation.

        Returns:
            A dictionary containing the source's provided protocols and their corresponding data.

        Raises:
            RuntimeError: If a cyclic dependency is detected between sources.
            Exception: Re-raises any exception that occurs during source execution.
        """
        source = group.sources[source_name]
        h = source.config_hash

        # Cross-group cache hit
        if h in global_cache:
            if isinstance(global_cache[h], Exception):
                raise global_cache[h]
            return global_cache[h]

        if _evaluating is None:
            _evaluating = set()

        if source_name in _evaluating:
            raise RuntimeError(
                f"Cyclic dependency detected for source '{source_name}' in group '{group.prefix}'."
            )

        _evaluating.add(source_name)

        try:
            # Evaluate dependencies first
            deps_data: dict[str, dict[str, Any]] = {}
            for req_role in source.requires:
                mapped_dep_name = getattr(source.cfg, "dependencies", {}).get(
                    req_role, req_role
                )
                try:
                    deps_data[req_role] = self._eval_source(
                        group, mapped_dep_name, model, batch, global_cache, _evaluating
                    )
                except Exception as e:
                    # If a dependency fails, mark this as failed too
                    global_cache[h] = e
                    raise e

            # Evaluate this source
            try:
                result = source(model, batch, deps_data)
                global_cache[h] = result
                return result
            except Exception as e:
                global_cache[h] = e
                raise e
        finally:
            _evaluating.remove(source_name)

    def update(self, model: torch.nn.Module, batch: dict[str, Any]):
        """Runs sources and metrics for a given batch.

        Args:
            model: PyTorch model.
            batch: Data dictionary.
        """
        with metrics_group(self.group_name) as should_log:
            if not should_log:
                return

            # Global cache for the entire batch. Keys are source config hashes.
            global_source_cache: dict[str, Any] = {}

            for group in self.groups:
                for metric, m_cfg in group.metrics:
                    metric_name = m_cfg.type.get("_name") or metric.__class__.__name__

                    if m_cfg.slice_filter:
                        try:
                            if not eval(
                                m_cfg.slice_filter,
                                {"batch": batch, "torch": torch, "model": model},
                            ):
                                continue
                        except Exception as e:
                            logger.error(
                                f"Filter evaluation failed for '{metric_name}' in group '{group.prefix}': {e}"
                            )
                            continue

                    sources_data: dict[str, dict[str, Any]] = {}
                    execution_failed = False

                    for req_role in metric.requires:
                        mapped_source_name = m_cfg.role_mapping.get(req_role, req_role)
                        try:
                            sources_data[req_role] = self._eval_source(
                                group,
                                mapped_source_name,
                                model,
                                batch,
                                global_source_cache,
                            )
                        except Exception as e:
                            logger.error(
                                f"Source execution failed for role '{req_role}' "
                                f"(mapped to '{mapped_source_name}') in group '{group.prefix}': {e}"
                            )
                            execution_failed = True
                            break

                    if execution_failed:
                        continue

                    try:
                        batch_results = metric(batch, sources_data)
                    except Exception as e:
                        logger.error(
                            f"Metric computation failed for '{metric_name}' in group '{group.prefix}': {e}"
                        )
                        continue

                    for sub_name, log_kwargs in batch_results.items():
                        base_name = (
                            f"{metric_name}/{sub_name}"
                            if sub_name != metric_name
                            else metric_name
                        )
                        full_name = (
                            f"{group.prefix}/{base_name}"
                            if group.prefix != "default"
                            else base_name
                        )

                        acc_type = m_cfg.accumulators.get(
                            sub_name
                        ) or metric.accumulators.get(sub_name, "average")
                        factory = self._get_accumulator_factory(acc_type)

                        log_metric(
                            name=full_name,
                            meter_factory=factory,
                            reset=m_cfg.reset,
                            priority=m_cfg.priority,
                            **log_kwargs,
                        )

    def _get_accumulator_factory(self, acc_type: str) -> Callable[[], BaseMeter]:
        if acc_type == "average":
            return lambda: AverageMeter()
        if acc_type == "sum":
            return lambda: SummedMeter()
        if acc_type == "gather":
            return lambda: GatherMeter()
        if acc_type == "perplexity":
            return lambda: AveragedExponentMeter()
        raise ValueError(f"Unknown accumulator type: {acc_type}")

    def compute(self, raw_results: dict[str, Any]) -> dict[str, Any]:
        """Runs finalization logic for each metric based on raw computed results."""
        final_report: dict[str, Any] = {}

        for group in self.groups:
            for metric, m_cfg in group.metrics:
                metric_name = m_cfg.type.get("_name") or metric.__class__.__name__

                acc_data: dict[str, Any] = {}
                # Use m_cfg.accumulators if provided, otherwise fallback to metric's defaults
                configured_sub_metrics = (
                    m_cfg.accumulators.keys()
                    if m_cfg.accumulators
                    else metric.accumulators.keys()
                )
                expected_keys = (
                    configured_sub_metrics if configured_sub_metrics else [metric_name]
                )

                for sub_name in expected_keys:
                    base_name = (
                        f"{metric_name}/{sub_name}"
                        if sub_name != metric_name
                        else metric_name
                    )
                    full_name = (
                        f"{group.prefix}/{base_name}"
                        if group.prefix != "default"
                        else base_name
                    )

                    if full_name in raw_results:
                        acc_data[sub_name] = raw_results[full_name]

                if not acc_data:
                    continue

                try:
                    finalized = metric.finalize(acc_data)
                except Exception as e:
                    logger.error(
                        f"Metric finalization failed for '{metric_name}' in group '{group.prefix}': {e}"
                    )
                    continue

                for k, v in finalized.items():
                    base_name = (
                        f"{metric_name}/{k}" if k != metric_name else metric_name
                    )
                    full_name = (
                        f"{group.prefix}/{base_name}"
                        if group.prefix != "default"
                        else base_name
                    )
                    final_report[full_name] = v

        return final_report
