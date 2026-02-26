from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from optimus_dl.modules.metrics.base import (
    BaseMeter,
    log_metric,
    metrics_group,
)
from optimus_dl.modules.metrics.common import (
    AveragedExponentMeter,
    AverageMeter,
    GatherMeter,
    SummedMeter,
)
from optimus_dl.modules.metrics.metrics import (
    Metric,
    build_metric,
)
from optimus_dl.modules.metrics.source import (
    MetricSource,
    build_metric_source,
)

logger = logging.getLogger(__name__)


class ParsedGroup:
    """Internal representation of a validated metric group."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.sources: dict[str, MetricSource] = {}
        self.metrics: list[Metric] = []

    @property
    def protocols_to_sources(self):
        protocols_to_sources = defaultdict(list)
        for source_name, source in self.sources.items():
            for protocol in source.provides:
                protocols_to_sources[protocol].append(source_name)
        return {
            protocol: list(sources)
            for protocol, sources in protocols_to_sources.items()
        }


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
                sources_dict = cfg.get("sources", {})
                metrics_list = cfg.get("metrics", [])
            else:
                prefix = ""
                sources_dict = {}
                metrics_list = [cfg]

            group = ParsedGroup(prefix=prefix)

            # 1. Build Sources
            for role, s_cfg in sources_dict.items():
                group.sources[role] = build_metric_source(s_cfg)

            # 2. Validate Source Dependencies (Internal Handshake)
            for source in group.sources.values():
                self._validate_handshake(
                    component_requires=source.requires,
                    available_sources=group.sources,
                )

            # 3. Build Metrics and perform Protocol Handshake
            for m_dict in metrics_list:
                metric_impl = build_metric(m_dict)

                self._validate_handshake(
                    component_requires=metric_impl.requires,
                    available_sources=group.sources,
                )

                group.metrics.append(metric_impl)

            self.groups.append(group)

    def _validate_handshake(
        self,
        component_requires: set[str],
        available_sources: dict[str, MetricSource],
    ):
        """Validates that all dependencies and protocols for a component are met."""
        available_protocols = set()
        for source in available_sources.values():
            available_protocols |= source.provides

        missing = component_requires - available_protocols
        if missing:
            logger.debug(
                f"Handshake missing protocols: {missing}. These must be provided via 'computed_data' during update."
            )

    @property
    def required_external_protocols(self) -> set[str]:
        """Returns the set of protocols required by metrics but not provided by internal sources."""
        external = set()
        for group in self.groups:
            internal_provides = set()
            for source in group.sources.values():
                internal_provides |= source.provides

            for metric in group.metrics:
                external |= metric.requires - internal_provides
        return external

    def _eval_source(
        self,
        group: ParsedGroup,
        source_name: str,
        data: dict[str, Any],
        global_cache: dict[str, dict[str, Any]],
        _evaluating: set[str] | None = None,
    ) -> dict[str, Any]:
        """Recursively evaluates a source and its dependencies, using a global cross-group cache.

        Args:
            group: The ParsedGroup containing the source.
            source_name: The name of the source to evaluate within the group.
            data: The input data to pass to the source.
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
                f"Cyclic dependency detected for source '{source_name}' in group '{group.prefix}'. {_evaluating = }"
            )

        _evaluating.add(source_name)

        protocols_to_sources = group.protocols_to_sources

        try:
            # Evaluate dependencies first
            deps_data: dict[str, dict[str, Any]] = {}

            for req_protocol in source.requires:
                try:
                    providers = protocols_to_sources[req_protocol]
                    if len(providers) == 0:
                        raise ValueError(
                            f"No source provides the required protocol {req_protocol}"
                        )

                    provider = providers[0]

                    deps_data[req_protocol] = self._eval_source(
                        group=group,
                        source_name=provider,
                        data=data,
                        global_cache=global_cache,
                        _evaluating=_evaluating,
                    )
                except Exception as e:
                    # If a dependency fails, mark this as failed too
                    global_cache[h] = e
                    raise e

            # Evaluate this source
            try:
                result = source(deps_data, **data)
                global_cache[h] = result
                return result
            except Exception as e:
                global_cache[h] = e
                raise e
        finally:
            _evaluating.remove(source_name)

    def update(self, data: dict[str, Any], computed_data: dict[str, Any] | None = None):
        """Runs sources and metrics for a given batch.

        Args:
            data: Data dictionary containing inputs for sources (model, batch).
            computed_data: Optional dictionary mapping protocol names to already computed data.
                This allows reusing results (like logits) to avoid redundant forward passes.
        """
        with metrics_group(self.group_name, force_recreate=False) as should_log:
            if not should_log:
                return
            # Global cache for the entire batch. Keys are source config hashes.
            global_source_cache: dict[str, Any] = {}

            # Seed cache with computed data if provided
            computed_data = computed_data or {}

            for group in self.groups:
                protocols_to_sources = group.protocols_to_sources

                for i, metric in enumerate(group.metrics):
                    metric_name = metric.nested_name or getattr(
                        metric.cfg, "_name", f"metric_{i}"
                    )

                    sources_data: dict[str, dict[str, Any]] = {}
                    execution_failed = False

                    for req_protocol in metric.requires:
                        # 1. Try precomputed data first
                        if req_protocol in computed_data:
                            sources_data[req_protocol] = {
                                req_protocol: computed_data[req_protocol]
                            }
                            continue

                        # 2. Fallback to source evaluation
                        try:
                            providers = protocols_to_sources[req_protocol]
                            if len(providers) == 0:
                                raise ValueError(
                                    f"No source provides the required protocol {req_protocol}"
                                )

                            provider = providers[0]
                            sources_data[req_protocol] = self._eval_source(
                                group=group,
                                source_name=provider,
                                data=data,
                                global_cache=global_source_cache,
                            )
                        except Exception as e:
                            logger.error(
                                f"Source execution failed for the metric {metric} in group '{group.prefix}': {e}"
                            )
                            execution_failed = True
                            break

                    if execution_failed:
                        continue

                    try:
                        batch_results = metric(sources_data)
                    except Exception as e:
                        logger.error(
                            f"Metric computation failed for '{metric_name}' in group '{group.prefix}': {e}"
                        )
                        continue

                    for sub_name, log_kwargs in batch_results.items():
                        is_internal = sub_name.startswith("_")
                        base_name = (
                            f"{metric_name}/{sub_name}"
                            if metric_name != sub_name
                            else metric_name
                        )
                        full_name = (
                            f"{group.prefix}/{base_name}" if group.prefix else base_name
                        )

                        if is_internal:
                            full_name = f"_internal/{full_name}"

                        acc_type = metric.accumulators.get(sub_name)
                        if acc_type is None:
                            logger.warning(
                                f"No accumulator defined for sub-metric '{sub_name}' in metric '{metric_name}'. Skipping."
                            )
                            continue

                        factory = self._get_accumulator_factory(acc_type)

                        log_metric(
                            name=full_name,
                            meter_factory=factory,
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
            for i, metric in enumerate(group.metrics):
                metric_name = metric.nested_name or getattr(
                    metric.cfg, "_name", f"metric_{i}"
                )

                acc_data: dict[str, Any] = {}
                # Use metric.accumulators if provided, otherwise fallback to metric's defaults
                expected_keys = metric.accumulators.keys()

                for sub_name in expected_keys:
                    is_internal = sub_name.startswith("_")
                    base_name = (
                        f"{metric_name}/{sub_name}"
                        if metric_name != sub_name
                        else metric_name
                    )
                    full_name = (
                        f"{group.prefix}/{base_name}" if group.prefix else base_name
                    )

                    if is_internal:
                        full_name = f"_internal/{full_name}"

                    if full_name in raw_results:
                        acc_data[sub_name] = raw_results.pop(full_name)

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
                    if k.startswith("_"):
                        continue

                    base_name = (
                        f"{metric_name}/{k}" if k != metric_name else metric_name
                    )
                    full_name = (
                        f"{group.prefix}/{base_name}" if group.prefix else base_name
                    )
                    final_report[full_name] = v

        return raw_results | final_report
