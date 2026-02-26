from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
)


from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.metrics.base import (
    compute_metrics,
    metrics_group,
)
from optimus_dl.modules.metrics.engine import MetricEngine
from optimus_dl.modules.metrics.metrics import (
    Metric,
    register_metric,
)
from optimus_dl.modules.metrics.source import (
    MetricSource,
    register_metric_source,
)


@dataclass
class ExternalTestMetricConfig(RegistryConfigStrict):
    nested_name: str | None = None


@register_metric("external_test_metric", ExternalTestMetricConfig)
class ExternalTestMetric(Metric):
    """Metric used for testing external protocol requirements."""

    @property
    def requires(self) -> set[str]:
        return {"ext_proto_1", "ext_proto_2", "int_proto"}

    @property
    def accumulators(self) -> dict[str, str]:
        return {"val": "average"}

    def __call__(
        self, sources_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        return {"val": {"value": 1.0, "weight": 1.0}}


@dataclass
class ExternalDataSourceMetricConfig(RegistryConfigStrict):
    nested_name: str | None = None


@register_metric("ext_data_source_metric", ExternalDataSourceMetricConfig)
class ExternalDataSourceMetric(Metric):
    """Metric that consumes external data provided during update."""

    @property
    def requires(self) -> set[str]:
        return {"ext_data"}

    @property
    def accumulators(self) -> dict[str, str]:
        return {"val": "average"}

    def __call__(
        self, sources_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        # The engine wraps injected data in a dict mapping protocol to its value
        data = sources_data.get("ext_data", {})
        val = data.get("ext_data", 0.0)
        return {"val": {"value": float(val), "weight": 1.0}}


def test_required_external_protocols():
    """Test that MetricEngine correctly identifies missing internal protocols."""

    @dataclass
    class SimpleSourceConfig(RegistryConfigStrict):
        pass

    @register_metric_source("simple_internal_source", SimpleSourceConfig)
    class SimpleSource(MetricSource):
        @property
        def provides(self) -> set[str]:
            return {"int_proto"}

        def __call__(self, dependencies, **kwargs):
            return {"int_proto": 1}

    configs = [
        {
            "_name": "source_group",
            "prefix": "test",
            "sources": {"s1": {"_name": "simple_internal_source"}},
            "metrics": [{"_name": "external_test_metric"}],
        }
    ]

    engine = MetricEngine("test_engine_handshake", configs)

    # external_test_metric requires {"ext_proto_1", "ext_proto_2", "int_proto"}
    # simple_internal_source provides {"int_proto"}
    # So required_external_protocols should be {"ext_proto_1", "ext_proto_2"}

    external = engine.required_external_protocols
    assert "ext_proto_1" in external
    assert "ext_proto_2" in external
    assert "int_proto" not in external
    assert len(external) == 2


def test_engine_update_with_external_data_injection():
    """Test that data provided via 'computed_data' is correctly consumed by metrics."""
    configs = [{"_name": "ext_data_source_metric"}]

    engine = MetricEngine("test_engine_injection", configs)

    with metrics_group("test_engine_injection", force_recreate=True):
        # Inject 'ext_data' which is required by 'ext_data_source_metric'
        engine.update(data={}, computed_data={"ext_data": 42.0})

    results = compute_metrics("test_engine_injection")

    # Metric naming should now be descriptive: 'ext_data_source_metric/val'
    # because 'ext_data_source_metric' != 'val'
    assert "ext_data_source_metric/val" in results
    assert results["ext_data_source_metric/val"] == 42.0
