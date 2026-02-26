from dataclasses import (
    dataclass,
)
from unittest.mock import MagicMock

from optimus_dl.modules.metrics.base import (
    _active_meter_groups,
    _meter_groups,
)
from optimus_dl.modules.metrics.engine import MetricEngine
from optimus_dl.modules.metrics.metrics import (
    Metric,
    MetricConfig,
    register_metric,
)
from optimus_dl.modules.metrics.source import (
    MetricSource,
    MetricSourceConfig,
    register_metric_source,
)


@dataclass
class ValueSourceConfig(MetricSourceConfig):
    _name: str = "value_source"
    val: int = 1


@register_metric_source("value_source", ValueSourceConfig)
class ValueSource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"value_proto"}

    def __call__(self, dependencies, **kwargs):
        model = kwargs["model"]
        # We mock a property to count calls
        model.source_called += 1
        return {"value_proto": self.cfg.val}


@dataclass
class MultiProtocolSourceConfig(MetricSourceConfig):
    _name: str = "multi_proto_source"


@register_metric_source("multi_proto_source", MultiProtocolSourceConfig)
class MultiProtocolSource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"proto_a", "proto_b"}

    def __call__(self, dependencies, **kwargs):
        return {"proto_a": 10, "proto_b": 20}


@dataclass
class SimpleMetricConfig(MetricConfig):
    _name: str = "simple_metric"
    required_proto: str = "value_proto"


@register_metric("simple_metric", SimpleMetricConfig)
class SimpleMetric(Metric):
    @property
    def requires(self) -> set[str]:
        return {self.cfg.required_proto}

    @property
    def accumulators(self) -> dict[str, str]:
        return {self.cfg.required_proto: "average"}

    def __call__(self, sources_data):
        val = sources_data[self.cfg.required_proto][self.cfg.required_proto]
        return {self.cfg.required_proto: {"value": val, "weight": 1.0}}


class TestMetricEngineExtended:
    def setup_method(self):
        _meter_groups.clear()
        _active_meter_groups.clear()

    def test_source_evaluated_once_per_batch(self):
        """Ensure a source used by multiple metrics is only evaluated once."""
        configs = [
            {
                "_name": "source_group",
                "prefix": "test",
                "sources": {
                    "shared_source": {"_name": "value_source", "val": 42},
                },
                "metrics": [
                    {
                        "_name": "simple_metric",
                        "required_proto": "value_proto",
                        "nested_name": "m1",
                    },
                    {
                        "_name": "simple_metric",
                        "required_proto": "value_proto",
                        "nested_name": "m2",
                    },
                ],
            }
        ]

        engine = MetricEngine("test_group", configs)
        model = MagicMock()
        model.source_called = 0
        batch = {}

        engine.update({"model": model, "batch": batch})

        # Source should only be called once, despite being used by two metrics
        assert model.source_called == 1

        from optimus_dl.modules.metrics import compute_metrics

        raw_results = compute_metrics("test_group", aggregate=False)
        results = engine.compute(raw_results)

        # Descriptive naming with nested_name: prefix / nested_name / sub_name
        # Since nested_name is provided, it is used as is.
        assert results["test/m1/value_proto"] == 42.0
        assert results["test/m2/value_proto"] == 42.0

    def test_multi_protocol_source(self):
        """Ensure a source can provide multiple protocols and metrics can consume them selectively."""
        configs = [
            {
                "_name": "source_group",
                "prefix": "multi",
                "sources": {
                    "provider": {"_name": "multi_proto_source"},
                },
                "metrics": [
                    {
                        "_name": "simple_metric",
                        "required_proto": "proto_a",
                        "nested_name": "ma",
                    },
                    {
                        "_name": "simple_metric",
                        "required_proto": "proto_b",
                        "nested_name": "mb",
                    },
                ],
            }
        ]

        engine = MetricEngine("multi_group", configs)
        model = MagicMock()
        batch = {}

        engine.update({"model": model, "batch": batch})

        from optimus_dl.modules.metrics import compute_metrics

        raw_results = compute_metrics("multi_group", aggregate=False)
        results = engine.compute(raw_results)

        assert results["multi/ma/proto_a"] == 10.0
        assert results["multi/mb/proto_b"] == 20.0

    def test_handshake_missing_protocol(self):
        """Ensure missing protocols are identified via required_external_protocols."""
        configs = [
            {
                "_name": "source_group",
                "prefix": "fail",
                "sources": {
                    "provider": {"_name": "value_source", "val": 1},
                },
                "metrics": [
                    {
                        "_name": "simple_metric",
                        "required_proto": "non_existent_proto",
                    }
                ],
            }
        ]

        # Init no longer raises ValueError, but identifies external requirements
        engine = MetricEngine("fail_group", configs)
        assert "non_existent_proto" in engine.required_external_protocols
