import pytest
from unittest.mock import MagicMock

from optimus_dl.modules.metrics.source import (
    MetricSource,
    MetricSourceConfig,
    register_source,
)
from optimus_dl.modules.metrics.definitions import (
    Metric,
    MetricConfig,
    register_metric,
)
from optimus_dl.modules.metrics.engine import MetricEngine
from optimus_dl.modules.metrics.base import _meter_groups, _active_meter_groups

from dataclasses import dataclass, field


@dataclass
class ValueSourceConfig(MetricSourceConfig):
    _name: str = "value_source"
    val: int = 1


@register_source("value_source", ValueSourceConfig)
class ValueSource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"value_proto"}

    def __call__(self, model, batch, dependencies):
        # We mock a property to count calls
        model.source_called += 1
        return {"value_proto": self.cfg.val}


@dataclass
class MultiProtocolSourceConfig(MetricSourceConfig):
    _name: str = "multi_proto_source"


@register_source("multi_proto_source", MultiProtocolSourceConfig)
class MultiProtocolSource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"proto_a", "proto_b"}

    def __call__(self, model, batch, dependencies):
        return {"proto_a": 10, "proto_b": 20}


@dataclass
class SimpleMetricConfig(MetricConfig):
    _name: str = "simple_metric"
    required_proto: str = "value_proto"
    accumulators: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.accumulators:
            self.accumulators = {self.required_proto: {"_name": "mean"}}


@register_metric("simple_metric", SimpleMetricConfig)
class SimpleMetric(Metric):
    @property
    def requires(self) -> dict[str, set[str]]:
        return {"input": {self.cfg.required_proto}}

    def __call__(self, batch, sources_data):
        val = sources_data["input"][self.cfg.required_proto]
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
                        "_name": "metric_1",
                        "type": {"_name": "simple_metric", "required_proto": "value_proto"},
                        "role_mapping": {"input": "shared_source"},
                        "accumulators": {"value_proto": "average"}
                    },
                    {
                        "_name": "metric_2",
                        "type": {"_name": "simple_metric", "required_proto": "value_proto"},
                        "role_mapping": {"input": "shared_source"},
                        "accumulators": {"value_proto": "average"}
                    }
                ]
            }
        ]

        engine = MetricEngine("test_group", configs)
        model = MagicMock()
        model.source_called = 0
        batch = {}

        engine.update(model, batch)

        # Source should only be called once, despite being used by two metrics
        assert model.source_called == 1

        from optimus_dl.modules.metrics import compute_metrics
        raw_results = compute_metrics("test_group", aggregate=False)
        results = engine.compute(raw_results)

        assert results["test/simple_metric/value_proto"] == 42.0

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
                        "_name": "metric_a",
                        "type": {"_name": "simple_metric", "required_proto": "proto_a"},
                        "role_mapping": {"input": "provider"},
                        "accumulators": {"proto_a": "average"}
                    },
                    {
                        "_name": "metric_b",
                        "type": {"_name": "simple_metric", "required_proto": "proto_b"},
                        "role_mapping": {"input": "provider"},
                        "accumulators": {"proto_b": "average"}
                    }
                ]
            }
        ]

        engine = MetricEngine("multi_group", configs)
        model = MagicMock()
        batch = {}

        engine.update(model, batch)

        from optimus_dl.modules.metrics import compute_metrics
        raw_results = compute_metrics("multi_group", aggregate=False)
        results = engine.compute(raw_results)

        assert results["multi/simple_metric/proto_a"] == 10.0
        assert results["multi/simple_metric/proto_b"] == 20.0

    def test_handshake_missing_protocol(self):
        """Ensure handshake fails if a source does not provide the explicitly required protocol."""
        configs = [
            {
                "_name": "source_group",
                "prefix": "fail",
                "sources": {
                    "provider": {"_name": "value_source", "val": 1},
                },
                "metrics": [
                    {
                        "_name": "metric_fail",
                        "type": {"_name": "simple_metric", "required_proto": "non_existent_proto"},
                        "role_mapping": {"input": "provider"}
                    }
                ]
            }
        ]

        with pytest.raises(ValueError, match="does not provide required protocols {'non_existent_proto'}"):
            MetricEngine("fail_group", configs)


