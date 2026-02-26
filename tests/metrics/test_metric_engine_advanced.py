from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

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
from optimus_dl.modules.metrics.sources import (
    CausalLMSource,
    CausalLMSourceConfig,
)


@dataclass
class ForwardSourceConfig(MetricSourceConfig):
    _name: str = "forward"


@register_metric_source("forward")
class ForwardSource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"logits"}

    def __call__(self, dependencies, **kwargs):
        return {"logits": torch.tensor([1.0])}


@dataclass
class DummySourceConfig(MetricSourceConfig):
    _name: str = "dummy_source"
    val: int = 1


@register_metric_source("dummy_source", DummySourceConfig)
class DummySource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"dummy_proto"}

    def __call__(self, dependencies, **kwargs):
        return {"dummy_proto": self.cfg.val}


@dataclass
class DependentSourceConfig(MetricSourceConfig):
    _name: str = "dependent_source"
    multiplier: int = 2


@register_metric_source("dependent_source", DependentSourceConfig)
class DependentSource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"dep_proto"}

    @property
    def requires(self) -> set[str]:
        return {"dummy_proto"}

    def __call__(self, dependencies, **kwargs):
        base_val = dependencies["dummy_proto"]["dummy_proto"]
        return {"dep_proto": base_val * self.cfg.multiplier}


@dataclass
class DummyMetricConfig(MetricConfig):
    _name: str = "dummy_metric"


@register_metric("dummy_metric", DummyMetricConfig)
class DummyMetric(Metric):
    @property
    def requires(self) -> set[str]:
        return {"dep_proto"}

    def __call__(self, sources_data):
        val = sources_data["dep_proto"]["dep_proto"]
        return {"dummy_metric": {"value": val, "weight": 1.0}}


class TestMetricEngineAdvanced:
    def setup_method(self):
        _meter_groups.clear()
        _active_meter_groups.clear()

    def test_source_config_hashing(self):
        cfg1 = ForwardSourceConfig()
        cfg2 = ForwardSourceConfig()
        cfg3 = CausalLMSourceConfig()

        src1 = ForwardSource(cfg1)
        src2 = ForwardSource(cfg2)
        src3 = CausalLMSource(cfg3)

        assert src1.config_hash == src2.config_hash
        assert src1.config_hash != src3.config_hash

    def test_dependency_resolution(self):
        configs = [
            {
                "_name": "source_group",
                "prefix": "test",
                "sources": {
                    "provider": {"_name": "dummy_source", "val": 5},
                    "consumer": {
                        "_name": "dependent_source",
                        "multiplier": 3,
                    },
                },
                "metrics": [{"_name": "dummy_metric"}],
            }
        ]

        engine = MetricEngine("test_group", configs)
        model = MagicMock()
        batch = {}

        # Run update
        engine.update({"model": model, "batch": batch})

        # Compute results
        from optimus_dl.modules.metrics import compute_metrics

        raw_results = compute_metrics("test_group", aggregate=False)
        results = engine.compute(raw_results)

        # 5 * 3 = 15
        # Descriptive naming: 'test/dummy_metric'
        assert "test/dummy_metric" in results
        assert results["test/dummy_metric"] == 15.0

    def test_cross_group_caching(self):
        # We will use a mock model to count forward passes
        model = MagicMock()
        model.return_value = {
            "logits": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        }

        configs = [
            {
                "_name": "source_group",
                "prefix": "group1",
                "sources": {"default": {"_name": "causal_lm"}},
                "metrics": [{"_name": "accuracy"}],
            },
            {
                "_name": "source_group",
                "prefix": "group2",
                "sources": {"default": {"_name": "causal_lm"}},
                "metrics": [{"_name": "accuracy"}],
            },
        ]

        engine = MetricEngine("test_group", configs)

        # Provide labels to avoid None comparison in AccuracyMetric
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]]),
        }
        engine.update({"model": model, "batch": batch})

        # Even though there are two groups, they use the exact same source config.
        # The model should only be called once per batch.
        assert model.call_count == 1

    def test_accuracy_metric_causal_lm(self):
        configs = [
            {
                "_name": "source_group",
                "prefix": "",
                "sources": {"default": {"_name": "causal_lm"}},
                "metrics": [{"_name": "accuracy"}],
            }
        ]

        engine = MetricEngine("test_group", configs)

        # Batch: 1 sequence, length 3. Labels are [2, 3] (shifted)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[2, 3]]),
        }

        # Model predicts 2 for the first token (correct), 4 for the second (incorrect)
        # CausalLMSource returns logits for length-1 sequence if we shift labels.
        # Let's mock model to return logits already matching labels length.
        logits = torch.zeros(1, 2, 10)
        logits[0, 0, 2] = 10.0  # Pos 0 -> Target 2
        logits[0, 1, 4] = 10.0  # Pos 1 -> Target 3 (so incorrect)

        model = MagicMock()
        model.return_value = {"logits": logits}

        engine.update({"model": model, "batch": batch})

        from optimus_dl.modules.metrics import compute_metrics

        raw_results = compute_metrics("test_group", aggregate=False)
        results = engine.compute(raw_results)

        # Accuracy should be 0.5 (1/2)
        # Descriptive naming: 'accuracy' (no prefix, _name == accuracy == sub_name)
        assert "accuracy" in results
        assert results["accuracy"] == 0.5

    def test_cyclic_dependency_detection(self, caplog):
        @dataclass
        class CycleSourceConfig(MetricSourceConfig):
            _name: str = "cycle_source"

        @register_metric_source("cycle_source", CycleSourceConfig)
        class CycleSource(MetricSource):
            @property
            def provides(self) -> set[str]:
                return {"cycle_proto"}

            @property
            def requires(self) -> set[str]:
                return {"cycle_proto"}

            def __call__(self, dependencies, **kwargs):
                return {"cycle_proto": 1}

        configs_cycle = [
            {
                "_name": "source_group",
                "prefix": "test_cycle",
                "sources": {
                    "source_a": {"_name": "cycle_source"},
                },
                "metrics": [{"_name": "cycle_metric"}],
            }
        ]

        @dataclass
        class CycleMetricConfig(MetricConfig):
            _name: str = "cycle_metric"

        @register_metric("cycle_metric", CycleMetricConfig)
        class CycleMetric(Metric):
            @property
            def requires(self) -> set[str]:
                return {"cycle_proto"}

            def __call__(self, sources_data):
                return {"cycle_metric": {"value": 1.0, "weight": 1.0}}

        engine = MetricEngine("test_group", configs_cycle)

        model = MagicMock()
        batch = {}

        # update should log the cyclic error and skip metric computation
        engine.update({"model": model, "batch": batch})

        assert "Cyclic dependency detected for source" in caplog.text

    def test_missing_protocol(self):
        configs = [
            {
                "_name": "source_group",
                "prefix": "test",
                "sources": {
                    "provider": {"_name": "dummy_source", "val": 5},
                },
                "metrics": [{"_name": "dummy_metric"}],  # Requires dep_proto
            }
        ]

        # Use required_external_protocols check instead of just raises ValueError on init
        # because we changed init to only debug log missing protocols.
        engine = MetricEngine("test_group", configs)
        assert "dep_proto" in engine.required_external_protocols

    def test_metric_finalize(self):
        """Test metrics that use the finalize method for post-aggregation logic."""

        @dataclass
        class DataValueSourceConfig(MetricSourceConfig):
            _name: str = "data_value_source"

        @register_metric_source("data_value_source", DataValueSourceConfig)
        class DataValueSource(MetricSource):
            @property
            def provides(self) -> set[str]:
                return {"val_proto"}

            def __call__(self, dependencies, **kwargs):
                # Return value from batch if present, else 1
                batch = kwargs.get("batch", {})
                return {"val_proto": batch.get("val", 1.0)}

        @dataclass
        class FinalizeMetricConfig(MetricConfig):
            _name: str = "finalize_metric"

        @register_metric("finalize_metric", FinalizeMetricConfig)
        class FinalizeMetric(Metric):
            @property
            def requires(self) -> set[str]:
                return {"val_proto"}

            @property
            def accumulators(self) -> dict[str, str]:
                return {"sum_a": "sum", "sum_b": "sum"}

            def __call__(self, sources_data):
                val = sources_data["val_proto"]["val_proto"]
                return {"sum_a": {"value": val}, "sum_b": {"value": 1.0}}

            def finalize(self, aggregated_data):
                a = aggregated_data["sum_a"]
                b = aggregated_data["sum_b"]
                return {"ratio": a / b if b != 0 else 0.0}

        configs = [
            {
                "_name": "source_group",
                "prefix": "test",
                "sources": {"provider": {"_name": "data_value_source"}},
                "metrics": [{"_name": "finalize_metric"}],
            }
        ]

        engine = MetricEngine("finalize_group", configs)
        model = MagicMock()

        # Batch 1: val=10 -> sum_a=10, sum_b=1
        engine.update({"model": model, "batch": {"val": 10.0}})
        # Batch 2: val=20 -> sum_a=30, sum_b=2
        engine.update({"model": model, "batch": {"val": 20.0}})

        from optimus_dl.modules.metrics import compute_metrics

        raw_results = compute_metrics("finalize_group", aggregate=False)

        # Verify raw results contain the sums. Descriptive naming: 'test/finalize_metric/sum_a'
        assert "test/finalize_metric/sum_a" in raw_results
        assert raw_results["test/finalize_metric/sum_a"] == 30.0
        assert "test/finalize_metric/sum_b" in raw_results
        assert raw_results["test/finalize_metric/sum_b"] == 2.0

        # Run engine compute
        final_results = engine.compute(raw_results)

        # Verify finalized ratio: 30 / 2 = 15
        assert "test/finalize_metric/ratio" in final_results
        assert final_results["test/finalize_metric/ratio"] == 15.0
        # sum_a and sum_b should be gone as they were consumed by compute
        assert "test/finalize_metric/sum_a" not in final_results

    def test_internal_metrics(self):
        """Test that metrics starting with _ are handled as internal."""

        @dataclass
        class InternalMetricConfig(MetricConfig):
            _name: str = "internal_metric"

        @register_metric("internal_metric", InternalMetricConfig)
        class InternalMetric(Metric):
            @property
            def requires(self) -> set[str]:
                return {"dummy_proto"}

            @property
            def accumulators(self) -> dict[str, str]:
                return {"public": "average", "_internal": "average"}

            def __call__(self, sources_data):
                val = sources_data["dummy_proto"]["dummy_proto"]
                return {
                    "public": {"value": val, "weight": 1.0},
                    "_internal": {"value": val * 2, "weight": 1.0},
                }

        configs = [
            {
                "_name": "source_group",
                "prefix": "test",
                "sources": {"provider": {"_name": "dummy_source", "val": 5}},
                "metrics": [{"_name": "internal_metric"}],
            }
        ]

        engine = MetricEngine("internal_group", configs)
        model = MagicMock()
        engine.update({"model": model, "batch": {}})

        from optimus_dl.modules.metrics import compute_metrics

        # Internal metrics are stored under _internal/ prefix in the MeterGroup
        raw_results = compute_metrics("internal_group", aggregate=False)

        assert "test/internal_metric/public" in raw_results
        assert "_internal/test/internal_metric/_internal" in raw_results
        assert raw_results["test/internal_metric/public"] == 5.0
        assert raw_results["_internal/test/internal_metric/_internal"] == 10.0

        # Compute should keep public and filter out internal if finalize filters them
        # (Default Metric.finalize filters k.startswith('_'))
        final_results = engine.compute(raw_results)

        assert "test/internal_metric/public" in final_results
        assert "test/internal_metric/_internal" not in final_results
        # The raw _internal/ key should be gone as it was consumed by compute
        assert "_internal/test/internal_metric/_internal" not in raw_results
