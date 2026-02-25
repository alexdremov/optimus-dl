import pytest
import torch
from typing import Any
from unittest.mock import MagicMock

from optimus_dl.modules.metrics.source import (
    StandardProtocols,
    MetricSource,
    MetricSourceConfig,
    ForwardSourceConfig,
    CausalLMSourceConfig,
    register_source,
)
from optimus_dl.modules.metrics.definitions import (
    Metric,
    MetricConfig,
    AccuracyMetricConfig,
    register_metric,
)
from optimus_dl.modules.metrics.engine import MetricEngine
from optimus_dl.modules.metrics.base import _meter_groups, _active_meter_groups

from dataclasses import dataclass

@dataclass
class DummySourceConfig(MetricSourceConfig):
    _name: str = "dummy_source"
    val: int = 1


@register_source("dummy_source", DummySourceConfig)
class DummySource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"dummy_proto"}

    def __call__(self, model, batch, dependencies):
        return {"dummy_proto": self.cfg.val}


@dataclass
class DependentSourceConfig(MetricSourceConfig):
    _name: str = "dependent_source"
    multiplier: int = 2


@register_source("dependent_source", DependentSourceConfig)
class DependentSource(MetricSource):
    @property
    def provides(self) -> set[str]:
        return {"dep_proto"}

    @property
    def requires(self) -> dict[str, set[str]]:
        return {"base": {"dummy_proto"}}

    def __call__(self, model, batch, dependencies):
        base_val = dependencies["base"]["dummy_proto"]
        return {"dep_proto": base_val * self.cfg.multiplier}


@dataclass
class DummyMetricConfig(MetricConfig):
    _name: str = "dummy_metric"


@register_metric("dummy_metric", DummyMetricConfig)
class DummyMetric(Metric):
    @property
    def requires(self) -> dict[str, set[str]]:
        return {"input": {"dep_proto"}}

    def __call__(self, batch, sources_data):
        val = sources_data["input"]["dep_proto"]
        return {"dummy_metric": {"value": val, "weight": 1.0}}


class TestMetricEngineAdvanced:
    def setup_method(self):
        _meter_groups.clear()
        _active_meter_groups.clear()

    def test_source_config_hashing(self):
        cfg1 = ForwardSourceConfig()
        cfg2 = ForwardSourceConfig()
        cfg3 = CausalLMSourceConfig()

        from optimus_dl.modules.metrics.source import ForwardSource, CausalLMSource

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
                        "dependencies": {"base": "provider"}
                    },
                },
                "metrics": [
                    {
                        "_name": "dummy_metric",
                        "type": {"_name": "dummy_metric"},
                        "role_mapping": {"input": "consumer"}
                    }
                ]
            }
        ]

        engine = MetricEngine("test_group", configs)
        model = MagicMock()
        batch = {}

        # Run update
        engine.update(model, batch)

        # Compute results
        from optimus_dl.modules.metrics import compute_metrics
        raw_results = compute_metrics("test_group", aggregate=False)
        results = engine.compute(raw_results)

        # 5 * 3 = 15
        assert "test/dummy_metric" in results
        assert results["test/dummy_metric"] == 15.0

    def test_cross_group_caching(self):
        # We will use a mock model to count forward passes
        model = MagicMock()
        model.return_value = {"logits": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])}

        configs = [
            {
                "_name": "source_group",
                "prefix": "group1",
                "sources": {"default": {"_name": "causal_lm"}},
                "metrics": [{"_name": "accuracy", "type": {"_name": "accuracy"}}]
            },
            {
                "_name": "source_group",
                "prefix": "group2",
                "sources": {"default": {"_name": "causal_lm"}},
                "metrics": [{"_name": "accuracy", "type": {"_name": "accuracy"}}]
            }
        ]

        engine = MetricEngine("test_group", configs)

        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        engine.update(model, batch)

        # Even though there are two groups, they use the exact same source config.
        # The model should only be called once per batch.
        assert model.call_count == 1

    def test_accuracy_metric_causal_lm(self):
        configs = [
            {
                "_name": "accuracy",
                "type": {"_name": "accuracy"}
            }
        ]

        engine = MetricEngine("test_group", configs)

        # Batch: 1 sequence, length 3
        batch = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Model predicts 2 for the first token (correct), 4 for the second (incorrect)
        # Logits shape: [1, 3, vocab_size]
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 2] = 10.0 # Predicts 2
        logits[0, 1, 4] = 10.0 # Predicts 4
        logits[0, 2, 5] = 10.0 # Ignored (shifted)

        model = MagicMock()
        model.return_value = {"logits": logits}

        engine.update(model, batch)

        from optimus_dl.modules.metrics import compute_metrics
        raw_results = compute_metrics("test_group", aggregate=False)
        results = engine.compute(raw_results)

        # Targets are [2, 3]
        # Predictions are [2, 4]
        # Accuracy = 1 / 2 = 0.5
        assert results["accuracy"] == 0.5

    def test_cyclic_dependency_detection(self, caplog):
        configs = [
            {
                "_name": "source_group",
                "prefix": "test_cycle",
                "sources": {
                    "source_a": {"_name": "dependent_source", "multiplier": 1, "dependencies": {"base": "source_b"}},
                    "source_b": {"_name": "dependent_source", "multiplier": 1, "dependencies": {"base": "source_a"}},
                },
                "metrics": []
            }
        ]

        @dataclass
        class CycleSourceConfig(MetricSourceConfig):
            _name: str = "cycle_source"
            
        @register_source("cycle_source", CycleSourceConfig)
        class CycleSource(MetricSource):
            @property
            def provides(self) -> set[str]:
                return {"cycle_proto"}
            @property
            def requires(self) -> dict[str, set[str]]:
                return {"base": {"cycle_proto"}}
            def __call__(self, model, batch, dependencies):
                return {"cycle_proto": 1}

        configs_cycle = [
            {
                "_name": "source_group",
                "prefix": "test_cycle",
                "sources": {
                    "source_a": {"_name": "cycle_source", "dependencies": {"base": "source_b"}},
                    "source_b": {"_name": "cycle_source", "dependencies": {"base": "source_a"}},
                },
                "metrics": [
                    {
                        "_name": "cycle_metric",
                        "type": {"_name": "cycle_metric"},
                        "role_mapping": {"input": "source_a"}
                    }
                ]
            }
        ]
        
        @dataclass
        class CycleMetricConfig(MetricConfig):
            _name: str = "cycle_metric"

        @register_metric("cycle_metric", CycleMetricConfig)
        class CycleMetric(Metric):
            @property
            def requires(self) -> dict[str, set[str]]:
                return {"input": {"cycle_proto"}}
            def __call__(self, batch, sources_data):
                return {"cycle_metric": {"value": 1.0, "weight": 1.0}}

        engine = MetricEngine("test_group", configs_cycle)
        
        model = MagicMock()
        batch = {}
        
        # update should log the cyclic error and skip metric computation
        engine.update(model, batch)
        
        assert "Cyclic dependency detected for source" in caplog.text

    def test_slice_filter(self):
        configs = [
            {
                "_name": "source_group",
                "prefix": "filtered",
                "sources": {
                    "provider": {"_name": "dummy_source", "val": 10},
                    "consumer": {"_name": "dependent_source", "multiplier": 1, "dependencies": {"base": "provider"}},
                },
                "metrics": [
                    {
                        "_name": "dummy_metric",
                        "type": {"_name": "dummy_metric"},
                        "role_mapping": {"input": "consumer"},
                        "slice_filter": "batch.get('is_valid', False)"
                    }
                ]
            }
        ]

        engine = MetricEngine("test_group", configs)
        model = MagicMock()
        
        # Batch 1 is not valid
        engine.update(model, {"is_valid": False})
        
        # Batch 2 is valid
        engine.update(model, {"is_valid": True})
        
        from optimus_dl.modules.metrics import compute_metrics
        raw_results = compute_metrics("test_group", aggregate=False)
        results = engine.compute(raw_results)
        
        # Only the second batch should be logged, so the value is 10 (since dummy_metric just returns val)
        assert "filtered/dummy_metric" in results
        assert results["filtered/dummy_metric"] == 10.0

    def test_missing_role_mapping(self):
        configs = [
            {
                "_name": "source_group",
                "prefix": "test",
                "sources": {
                    "provider": {"_name": "dummy_source", "val": 5},
                },
                "metrics": [
                    {
                        "_name": "dummy_metric",
                        "type": {"_name": "dummy_metric"},
                        # Intentionally missing role_mapping for 'input'
                    }
                ]
            }
        ]

        with pytest.raises(ValueError, match="requires role 'input' mapped to 'input'"):
            MetricEngine("test_group", configs)
