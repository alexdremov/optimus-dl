from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

import torch

from optimus_dl.core.registry import (
    RegistryConfigStrict,
    make_registry,
)
from optimus_dl.modules.metrics.source import StandardProtocols


@dataclass
class MetricConfig(RegistryConfigStrict):
    pass


metric_registry, register_metric, build_metric = make_registry("metric")


class Metric(ABC):
    """Stateless definition for computing metrics from model/source data.

    A Metric implementation defines:
    - What data it requires via `requires` mapping (Role -> set of protocol strings).
    - How to calculate raw results for a batch, potentially emitting multiple sub-values.
    - How to finalize those values after they've been aggregated (e.g., F1 from counts).
    """

    def __init__(self, cfg: MetricConfig):
        self.cfg = cfg

    @property
    @abstractmethod
    def requires(self) -> dict[str, set[str]]:
        """Mapping from source role name to a set of required protocol strings."""
        raise NotImplementedError

    @property
    def accumulators(self) -> dict[str, str]:
        """Define how each sub-metric should be aggregated across batches.
        
        Returns a mapping from sub-metric names to accumulator types
        (e.g., 'average', 'sum', 'gather', 'perplexity').
        """
        return {self.cfg._name: "average"}

    @abstractmethod
    def __call__(
        self, batch: dict[str, Any], sources_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Compute raw metric values for the batch.

        Args:
            batch: The original batch dictionary.
            sources_data: Dict mapping role -> Protocol string -> data.

        Returns:
            Dict mapping sub-metric names to log kwargs (e.g., {'value': ..., 'weight': ...})
            for accumulators.
        """
        raise NotImplementedError

    def finalize(self, aggregated_data: dict[str, Any]) -> dict[str, Any]:
        """Perform final calculations on aggregated data.

        Args:
            aggregated_data: Dict mapping sub-metric names to their
                computed/aggregated values from accumulators.

        Returns:
            Dict of final metrics to be logged/reported.
        """
        return aggregated_data


@dataclass
class AccuracyMetricConfig(MetricConfig):
    _name: str = "accuracy"
    top_k: int = 1
    ignore_index: int = -100


@register_metric("accuracy", AccuracyMetricConfig)
class AccuracyMetric(Metric):
    """Top-K accuracy calculation."""

    @property
    def requires(self) -> dict[str, set[str]]:
        return {"default": {StandardProtocols.LOGITS, StandardProtocols.TARGETS}}

    def __call__(
        self, batch: dict[str, Any], sources_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        logits = sources_data["default"][StandardProtocols.LOGITS]
        targets = sources_data["default"][StandardProtocols.TARGETS]

        # Flatten
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        valid_mask = targets != self.cfg.ignore_index
        valid_targets = targets[valid_mask]
        valid_logits = logits[valid_mask]

        if valid_targets.numel() == 0:
            return {"accuracy": {"value": 0.0, "weight": 0.0}}

        if self.cfg.top_k == 1:
            predictions = torch.argmax(valid_logits, dim=-1)
            correct = (predictions == valid_targets).float().sum()
        else:
            _, top_k_indices = torch.topk(valid_logits, self.cfg.top_k, dim=-1)
            correct = (
                (top_k_indices == valid_targets.unsqueeze(-1))
                .any(dim=-1)
                .float()
                .sum()
            )

        return {
            "accuracy": {
                "value": correct / valid_targets.numel(),
                "weight": valid_targets.numel(),
            }
        }
