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


@dataclass
class MetricConfig(RegistryConfigStrict):
    sources: list[str] = field(default_factory=lambda: ["forward"])


metric_registry, register_metric, build_metric = make_registry("metric")


class Metric(ABC):
    """Stateless definition for computing metrics from model/source data.

    A Metric implementation defines:
    - Which sources it requires (e.g. ['forward', 'generation'])
    - How to calculate raw results for a batch, potentially emitting multiple values.
    - How to finalize those values after they've been aggregated (e.g., F1 from counts).

    The returned values from __call__ are passed to stateful MetricAccumulators.
    """

    def __init__(self, cfg: MetricConfig):
        self.cfg = cfg

    @property
    def required_sources(self) -> list[str]:
        """List of source names this metric requires."""
        return self.cfg.sources

    @abstractmethod
    def __call__(
        self, batch: dict[str, Any], sources_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Compute raw metric values for the batch.

        Returns:
            Dict mapping sub-metric names to log kwargs for accumulators.
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

    def __call__(
        self, batch: dict[str, Any], sources_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        logits = sources_data["forward"]["logits"]
        targets = batch["input_ids"][:, 1:]
        logits = logits[:, :-1, :]

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
            correct = (predictions == valid_targets).float().sum().item()
        else:
            _, top_k_indices = torch.topk(valid_logits, self.cfg.top_k, dim=-1)
            correct = (
                (top_k_indices == valid_targets.unsqueeze(-1))
                .any(dim=-1)
                .float()
                .sum()
                .item()
            )

        return {
            "accuracy": {
                "value": correct / valid_targets.numel(),
                "weight": valid_targets.numel(),
            }
        }
