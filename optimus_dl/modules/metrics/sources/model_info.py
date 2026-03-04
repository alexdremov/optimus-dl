from dataclasses import dataclass
from typing import Any

import torch

from optimus_dl.modules.metrics.source import (
    MetricSource,
    MetricSourceConfig,
    StandardProtocols,
    register_metric_source,
)


@dataclass
class ModelInfoSourceConfig(MetricSourceConfig):
    """Configuration for ModelInfoSource."""


@register_metric_source("model_info", ModelInfoSourceConfig)
class ModelInfoSource(MetricSource):
    """Source for extracting static info about model."""

    cfg: ModelInfoSourceConfig

    @property
    def provides(self) -> set[str]:
        return {StandardProtocols.MODEL_PARAMETERS_COUNT}

    @torch.no_grad()
    def __call__(
        self,
        dependencies: dict[str, dict[str, Any]],
        model: Any,
        batch: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        model_parameters_count = sum(p.numel() for p in model.parameters())
        model_trainable_parameters_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        return {
            StandardProtocols.MODEL_PARAMETERS_COUNT: dict(
                parameters_count=model_parameters_count,
                trainable_parameters_count=model_trainable_parameters_count,
            )
        }
