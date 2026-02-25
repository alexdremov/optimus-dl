from dataclasses import dataclass
from typing import Any

from optimus_dl.modules.metrics.source import (
    MetricSource,
    MetricSourceConfig,
    StandardProtocols,
    register_metric_source,
)


@dataclass
class CausalLMSourceConfig(MetricSourceConfig):
    """Configuration for CausalLMSource."""

    _name: str = "causal_lm"


@register_metric_source("causal_lm", CausalLMSourceConfig)
class CausalLMSource(MetricSource):
    """Source for Causal LM that extracts logits and labels from the model and batch."""

    @property
    def provides(self) -> set[str]:
        return {StandardProtocols.LOGITS, StandardProtocols.LABELS}

    def __call__(
        self,
        dependencies: dict[str, dict[str, Any]],
        model: Any,
        batch: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the source.

        Args:
            dependencies: Data from required sources (none for this source).
            model: The model to run forward pass on.
            batch: The input batch, expected to contain 'labels'.
            **kwargs: Additional arguments (like criterion if needed).
        """
        # Run model forward pass
        output = model(batch)

        # Handle different output types (dict, Namespace, or raw Tensor)
        if isinstance(output, dict):
            logits = output.get("logits")
        elif hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output  # Assume it's the logits tensor

        # Extract labels from batch
        labels = None
        if isinstance(batch, dict):
            labels = batch.get("labels")
        elif hasattr(batch, "labels"):
            labels = batch.labels

        return {
            StandardProtocols.LOGITS: logits,
            StandardProtocols.LABELS: labels,
        }
