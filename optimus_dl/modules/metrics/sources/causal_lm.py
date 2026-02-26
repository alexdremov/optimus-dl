import copy
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
class CausalLMSourceConfig(MetricSourceConfig):
    """Configuration for CausalLMSource."""

    _name: str = "causal_lm"
    padding_token_id: int = -100


@register_metric_source("causal_lm", CausalLMSourceConfig)
class CausalLMSource(MetricSource):
    """Source for Causal LM that extracts logits and labels from the model and batch."""

    cfg: CausalLMSourceConfig

    @property
    def provides(self) -> set[str]:
        return {StandardProtocols.LOGITS, StandardProtocols.CLASSIFICATION}

    @torch.no_grad()
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
            batch: The input batch, expected to contain 'input_ids'.
            **kwargs: Additional arguments (like criterion if needed).
        """
        batch = copy.copy(batch)
        input_ids = batch.pop("input_ids")
        batch["input_ids"] = input_ids[:, :-1]

        output = model(**batch)

        targets = input_ids[:, 1:]

        # Handle different output types (dict, Namespace, or raw Tensor)
        if isinstance(output, dict):
            logits = output.get("logits")
        elif hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output  # Assume it's the logits tensor

        mask = targets != self.cfg.padding_token_id
        if "seq_lens" in batch:
            mask = mask & (
                torch.arange(mask.shape[1], device=mask.device)
                < batch["seq_lens"][:, None]
            )

        classification = dict(
            predictions=logits.argmax(dim=-1),
            targets=targets,
            mask=mask,
        )

        return {
            StandardProtocols.LOGITS: logits,
            StandardProtocols.CLASSIFICATION: classification,
        }
