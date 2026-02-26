import copy
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

import torch
import torch.nn.functional as F

from optimus_dl.modules.metrics.source import (
    MetricSource,
    MetricSourceConfig,
    StandardProtocols,
    register_metric_source,
)


@dataclass
class GenerationSourceConfig(MetricSourceConfig):
    """Configuration for GenerationSource.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (1.0 = no change, < 1.0 = sharper, > 1.0 = smoother).
        top_k: If set, only sample from the top k tokens.
        top_p: If set, only sample from tokens with cumulative probability >= p.
        do_sample: Whether to use sampling; if False, uses greedy search.
        eos_token_id: Token ID that signals the end of a sequence.
    """

    _name: str = "generation"
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False
    eos_token_id: int | None = None


@register_metric_source("generation", GenerationSourceConfig)
class GenerationSource(MetricSource):
    """Source that generates new tokens from a model given a prompt.

    Supports greedy search and various sampling techniques (temperature, top-k, top-p).
    The generated sequences are provided under the 'generated_tokens' protocol.
    """

    cfg: GenerationSourceConfig

    def __init__(self, cfg: GenerationSourceConfig):
        super().__init__(cfg)

    @property
    def provides(self) -> set[str]:
        return {StandardProtocols.GENERATED_TOKENS}

    @torch.no_grad()
    def __call__(
        self,
        dependencies: dict[str, dict[str, Any]],
        model: Any,
        batch: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute generation.

        Args:
            dependencies: Data from required sources (none).
            model: The model to generate from.
            batch: Input batch, expected to contain 'input_ids'.
            **kwargs: Additional arguments.
        """
        batch = copy.copy(batch)
        input_ids = batch.get("input_ids")
        if input_ids is None and hasattr(batch, "input_ids"):
            input_ids = batch.input_ids

        if input_ids is None:
            raise ValueError("GenerationSource requires 'input_ids' in the batch")

        # Start generation
        batch_size = input_ids.size(0)
        device = input_ids.device
        curr_ids = input_ids

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Use eos_token_id or a default (0) for padding finished sequences
        pad_token_id = self.cfg.eos_token_id if self.cfg.eos_token_id is not None else 0

        # Store only the new tokens
        generated = []

        for _ in range(self.cfg.max_new_tokens):
            # If all sequences are finished, we can stop early
            if self.cfg.eos_token_id is not None and finished.all():
                break

            # Forward pass to get logits for the last token
            # Note: We update batch input_ids for models that might use other batch info
            batch["input_ids"] = curr_ids
            outputs = model(**batch)

            if isinstance(outputs, dict):
                logits = outputs.get("logits")
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Take logits for the last position: [B, T, V] -> [B, V]
            next_token_logits = logits[:, -1, :]

            if not self.cfg.do_sample:
                # Greedy search
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Sampling logic
                if self.cfg.temperature != 1.0:
                    next_token_logits = next_token_logits / self.cfg.temperature

                if self.cfg.top_k is not None:
                    v, _ = torch.topk(
                        next_token_logits,
                        min(self.cfg.top_k, next_token_logits.size(-1)),
                    )
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

                if self.cfg.top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > self.cfg.top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            # Mask out tokens for finished sequences
            if self.cfg.eos_token_id is not None:
                next_tokens = torch.where(
                    finished.unsqueeze(1),
                    torch.tensor(pad_token_id, device=device),
                    next_tokens,
                )
                # Update finished mask
                finished |= next_tokens.squeeze(1) == self.cfg.eos_token_id

            # Append generated tokens
            generated.append(next_tokens)
            curr_ids = torch.cat([curr_ids, next_tokens], dim=1)

        return {
            StandardProtocols.GENERATED_TOKENS: (
                torch.cat(generated, dim=1)
                if generated
                else torch.empty((batch_size, 0), dtype=torch.long, device=device)
            )
        }
