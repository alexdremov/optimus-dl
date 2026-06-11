"""Generation engines for model rollout."""

import logging
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from optimus_dl.core.registry import (
    RegistryConfigStrict,
    make_registry,
)
from optimus_dl.modules.model.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig(RegistryConfigStrict):
    """Configuration for generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        top_k: Optional top-k sampling.
        stop_token_id: Optional token ID that triggers early stopping.
    """

    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    stop_token_id: int | None = None


class GenerationEngine(ABC):
    """Base class for generation engines."""

    @abstractmethod
    def generate(
        self,
        model: BaseModel,
        input_ids: torch.Tensor,
        gen_config: GenerationConfig,
    ) -> torch.Tensor:
        """Generate completions for a batch of prompts.

        Args:
            model: The model to use for generation.
            input_ids: Prompt token IDs of shape (batch_size, seq_len).
            gen_config: Generation parameters.

        Returns:
            Tensor of generated token IDs (including prompts or just completions).
        """
        pass


@dataclass
class NativeEngineConfig(RegistryConfigStrict):
    """Configuration for the Native generation engine."""

    pass


class NativeEngine(GenerationEngine):
    """Native PyTorch implementation of batched generation."""

    def __init__(self, cfg: NativeEngineConfig):
        self.cfg = cfg

    @torch.no_grad()
    def generate(
        self,
        model: BaseModel,
        input_ids: torch.Tensor,
        gen_config: GenerationConfig,
    ) -> torch.Tensor:
        """Execute batched generation loop."""
        model.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        # We'll accumulate all tokens in this tensor
        all_ids = input_ids.clone()

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(gen_config.max_new_tokens):
            # Forward pass
            # Note: We don't use KV caching here for simplicity in the native engine.
            # vLLM engine will handle optimized caching.
            outputs = model(all_ids)
            logits = outputs["logits"][:, -1, :]

            # Sampling
            if gen_config.temperature > 0:
                logits = logits / gen_config.temperature
                if gen_config.top_k is not None:
                    v, _ = torch.topk(logits, min(gen_config.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Mask tokens for finished sequences (if stop_token_id is set)
            if gen_config.stop_token_id is not None:
                next_tokens = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_tokens, gen_config.stop_token_id),
                    next_tokens,
                )
                finished |= (next_tokens == gen_config.stop_token_id).squeeze(1)

            all_ids = torch.cat([all_ids, next_tokens], dim=1)

            if finished.all():
                break

        return all_ids


(
    _GENERATION_ENGINE,
    register_generation_engine,
    build_generation_engine,
) = make_registry("generation_engine")

register_generation_engine("native", NativeEngineConfig)(NativeEngine)
