"""Generation engines for model rollout."""

import logging
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from optimus_dl.core.log import info_once
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
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate completions for a batch of prompts.

        Args:
            model: The model to use for generation.
            input_ids: Prompt token IDs of shape (batch_size, seq_len).
                May contain right-padding.
            gen_config: Generation parameters.
            seq_lens: Optional ``(batch_size,)`` tensor of true (unpadded)
                prompt lengths. When given, each sequence's continuation is
                appended directly after its last real token and padding is
                masked out of attention — i.e. generation is exact for
                variable-length, right-padded batches.

        Returns:
            Tensor of generated token IDs (including prompts),
            shape ``(batch_size, max_seq_len + num_new_tokens)``.
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
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute batched generation loop.

        Supports right-padded batches via ``seq_lens``: every sequence keeps
        its own read/write *cursor*, so a shorter sequence's continuation is
        written immediately after its last real token instead of after the
        padded block. Attention is masked with ``seq_lens`` so pad tokens are
        never conditioned on.

        When the model exposes ``get_kv_caches`` (e.g. Llama without sliding
        window), generation runs on the incremental KV-cache fast path: one
        full-sequence prefill, then single-token decode steps — O(T) per new
        token instead of O(T²). Models that don't support caching (or compiled
        wrappers) fall back to the legacy full re-forward loop automatically.
        """
        model.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        rows = torch.arange(batch_size, device=device)

        if seq_lens is not None:
            cur_lens = seq_lens.to(device).clone()
        else:
            cur_lens = torch.full(
                (batch_size,), input_ids.size(1), dtype=torch.long, device=device
            )

        # Pre-allocate the output buffer once: prompts + worst-case generation.
        # Trailing cells stay as pad (0) and are masked by ``cur_lens`` throughout.
        out_width = input_ids.size(1) + gen_config.max_new_tokens
        all_ids = torch.zeros(
            (batch_size, out_width), dtype=input_ids.dtype, device=device
        )
        all_ids[:, : input_ids.size(1)] = input_ids

        kv_caches = self._maybe_create_kv_caches(model, batch_size, out_width, device)
        if kv_caches is not None:
            return self._generate_cached(
                model,
                all_ids,
                gen_config,
                cur_lens,
                rows,
                kv_caches,
            )

        # Legacy path: full-sequence re-forward per generated token.
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(gen_config.max_new_tokens):
            # ``active`` must be captured BEFORE the finished-set update below:
            # a sequence emitting its first EOS this step still gets that token
            # written (and its cursor advanced); only *subsequent* steps skip it.
            active = ~finished

            # Forward pass; read logits at each sequence's own last real token.
            outputs = model(all_ids, seq_lens=cur_lens)
            logits = outputs["logits"][rows, cur_lens - 1]

            next_tokens = self._sample_next(logits, gen_config)

            # Mask tokens for finished sequences (if stop_token_id is set)
            if gen_config.stop_token_id is not None:
                next_tokens = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_tokens, gen_config.stop_token_id),
                    next_tokens,
                )
                finished |= (next_tokens == gen_config.stop_token_id).squeeze(1)

            # Write new tokens at each sequence's own cursor; finished
            # sequences do not advance (their tail stays masked-out pad).
            all_ids[rows[active], cur_lens[active]] = next_tokens.squeeze(-1)[active]
            cur_lens += active.long()

            if finished.all():
                break

        # Trim trailing pad columns that no sequence reached.
        return all_ids[:, : int(cur_lens.max().item())]

    @staticmethod
    def _maybe_create_kv_caches(model, batch_size, max_len, device):
        """Return per-layer KV caches if the model supports them, else None.

        Compiled models are excluded: torch.compile cannot safely trace the
        in-place cache writes, so they use the legacy path (correct, slower).
        """
        get_caches = getattr(model, "get_kv_caches", None)
        if get_caches is None or not callable(get_caches):
            return None
        try:
            import torch._dynamo as _dynamo

            if isinstance(model, _dynamo.OptimizedModule):
                info_once(
                    logging.getLogger(__name__),
                    "torch.compile model detected; KV-cached generation disabled",
                )
                return None
        except ImportError:  # pragma: no cover - dynamo always present in torch
            pass
        caches = get_caches(batch_size=batch_size, max_len=max_len, device=device)
        if caches is None:
            info_once(
                logging.getLogger(__name__),
                "Model does not support KV caching; using full re-forward generation",
            )
        return caches

    @staticmethod
    def _sample_next(
        logits: torch.Tensor, gen_config: GenerationConfig
    ) -> torch.Tensor:
        """Sample the next token from ``(B, V)`` logits. Returns ``(B, 1)``."""
        if gen_config.temperature > 0:
            logits = logits / gen_config.temperature
            if gen_config.top_k is not None:
                v, _ = torch.topk(logits, min(gen_config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _generate_cached(
        self,
        model,
        all_ids: torch.Tensor,
        gen_config: GenerationConfig,
        cur_lens: torch.Tensor,
        rows: torch.Tensor,
        kv_caches,
    ) -> torch.Tensor:
        """Incremental generation: one prefill pass, then single-token steps.

        Semantics match the legacy path exactly, including right-padded
        prompts, per-row cursors and stop-token handling:

        * **Prefill**: forward the prompt region ``[:max_len]`` with
          ``seq_lens``, filling each layer's cache. Rows shorter than the
          padded width write garbage into their tail slots, but those slots
          are never attended: every later step masks by per-row length.
        * **Decode**: each row's token is written into its cache at its own
          length slot (right-packed), so row ``i``'s keys/values stay
          contiguous at ``[0, cur_lens[i])``. Inactive (finished) rows feed a
          placeholder that overwrites the same slot every step and whose
          logits are discarded.
        """
        device = all_ids.device
        batch_size = all_ids.size(0)
        prompt_width = int(cur_lens.max().item())

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # --- Prefill ---
        position_ids = (
            torch.arange(prompt_width, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        outputs = model(
            all_ids[:, :prompt_width],
            seq_lens=cur_lens,
            position_ids=position_ids,
            kv_caches=kv_caches,
        )
        logits = outputs["logits"][rows, cur_lens - 1]

        # --- Decode steps ---
        for _ in range(gen_config.max_new_tokens):
            active = ~finished

            next_tokens = self._sample_next(logits, gen_config)

            if gen_config.stop_token_id is not None:
                next_tokens = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_tokens, gen_config.stop_token_id),
                    next_tokens,
                )
                finished |= (next_tokens == gen_config.stop_token_id).squeeze(1)

            # Write sampled tokens at each active sequence's cursor.
            all_ids[rows[active], cur_lens[active]] = next_tokens.squeeze(-1)[active]

            if finished.all():
                break

            # Feed the sampled tokens (placeholders for finished rows) as the
            # next decode step. The token of row ``i`` sits at absolute
            # position ``cur_lens[i]`` and extends its valid length by one;
            # the attention layer derives its write slot from ``seq_lens``.
            outputs = model(
                next_tokens.view(batch_size, 1),
                seq_lens=cur_lens + 1,
                position_ids=cur_lens.view(batch_size, 1),
                kv_caches=kv_caches,
            )
            logits = outputs["logits"][:, -1, :]

            cur_lens += active.long()

        # Trim trailing pad columns that no sequence reached.
        return all_ids[:, : int(cur_lens.max().item())]


(
    _GENERATION_ENGINE,
    register_generation_engine,
    build_generation_engine,
) = make_registry("generation_engine")

register_generation_engine("native", NativeEngineConfig)(NativeEngine)
