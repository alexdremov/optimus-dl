"""Utility functions for loading Hugging Face models."""

import logging
from typing import (
    Any,
)

import torch

logger = logging.getLogger(__name__)


def permute_rope_weight(
    w: torch.Tensor, n_heads: int, head_dim: int, interleaved: bool = True
) -> torch.Tensor:
    """Permute weights for Rotary Positional Embeddings.

    HF typically uses a half-half split (first half of head_dim is cos, second is sin).
    Optimus-DL uses interleaved (cos, sin, cos, sin...).

    Args:
        w: Weight tensor of shape (n_heads * head_dim, input_dim) or (n_heads * head_dim,).
        n_heads: Number of attention heads.
        head_dim: Dimension of each head.
        interleaved: If True, permutes to interleaved format. If False, returns as is.

    Returns:
        Permuted weight tensor.
    """
    if not interleaved:
        return w

    original_shape = w.shape
    # Determine if weights are shared across heads (e.g. Q/K norm in some models)
    effective_n_heads = n_heads
    if w.shape[0] == head_dim:
        effective_n_heads = 1
    elif w.shape[0] != n_heads * head_dim:
        # Fallback for unexpected shapes - don't permute if we can't reason about it
        logger.warning(
            f"Unexpected shape for RoPE permutation: {w.shape}. Expected first dim to be {head_dim} or {n_heads * head_dim}. Skipping permutation."
        )
        return w

    # Handle both weight (2D) and bias (1D)
    if w.ndim == 1:
        w = w.view(effective_n_heads, head_dim)
        w1 = w[:, : head_dim // 2]
        w2 = w[:, head_dim // 2 :]
        w_new = torch.stack((w1, w2), dim=2)
        return w_new.reshape(original_shape)

    # 2D case: (output_dim, input_dim)
    w = w.view(effective_n_heads, head_dim, -1)
    w1 = w[:, : head_dim // 2, :]
    w2 = w[:, head_dim // 2 :, :]
    # Interleave: (x0, x_half, x1, x_half+1...)
    w_new = torch.stack((w1, w2), dim=2)
    return w_new.reshape(original_shape)


class WeightMapper:
    """Helper to map and copy weights from HF state dict to local model."""

    def __init__(
        self, hf_sd: dict[str, torch.Tensor], local_sd: dict[str, torch.Tensor]
    ):
        self.hf_sd = hf_sd
        self.local_sd = local_sd
        self.loaded_keys: set[str] = set()

    def copy(
        self,
        src_key: str,
        dest_key: str,
        permute: bool = False,
        n_heads: int | None = None,
        head_dim: int | None = None,
        transpose: bool = False,
    ):
        """Copy weight from HF state dict to local state dict."""
        if src_key not in self.hf_sd:
            if dest_key in self.local_sd:
                logger.warning(f"Missing key in HF model: {src_key}")
            return
        w = self.hf_sd[src_key]

        if transpose:
            w = w.t()

        if permute:
            assert n_heads is not None and head_dim is not None
            w = permute_rope_weight(w, n_heads, head_dim)

        if dest_key not in self.local_sd:
            logger.warning(f"Extra key in HF model not in local model: {dest_key}")
            return

        if self.local_sd[dest_key].shape != w.shape:
            logger.warning(
                f"Shape mismatch for {dest_key}: {self.local_sd[dest_key].shape} vs {w.shape}. Attempting reshape."
            )
            w = w.view(self.local_sd[dest_key].shape)

        self.local_sd[dest_key].copy_(w)
        self.loaded_keys.add(dest_key)

    def validate(
        self,
        tie_word_embeddings: bool = False,
        ignore_patterns: list[str] | None = None,
    ):
        """Validate that all expected keys were loaded."""
        expected_keys = set(self.local_sd.keys())
        missing_keys = expected_keys - self.loaded_keys

        # Filter out ignored patterns
        if ignore_patterns:
            missing_keys = {
                k
                for k in missing_keys
                if not any(pattern in k for pattern in ignore_patterns)
            }

        # Common ignorable keys
        missing_keys = {
            k for k in missing_keys if "inv_freq" not in k and "bias" not in k
        }

        if tie_word_embeddings:
            if (
                "transformer.wte.weight" in self.loaded_keys
                and "lm_head.weight" in missing_keys
            ):
                missing_keys.remove("lm_head.weight")
            if (
                "lm_head.weight" in self.loaded_keys
                and "transformer.wte.weight" in missing_keys
            ):
                missing_keys.remove("transformer.wte.weight")

        if missing_keys:
            logger.warning(f"Missing keys in loaded model: {missing_keys}")
        else:
            logger.info("All weights loaded successfully.")


def update_config_from_hf(
    optimus_cfg: Any, hf_config: Any, head_dim_fallback: int | None = None
):
    """Update Optimus-DL config from HF config with common attributes."""
    optimus_cfg.n_layer = hf_config.num_hidden_layers
    optimus_cfg.n_head = hf_config.num_attention_heads
    optimus_cfg.n_embd = hf_config.hidden_size
    optimus_cfg.vocab_size = hf_config.vocab_size
    optimus_cfg.sequence_length = getattr(hf_config, "max_position_embeddings", 2048)
    optimus_cfg.block_size = optimus_cfg.sequence_length
    optimus_cfg.rmsnorm_eps = getattr(hf_config, "rms_norm_eps", 1e-5)
    optimus_cfg.intermediate_size = getattr(hf_config, "intermediate_size", None)
    optimus_cfg.rope_theta = getattr(hf_config, "rope_theta", 10000.0)
    optimus_cfg.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    if hasattr(hf_config, "num_key_value_heads"):
        optimus_cfg.n_kv_head = hf_config.num_key_value_heads
    else:
        optimus_cfg.n_kv_head = hf_config.num_attention_heads

    if hasattr(hf_config, "head_dim") and isinstance(hf_config.head_dim, int):
        optimus_cfg.head_dim = hf_config.head_dim
    elif head_dim_fallback:
        optimus_cfg.head_dim = head_dim_fallback
    else:
        optimus_cfg.head_dim = optimus_cfg.n_embd // optimus_cfg.n_head
