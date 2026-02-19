"""Rotary Positional Embeddings (RoPE) implementation.

This module provides utilities for computing and applying Rotary Positional
Embeddings, as used in models like Llama and Qwen.
"""

import math

import torch
from torch.distributed.tensor import DTensor


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, scaling_config: dict | None = None
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponential (cis) with optional scaling.

    Args:
        dim: Dimension of the head.
        end: Maximum sequence length.
        theta: Base frequency for the positional encoding.
        scaling_config: Optional RoPE scaling configuration (e.g., YaRN).

    Returns:
        Tensor of shape (end, dim // 2, 2) representing the real and imaginary
        parts of the frequencies.
    """
    if scaling_config is None or scaling_config.get("rope_type") != "yarn":
        # Fallback to standard RoPE
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        return torch.stack((torch.cos(freqs), torch.sin(freqs)), dim=-1)

    # YaRN implementation
    factor = scaling_config.get("factor", 1.0)
    original_max_position_embeddings = scaling_config.get(
        "original_max_position_embeddings", 8192
    )
    attention_factor = scaling_config.get("attention_factor", 1.0)
    beta_fast = scaling_config.get("beta_fast", 32.0)
    beta_slow = scaling_config.get("beta_slow", 1.0)

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (
            dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        # HF uses truncate=True by default for YaRN
        return max(math.floor(low), 0), min(math.ceil(high), dim - 1)

    low, high = find_correction_range(
        beta_fast, beta_slow, dim, theta, original_max_position_embeddings
    )

    pos_freqs = theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    def linear_ramp_factor(min_val, max_val, dim_range):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim_range, dtype=torch.float32) - min_val) / (
            max_val - min_val
        )
        return torch.clamp(linear_func, 0, 1)

    extrapolation_factor = 1.0 - linear_ramp_factor(low, high, dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )

    t = torch.arange(end, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq).float()

    return torch.stack(
        (torch.cos(freqs) * attention_factor, torch.sin(freqs) * attention_factor),
        dim=-1,
    )


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis for broadcasting with x.

    Args:
        freqs_cis: Frequency tensor of shape (seq_len, head_dim // 2, 2).
        x: Input tensor to apply RoPE to.

    Returns:
        Reshaped frequency tensor compatible with x.
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (
        x.shape[1],
        x.shape[-2],
    ), f"{freqs_cis.shape = }, {x.shape = }"
    # New shape for broadcasting
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Positional Embeddings to Query and Key tensors.

    Handles both standard Tensors and distributed DTensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        freqs_cis: Precomputed frequency tensor.

    Returns:
        Tuple of (q, k) with rotary embeddings applied.
    """
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)

    # Handle DTensor (Tensor Parallelism)
    # We perform RoPE on local shards to avoid complex sharding propagation issues with reshape/select.
    is_q_dtensor = isinstance(q, DTensor)
    is_k_dtensor = isinstance(k, DTensor)
    is_freqs_cis_dtensor = isinstance(freqs_cis, DTensor)

    q_in = q.to_local() if is_q_dtensor else q
    k_in = k.to_local() if is_k_dtensor else k
    freqs_cis_in = freqs_cis.to_local() if is_freqs_cis_dtensor else freqs_cis

    q_in = q_in.float().reshape(*q_in.shape[:-1], -1, 2)
    k_in = k_in.float().reshape(*k_in.shape[:-1], -1, 2)

    freqs_cis_res = _reshape_for_broadcast(freqs_cis_in, q_in)

    # Perform manual "complex" multiplication
    q_cos = q_in[..., 0] * freqs_cis_res[..., 0] - q_in[..., 1] * freqs_cis_res[..., 1]
    q_sin = q_in[..., 0] * freqs_cis_res[..., 1] + q_in[..., 1] * freqs_cis_res[..., 0]
    k_cos = k_in[..., 0] * freqs_cis_res[..., 0] - k_in[..., 1] * freqs_cis_res[..., 1]
    k_sin = k_in[..., 0] * freqs_cis_res[..., 1] + k_in[..., 1] * freqs_cis_res[..., 0]

    # Combine the results back into the interleaved format expected by q and k
    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q_in.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k_in.shape).flatten(3)

    # Wrap back to DTensor if inputs were DTensor
    if is_q_dtensor:
        q_out = DTensor.from_local(q_out, q.device_mesh, q.placements)
    if is_k_dtensor:
        k_out = DTensor.from_local(k_out, k.device_mesh, k.placements)

    return q_out, k_out
