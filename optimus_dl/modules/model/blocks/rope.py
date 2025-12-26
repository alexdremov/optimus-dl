import torch
from torch.distributed.tensor import DTensor


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    # Stack the cos and sin parts in the last dimension to simulate complex numbers
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
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


def apply_rotary_emb(q, k, freqs_cis):
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
    freqs_cis = freqs_cis.to_local() if is_freqs_cis_dtensor else freqs_cis

    q_in = q_in.float().reshape(*q_in.shape[:-1], -1, 2)
    k_in = k_in.float().reshape(*k_in.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q_in)

    # Perform manual "complex" multiplication
    q_cos = q_in[..., 0] * freqs_cis[..., 0] - q_in[..., 1] * freqs_cis[..., 1]
    q_sin = q_in[..., 0] * freqs_cis[..., 1] + q_in[..., 1] * freqs_cis[..., 0]
    k_cos = k_in[..., 0] * freqs_cis[..., 0] - k_in[..., 1] * freqs_cis[..., 1]
    k_sin = k_in[..., 0] * freqs_cis[..., 1] + k_in[..., 1] * freqs_cis[..., 0]

    # Combine the results back into the interleaved format expected by q and k
    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q_in.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k_in.shape).flatten(3)

    # Wrap back to DTensor if inputs were DTensor
    if is_q_dtensor:
        q_out = DTensor.from_local(q_out, q.device_mesh, q.placements)
    if is_k_dtensor:
        k_out = DTensor.from_local(k_out, k.device_mesh, k.placements)

    return q_out, k_out
