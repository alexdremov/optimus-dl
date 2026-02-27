import torch
import pytest

from optimus_dl.modules.model.blocks.attention import RotarySelfAttention


def test_varlen_sliding_window_consistency(device):
    """Verify that varlen attention with a sliding window matches padded SDPA."""
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.float16

    n_embd = 128
    n_head = 4
    n_kv_head = 4  # Use MHA for simplicity first
    head_dim = 32
    sliding_window = 16

    # Initialize attention layer
    torch.manual_seed(42)
    attn = (
        RotarySelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            bias=False,
            sliding_window=sliding_window,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    # Create mock inputs: 2 docs of length 20 and 30
    seq_len1, seq_len2 = 20, 30
    total_tokens = seq_len1 + seq_len2
    max_len = max(seq_len1, seq_len2)

    # 1. SDPA Path (Padded Batch)
    x_padded = torch.randn(2, max_len, n_embd, device=device, dtype=dtype)
    seq_lens = torch.tensor([seq_len1, seq_len2], device=device)
    pos_padded = torch.zeros(2, max_len, dtype=torch.long, device=device)
    pos_padded[0, :seq_len1] = torch.arange(seq_len1)
    pos_padded[1, :seq_len2] = torch.arange(seq_len2)

    freqs_cis = torch.randn(max_len, head_dim // 2, 2, device=device, dtype=dtype)

    with torch.no_grad():
        # This will use SDPA with manual sliding window mask internally
        out_sdpa = attn(
            x_padded, freqs_cis=freqs_cis, seq_lens=seq_lens, position_ids=pos_padded
        )

    # 2. Varlen Path (Flat Batch)
    x_flat = torch.cat(
        [x_padded[0, :seq_len1], x_padded[1, :seq_len2]], dim=0
    ).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [0, seq_len1, total_tokens], device=device, dtype=torch.int32
    )
    pos_flat = (
        torch.cat([torch.arange(seq_len1), torch.arange(seq_len2)], dim=0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        # This will use flash_attn_varlen_func (CUDA) or _varlen_attn_fallback (CPU)
        # Both should handle sliding_window
        out_varlen = attn(
            x_flat,
            freqs_cis=freqs_cis,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_len,
            position_ids=pos_flat,
        )

    # Compare outputs
    out_sdpa_flat = torch.cat([out_sdpa[0, :seq_len1], out_sdpa[1, :seq_len2]], dim=0)

    assert out_varlen.shape == (1, total_tokens, n_embd)
    torch.testing.assert_close(
        out_varlen.squeeze(0), out_sdpa_flat, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("n_kv_head", [1, 2])
def test_varlen_sliding_window_gqa_consistency(device, n_kv_head):
    """Verify that varlen attention with sliding window AND GQA matches padded SDPA."""
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.float16

    n_embd = 128
    n_head = 4
    head_dim = 32
    sliding_window = 10

    # Initialize attention layer
    torch.manual_seed(42)
    attn = (
        RotarySelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            bias=False,
            sliding_window=sliding_window,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    # Create mock inputs
    seq_len1, seq_len2 = 15, 25
    total_tokens = seq_len1 + seq_len2
    max_len = max(seq_len1, seq_len2)

    x_padded = torch.randn(2, max_len, n_embd, device=device, dtype=dtype)
    seq_lens = torch.tensor([seq_len1, seq_len2], device=device)
    pos_padded = torch.zeros(2, max_len, dtype=torch.long, device=device)
    pos_padded[0, :seq_len1] = torch.arange(seq_len1)
    pos_padded[1, :seq_len2] = torch.arange(seq_len2)

    freqs_cis = torch.randn(max_len, head_dim // 2, 2, device=device, dtype=dtype)

    with torch.no_grad():
        out_sdpa = attn(
            x_padded, freqs_cis=freqs_cis, seq_lens=seq_lens, position_ids=pos_padded
        )

    x_flat = torch.cat(
        [x_padded[0, :seq_len1], x_padded[1, :seq_len2]], dim=0
    ).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [0, seq_len1, total_tokens], device=device, dtype=torch.int32
    )
    pos_flat = (
        torch.cat([torch.arange(seq_len1), torch.arange(seq_len2)], dim=0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out_varlen = attn(
            x_flat,
            freqs_cis=freqs_cis,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_len,
            position_ids=pos_flat,
        )

    out_sdpa_flat = torch.cat([out_sdpa[0, :seq_len1], out_sdpa[1, :seq_len2]], dim=0)
    torch.testing.assert_close(
        out_varlen.squeeze(0), out_sdpa_flat, atol=1e-4, rtol=1e-4
    )
