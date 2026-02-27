import torch

from optimus_dl.modules.model.blocks.attention import RotarySelfAttention


def test_varlen_attn_gqa_parity(device):
    """Verify if varlen attention (including GQA) matches SDPA results."""
    dtype = torch.float32  # Use float32 for CPU parity
    if device.type == "cuda":
        dtype = torch.float16  # Use float16 for CUDA parity

    n_embd = 128
    n_head = 4
    n_kv_head = 1
    head_dim = 32

    # Initialize attention layer
    torch.manual_seed(42)
    attn = (
        RotarySelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            bias=False,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    # Create mock inputs
    # Two documents: lengths 4 and 6. Total tokens = 10.
    seq_len1, seq_len2 = 4, 6
    total_tokens = seq_len1 + seq_len2
    max_len = max(seq_len1, seq_len2)

    # 1. SDPA Path (Padded Batch)
    # Shape (B=2, T=max_len, C)
    x_padded = torch.randn(2, max_len, n_embd, device=device, dtype=dtype)
    seq_lens = torch.tensor([seq_len1, seq_len2], device=device)

    # Position IDs for padded batch: (B, T)
    pos_padded = torch.zeros(2, max_len, dtype=torch.long, device=device)
    pos_padded[0, :seq_len1] = torch.arange(seq_len1)
    pos_padded[1, :seq_len2] = torch.arange(seq_len2)

    # RoPE freqs: (max_len, D/2, 2)
    freqs_cis = torch.randn(max_len, head_dim // 2, 2, device=device, dtype=dtype)

    # Forward SDPA
    with torch.no_grad():
        out_sdpa = attn(
            x_padded, freqs_cis=freqs_cis, seq_lens=seq_lens, position_ids=pos_padded
        )

    # 2. Varlen Path (Flat Batch)
    # Assemble flat inputs
    x_flat = torch.cat(
        [x_padded[0, :seq_len1], x_padded[1, :seq_len2]], dim=0
    ).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [0, seq_len1, total_tokens], device=device, dtype=torch.int32
    )

    # Position IDs for flat batch: (1, total_tokens)
    pos_flat = (
        torch.cat([torch.arange(seq_len1), torch.arange(seq_len2)], dim=0)
        .unsqueeze(0)
        .to(device)
    )

    # Forward Varlen
    with torch.no_grad():
        out_varlen = attn(
            x_flat,
            freqs_cis=freqs_cis,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_len,
            position_ids=pos_flat,
        )

    # Compare outputs
    # Extract non-padded parts from out_sdpa
    out_sdpa_flat = torch.cat([out_sdpa[0, :seq_len1], out_sdpa[1, :seq_len2]], dim=0)

    # Check shape
    assert out_varlen.shape == (1, total_tokens, n_embd)

    # Check values
    torch.testing.assert_close(
        out_varlen.squeeze(0), out_sdpa_flat, atol=1e-4, rtol=1e-4
    )


if __name__ == "__main__":
    test_varlen_attn_gqa_parity()
