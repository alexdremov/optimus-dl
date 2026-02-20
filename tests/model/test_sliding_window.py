import torch
import pytest

from optimus_dl.modules.model.blocks.attention import (
    FLEX_ATTENTION_AVAILABLE,
    sliding_window_mask,
)


# Only run if flex_attention is available
@pytest.mark.skipif(not FLEX_ATTENTION_AVAILABLE, reason="flex_attention not available")
class TestSlidingWindowConsistency:
    """Compare flex_attention sliding window with manual mask SDPA."""

    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    @pytest.mark.parametrize("window_size", [16, 32, 64])
    @pytest.mark.parametrize("n_head", [4])
    @pytest.mark.parametrize("head_dim", [32])
    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_flex_vs_manual_mask(self, seq_len, window_size, n_head, head_dim, device):
        """Verify that flex_attention produces same results as manual masking."""
        batch_size = 1

        # Inputs
        q = torch.randn(
            batch_size, n_head, seq_len, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn(
            batch_size, n_head, seq_len, head_dim, device=device, dtype=torch.float32
        )
        v = torch.randn(
            batch_size, n_head, seq_len, head_dim, device=device, dtype=torch.float32
        )

        # 1. Manual Mask with SDPA
        q_idx = torch.arange(seq_len, device=device).view(-1, 1)
        kv_idx = torch.arange(seq_len, device=device).view(1, -1)
        mask = (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)

        out_manual = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False
        )

        # 2. Flex Attention
        from functools import partial

        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        mask_fn = partial(sliding_window_mask, window_size=window_size)
        block_mask = create_block_mask(
            mask_fn, None, None, seq_len, seq_len, device=device
        )

        # flex_attention usually expects being compiled, but for testing we can call it directly
        out_flex = flex_attention(q, k, v, block_mask=block_mask)

        # Compare
        torch.testing.assert_close(out_manual, out_flex, atol=1e-5, rtol=1e-5)

    def test_sliding_window_mask_logic(self):
        """Unit test for the mask function itself."""
        window_size = 2
        assert sliding_window_mask(None, None, 0, 0, window_size)
        assert sliding_window_mask(None, None, 1, 0, window_size)
        assert sliding_window_mask(None, None, 1, 1, window_size)
        assert not sliding_window_mask(None, None, 2, 0, window_size)
        assert sliding_window_mask(None, None, 2, 1, window_size)
        assert sliding_window_mask(None, None, 2, 2, window_size)
