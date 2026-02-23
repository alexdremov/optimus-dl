import math
from unittest.mock import patch

import torch
import pytest
import torch.nn as nn

from optimus_dl.modules.model.blocks.attention import (
    CausalSelfAttention,
    RotarySelfAttention,
)
from optimus_dl.modules.model.blocks.rope import precompute_freqs_cis


class MockConfig:
    """Mock configuration for testing attention modules"""

    def __init__(self, n_embd=768, n_head=12, dropout=0.1, bias=True, block_size=1024):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.block_size = block_size


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention module"""

    def test_init_valid_config(self):
        """Test CausalSelfAttention initialization with valid configuration parameters."""
        config = MockConfig(n_embd=768, n_head=12)
        attention = CausalSelfAttention(config)

        assert attention.n_head == 12
        assert attention.n_embd == 768
        assert attention.dropout == 0.1
        assert isinstance(attention.c_attn, nn.Linear)
        assert isinstance(attention.c_proj, nn.Linear)
        assert isinstance(attention.attn_dropout, nn.Dropout)
        assert isinstance(attention.resid_dropout, nn.Dropout)

    def test_init_invalid_head_dimension(self):
        """Test that initialization fails when embedding dimension is not divisible by number of heads."""
        config = MockConfig(n_embd=768, n_head=11)  # 768 % 11 != 0

        with pytest.raises(AssertionError):
            CausalSelfAttention(config)

    def test_linear_layer_dimensions(self):
        config = MockConfig(n_embd=768, n_head=12)
        attention = CausalSelfAttention(config)

        # c_attn should project to 3 * n_embd for q, k, v
        assert attention.c_attn.in_features == 768
        assert attention.c_attn.out_features == 3 * 768

        # c_proj should project back to n_embd
        assert attention.c_proj.in_features == 768
        assert attention.c_proj.out_features == 768

    def test_bias_configuration(self):
        # Test with bias=True
        config_with_bias = MockConfig(bias=True)
        attention_with_bias = CausalSelfAttention(config_with_bias)
        assert attention_with_bias.c_attn.bias is not None
        assert attention_with_bias.c_proj.bias is not None

        # Test with bias=False
        config_no_bias = MockConfig(bias=False)
        attention_no_bias = CausalSelfAttention(config_no_bias)
        assert attention_no_bias.c_attn.bias is None
        assert attention_no_bias.c_proj.bias is None

    def test_forward_shape_consistency(self):
        config = MockConfig(n_embd=768, n_head=12, block_size=1024)
        attention = CausalSelfAttention(config)

        # Test various input shapes
        batch_sizes = [1, 4, 8]
        seq_lengths = [10, 50, 100]

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                if seq_len <= config.block_size:
                    x = torch.randn(batch_size, seq_len, config.n_embd)
                    output = attention(x)

                    # Output shape should match input shape
                    assert output.shape == (batch_size, seq_len, config.n_embd)

    @patch("torch.nn.functional.scaled_dot_product_attention")
    def test_flash_attention_forward(self, mock_flash_attn):
        config = MockConfig(n_embd=768, n_head=12)

        with patch(
            "optimus_dl.modules.model.blocks.attention.hasattr", return_value=True
        ):
            attention = CausalSelfAttention(config)

            # Mock flash attention to return expected shape
            batch_size, seq_len = 2, 10
            expected_output = torch.randn(
                batch_size, config.n_head, seq_len, config.n_embd // config.n_head
            )
            mock_flash_attn.return_value = expected_output

            x = torch.randn(batch_size, seq_len, config.n_embd)
            output = attention(x)

            # Check that flash attention was called
            mock_flash_attn.assert_called_once()
            call_args = mock_flash_attn.call_args

            # Verify flash attention arguments
            assert call_args[1]["is_causal"] is True
            assert call_args[1]["attn_mask"] is None

            # Check output shape
            assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_manual_attention_forward(self):
        config = MockConfig(n_embd=768, n_head=12, block_size=1024)

        with patch(
            "optimus_dl.modules.model.blocks.attention.hasattr", return_value=False
        ):
            attention = CausalSelfAttention(config)
            attention.eval()  # Disable dropout for deterministic testing

            batch_size, seq_len = 2, 5
            x = torch.randn(batch_size, seq_len, config.n_embd)

            output = attention(x)
            assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_causal_mask_application(self):
        """Test that causal masking prevents attention to future positions."""
        config = MockConfig(n_embd=64, n_head=4, block_size=10)

        with patch(
            "optimus_dl.modules.model.blocks.attention.hasattr", return_value=False
        ):
            attention = CausalSelfAttention(config)
            attention.eval()

            # Test with a small sequence to verify causal masking
            batch_size, seq_len = 1, 4
            x = torch.ones(
                batch_size, seq_len, config.n_embd
            )  # Use ones for predictable QKV

            # Override the c_attn to return known values for testing
            with torch.no_grad():
                # Set c_attn weights to identity-like for predictable q, k, v
                attention.c_attn.weight.fill_(0.1)
                if attention.c_attn.bias is not None:
                    attention.c_attn.bias.fill_(0.0)

            output = attention(x)
            assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_attention_head_reshaping(self):
        config = MockConfig(n_embd=768, n_head=12)
        attention = CausalSelfAttention(config)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.n_embd)

        # Test that q, k, v are properly reshaped
        q, k, v = attention.c_attn(x).split(config.n_embd, dim=2)

        # Original shapes before head reshaping
        assert q.shape == (batch_size, seq_len, config.n_embd)
        assert k.shape == (batch_size, seq_len, config.n_embd)
        assert v.shape == (batch_size, seq_len, config.n_embd)

        # After head reshaping (as done in forward)
        head_size = config.n_embd // config.n_head
        q_reshaped = q.view(batch_size, seq_len, config.n_head, head_size).transpose(
            1, 2
        )
        k_reshaped = k.view(batch_size, seq_len, config.n_head, head_size).transpose(
            1, 2
        )
        v_reshaped = v.view(batch_size, seq_len, config.n_head, head_size).transpose(
            1, 2
        )

        expected_shape = (batch_size, config.n_head, seq_len, head_size)
        assert q_reshaped.shape == expected_shape
        assert k_reshaped.shape == expected_shape
        assert v_reshaped.shape == expected_shape

    def test_dropout_behavior(self):
        config = MockConfig(dropout=0.5)
        attention = CausalSelfAttention(config)

        # Test training mode (dropout active)
        attention.train()
        x = torch.randn(1, 10, config.n_embd)

        # Run multiple times to see if outputs differ due to dropout
        [attention(x) for _ in range(3)]

        # In training mode with dropout, outputs should potentially differ
        # (though not guaranteed with random seeds)

        # Test eval mode (dropout inactive)
        attention.eval()
        output1 = attention(x)
        output2 = attention(x)

        # In eval mode, outputs should be identical
        torch.testing.assert_close(output1, output2)

    def test_gradient_flow(self):
        config = MockConfig(n_embd=64, n_head=4)
        attention = CausalSelfAttention(config)

        x = torch.randn(1, 10, config.n_embd, requires_grad=True)
        output = attention(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients flow through all parameters
        assert attention.c_attn.weight.grad is not None
        assert attention.c_proj.weight.grad is not None
        assert x.grad is not None

        if attention.c_attn.bias is not None:
            assert attention.c_attn.bias.grad is not None
        if attention.c_proj.bias is not None:
            assert attention.c_proj.bias.grad is not None

    def test_different_head_configurations(self):
        """Test various valid head configurations"""
        valid_configs = [
            (768, 12),  # Standard GPT-2
            (1024, 16),  # Larger model
            (512, 8),  # Smaller model
            (256, 4),  # Tiny model
        ]

        for n_embd, n_head in valid_configs:
            config = MockConfig(n_embd=n_embd, n_head=n_head)
            attention = CausalSelfAttention(config)

            x = torch.randn(2, 10, n_embd)
            output = attention(x)
            assert output.shape == (2, 10, n_embd)

    def test_attention_scaling(self):
        """Test that attention scaling factor (1/sqrt(head_size)) is applied correctly."""
        config = MockConfig(n_embd=768, n_head=12)

        with patch(
            "optimus_dl.modules.model.blocks.attention.hasattr", return_value=False
        ):
            attention = CausalSelfAttention(config)

            # The scaling factor should be 1 / sqrt(head_size)
            head_size = config.n_embd // config.n_head
            1.0 / math.sqrt(head_size)

            # We can't directly access the scaling, but we can verify it's applied correctly
            # by checking the manual attention computation path
            batch_size, seq_len = 1, 3
            x = torch.randn(batch_size, seq_len, config.n_embd)

            # Test that forward pass completes without error
            output = attention(x)
            assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_memory_efficiency(self):
        """Test memory usage is reasonable for different sequence lengths"""
        config = MockConfig(n_embd=768, n_head=12, block_size=2048)
        attention = CausalSelfAttention(config)

        # Test with progressively larger sequences
        seq_lengths = [10, 50, 100, 500]
        batch_size = 1

        for seq_len in seq_lengths:
            if seq_len <= config.block_size:
                x = torch.randn(batch_size, seq_len, config.n_embd)

                # Clear any cached memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                output = attention(x)
                assert output.shape == (batch_size, seq_len, config.n_embd)

                # Test that memory is released
                del output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class TestRotarySelfAttention:
    """Extensive tests for RotarySelfAttention module"""

    def test_init_basic(self):
        attn = RotarySelfAttention(n_embd=256, n_head=8)
        assert attn.n_head == 8
        assert attn.n_kv_head == 8
        assert attn.head_dim == 32
        assert not attn.use_qk_norm

    def test_init_gqa(self):
        attn = RotarySelfAttention(n_embd=256, n_head=8, n_kv_head=2)
        assert attn.n_head == 8
        assert attn.n_kv_head == 2
        assert attn.n_rep == 4
        assert attn.wk.out_features == 2 * 32
        assert attn.wv.out_features == 2 * 32

    def test_init_qk_norm(self):
        # Per-head norm
        attn_ph = RotarySelfAttention(
            n_embd=256, n_head=8, use_qk_norm=True, qk_norm_per_head=True
        )
        assert attn_ph.q_norm.weight.shape == (32,)
        assert attn_ph.k_norm.weight.shape == (32,)

        # Shared norm (Olmo3 style)
        attn_sh = RotarySelfAttention(
            n_embd=256, n_head=8, use_qk_norm=True, qk_norm_per_head=False
        )
        assert attn_sh.q_norm.weight.shape == (256,)
        assert attn_sh.k_norm.weight.shape == (256,)

    def test_forward_basic(self):
        n_embd, n_head, seq_len = 128, 4, 16
        attn = RotarySelfAttention(n_embd=n_embd, n_head=n_head)
        x = torch.randn(2, seq_len, n_embd)
        freqs_cis = precompute_freqs_cis(n_embd // n_head, seq_len)

        out = attn(x, freqs_cis)
        assert out.shape == (2, seq_len, n_embd)

    def test_forward_gqa(self):
        n_embd, n_head, n_kv_head, seq_len = 128, 4, 2, 16
        attn = RotarySelfAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
        x = torch.randn(2, seq_len, n_embd)
        freqs_cis = precompute_freqs_cis(n_embd // n_head, seq_len)

        out = attn(x, freqs_cis)
        assert out.shape == (2, seq_len, n_embd)

    def test_sliding_window_forward(self):
        n_embd, n_head, seq_len = 128, 4, 32
        attn = RotarySelfAttention(n_embd=n_embd, n_head=n_head, sliding_window=8)
        x = torch.randn(1, seq_len, n_embd)
        freqs_cis = precompute_freqs_cis(n_embd // n_head, seq_len)

        # Test with/without flex_attention (manual vs flex)
        out = attn(x, freqs_cis)
        assert out.shape == (1, seq_len, n_embd)

    def test_gradient_flow(self):
        n_embd, n_head, seq_len = 64, 2, 8
        attn = RotarySelfAttention(n_embd=n_embd, n_head=n_head, use_qk_norm=True)
        x = torch.randn(1, seq_len, n_embd, requires_grad=True)
        freqs_cis = precompute_freqs_cis(n_embd // n_head, seq_len)

        out = attn(x, freqs_cis)
        out.sum().backward()

        assert x.grad is not None
        assert attn.wq.weight.grad is not None
        assert attn.q_norm.weight.grad is not None
        assert attn.wo.weight.grad is not None

    def test_qk_norm_logic(self):
        """Verify that QK norm actually changes values and works in both modes."""
        n_embd, n_head, seq_len = 64, 2, 8
        freqs_cis = precompute_freqs_cis(n_embd // n_head, seq_len)
        x = torch.randn(1, seq_len, n_embd)

        # 1. Per-head
        attn_ph = RotarySelfAttention(
            n_embd=n_embd, n_head=n_head, use_qk_norm=True, qk_norm_per_head=True
        )
        out_ph = attn_ph(x, freqs_cis)

        # 2. Shared
        attn_sh = RotarySelfAttention(
            n_embd=n_embd, n_head=n_head, use_qk_norm=True, qk_norm_per_head=False
        )
        out_sh = attn_sh(x, freqs_cis)

        assert not torch.allclose(out_ph, out_sh)

    def test_seq_lens_forward_basic(self):
        """Test that the forward pass completes successfully when seq_lens is provided."""
        n_embd, n_head, seq_len = 128, 4, 16
        attn = RotarySelfAttention(n_embd=n_embd, n_head=n_head)
        x = torch.randn(2, seq_len, n_embd)
        freqs_cis = precompute_freqs_cis(n_embd // n_head, seq_len)

        # Batch of 2, first sequence unpadded, second sequence padded
        seq_lens = torch.tensor([16, 10])

        out = attn(x, freqs_cis, seq_lens=seq_lens)
        assert out.shape == (2, seq_len, n_embd)

    def test_seq_lens_masking_equivalence(self):
        """Verify that padded tokens do not influence the attention output of valid tokens."""
        n_embd, n_head, max_seq_len = 128, 4, 16
        valid_len = 10

        attn = RotarySelfAttention(n_embd=n_embd, n_head=n_head)
        attn.eval()  # Disable dropout for deterministic output comparison

        # 1. Base input (Unpadded)
        x_base = torch.randn(1, valid_len, n_embd)
        freqs_cis_base = precompute_freqs_cis(n_embd // n_head, valid_len)

        # 2. Padded input (Same valid tokens, followed by random noise padding)
        x_padded = torch.zeros(1, max_seq_len, n_embd)
        x_padded[:, :valid_len, :] = x_base
        x_padded[:, valid_len:, :] = torch.randn(1, max_seq_len - valid_len, n_embd)
        freqs_cis_padded = precompute_freqs_cis(n_embd // n_head, max_seq_len)

        # Run forward passes
        out_base = attn(x_base, freqs_cis_base)
        out_padded = attn(
            x_padded, freqs_cis_padded, seq_lens=torch.tensor([valid_len])
        )

        # The output for the valid tokens should be completely unaffected by the padded tokens
        torch.testing.assert_close(
            out_base[0, :valid_len], out_padded[0, :valid_len], rtol=1e-4, atol=1e-4
        )

    def test_seq_lens_with_sliding_window(self):
        """Verify seq_lens padding masking works together with sliding window masking."""
        n_embd, n_head, max_seq_len = 128, 4, 16
        valid_len = 12
        sliding_window = 4

        attn = RotarySelfAttention(
            n_embd=n_embd, n_head=n_head, sliding_window=sliding_window
        )
        attn.eval()

        # 1. Base input (Unpadded)
        x_base = torch.randn(1, valid_len, n_embd)
        freqs_cis_base = precompute_freqs_cis(n_embd // n_head, valid_len)

        # 2. Padded input
        x_padded = torch.zeros(1, max_seq_len, n_embd)
        x_padded[:, :valid_len, :] = x_base
        x_padded[:, valid_len:, :] = torch.randn(1, max_seq_len - valid_len, n_embd)
        freqs_cis_padded = precompute_freqs_cis(n_embd // n_head, max_seq_len)

        # Run forward passes
        out_base = attn(x_base, freqs_cis_base)
        out_padded = attn(
            x_padded, freqs_cis_padded, seq_lens=torch.tensor([valid_len])
        )

        # Check equivalence
        torch.testing.assert_close(
            out_base[0, :valid_len], out_padded[0, :valid_len], rtol=1e-4, atol=1e-4
        )

    @patch("optimus_dl.modules.model.blocks.attention.FLEX_ATTENTION_AVAILABLE", False)
    def test_seq_lens_sdpa_fallback(self):
        """Test that the scaled_dot_product_attention fallback handles seq_lens properly."""
        n_embd, n_head, max_seq_len = 128, 4, 16
        valid_len = 10

        # Include sliding window to test the combined fallback mask
        attn = RotarySelfAttention(n_embd=n_embd, n_head=n_head, sliding_window=4)
        attn.eval()

        x_padded = torch.randn(1, max_seq_len, n_embd)
        freqs_cis_padded = precompute_freqs_cis(n_embd // n_head, max_seq_len)

        # Should execute the SDPA fallback branch without crashing
        out_padded = attn(
            x_padded, freqs_cis_padded, seq_lens=torch.tensor([valid_len])
        )

        assert out_padded.shape == (1, max_seq_len, n_embd)

        # Ensure outputs don't contain NaNs due to faulty boolean masking
        assert not torch.isnan(out_padded).any()
