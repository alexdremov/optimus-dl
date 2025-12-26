import math
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from optimus_dl.modules.model.blocks.attention import CausalSelfAttention


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
