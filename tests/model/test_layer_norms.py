import torch
import torch.nn as nn

from optimus_dl.modules.model.blocks.layer_norms import (
    LayerNorm,
    RMSNorm,
)


class TestLayerNorm:
    """Tests for custom LayerNorm implementation"""

    def test_init_with_bias(self):
        layer_norm = LayerNorm(ndim=768, bias=True)

        assert isinstance(layer_norm.weight, nn.Parameter)
        assert isinstance(layer_norm.bias, nn.Parameter)
        assert layer_norm.weight.shape == (768,)
        assert layer_norm.bias.shape == (768,)

        # Check initial values
        assert torch.allclose(layer_norm.weight, torch.ones(768))
        assert torch.allclose(layer_norm.bias, torch.zeros(768))

    def test_init_without_bias(self):
        layer_norm = LayerNorm(ndim=768, bias=False)

        assert isinstance(layer_norm.weight, nn.Parameter)
        assert layer_norm.bias is None
        assert layer_norm.weight.shape == (768,)

        # Check initial values
        assert torch.allclose(layer_norm.weight, torch.ones(768))

    def test_forward_with_bias(self, device):
        layer_norm = LayerNorm(ndim=64, bias=True).to(device)

        # Test various input shapes
        inputs = [
            torch.randn(10, 64, device=device),  # 2D: (batch, features)
            torch.randn(5, 20, 64, device=device),  # 3D: (batch, seq, features)
            torch.randn(
                2, 8, 15, 64, device=device
            ),  # 4D: (batch, heads, seq, features)
        ]

        for x in inputs:
            output = layer_norm(x)
            assert output.shape == x.shape

            # Check normalization properties (approximately)
            # LayerNorm should normalize the last dimension
            normalized_dim = x.shape[-1]
            output_last_dim = output.view(-1, normalized_dim)

            # Mean should be close to 0, std close to 1
            mean = output_last_dim.mean(dim=-1)
            std = output_last_dim.std(dim=-1, unbiased=False)

            assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
            assert torch.allclose(std, torch.ones_like(std), atol=1e-4)

    def test_forward_without_bias(self, device):
        layer_norm = LayerNorm(ndim=64, bias=False).to(device)

        x = torch.randn(5, 10, 64, device=device)
        output = layer_norm(x)

        assert output.shape == x.shape

        # Check normalization properties
        output_reshaped = output.view(-1, 64)
        mean = output_reshaped.mean(dim=-1)
        std = output_reshaped.std(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4)

    def test_comparison_with_pytorch_layernorm(self, device):
        """Compare with PyTorch's built-in LayerNorm"""
        ndim = 128

        # Test with bias
        custom_ln = LayerNorm(ndim=ndim, bias=True).to(device)
        pytorch_ln = nn.LayerNorm(ndim, bias=True, eps=1e-5).to(device)

        # Set same parameters
        with torch.no_grad():
            pytorch_ln.weight.copy_(custom_ln.weight)
            pytorch_ln.bias.copy_(custom_ln.bias)

        x = torch.randn(10, 20, ndim, device=device)

        custom_output = custom_ln(x)
        pytorch_output = pytorch_ln(x)

        assert torch.allclose(custom_output, pytorch_output, atol=1e-6)

        # Test without bias
        custom_ln_no_bias = LayerNorm(ndim=ndim, bias=False).to(device)
        pytorch_ln_no_bias = nn.LayerNorm(ndim, bias=False, eps=1e-5).to(device)

        with torch.no_grad():
            pytorch_ln_no_bias.weight.copy_(custom_ln_no_bias.weight)

        custom_output_no_bias = custom_ln_no_bias(x)
        pytorch_output_no_bias = pytorch_ln_no_bias(x)

        assert torch.allclose(custom_output_no_bias, pytorch_output_no_bias, atol=1e-6)

    def test_gradient_flow(self, device):
        layer_norm = LayerNorm(ndim=64, bias=True).to(device)

        x = torch.randn(5, 10, 64, requires_grad=True, device=device)
        output = layer_norm(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert layer_norm.weight.grad is not None
        assert layer_norm.bias.grad is not None
        assert x.grad is not None

        # Gradients should have correct shapes
        assert layer_norm.weight.grad.shape == (64,)
        assert layer_norm.bias.grad.shape == (64,)
        assert x.grad.shape == x.shape

    def test_different_dimensions(self, device):
        """Test LayerNorm with various dimensions"""
        dimensions = [1, 16, 64, 256, 768, 1024, 2048]

        for dim in dimensions:
            layer_norm = LayerNorm(ndim=dim, bias=True).to(device)
            x = torch.randn(2, 10, dim, device=device)
            output = layer_norm(x)

            assert output.shape == (2, 10, dim)
            assert layer_norm.weight.shape == (dim,)
            assert layer_norm.bias.shape == (dim,)


class TestRMSNorm:
    """Tests for RMSNorm implementation"""

    def test_init(self):
        rms_norm = RMSNorm(dim=768, eps=1e-6)

        assert isinstance(rms_norm.weight, nn.Parameter)
        assert rms_norm.weight.shape == (768,)
        assert rms_norm.eps == 1e-6

        # Check initial values
        assert torch.allclose(rms_norm.weight, torch.ones(768))

    def test_init_custom_eps(self):
        eps_values = [1e-4, 1e-5, 1e-6, 1e-8]

        for eps in eps_values:
            rms_norm = RMSNorm(dim=64, eps=eps)
            assert rms_norm.eps == eps

    def test_norm_function(self, device):
        """Test the internal _norm function"""
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)

        x = torch.randn(5, 10, 64, device=device)
        normalized = rms_norm._norm(x)

        assert normalized.shape == x.shape

        # RMS should be approximately 1
        rms = torch.sqrt(normalized.pow(2).mean(-1, keepdim=True))
        expected_rms = torch.ones_like(rms)

        assert torch.allclose(rms, expected_rms, atol=1e-4)

    def test_forward(self, device):
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)

        # Test various input shapes
        inputs = [
            torch.randn(10, 64, device=device),  # 2D: (batch, features)
            torch.randn(5, 20, 64, device=device),  # 3D: (batch, seq, features)
            torch.randn(
                2, 8, 15, 64, device=device
            ),  # 4D: (batch, heads, seq, features)
        ]

        for x in inputs:
            output = rms_norm(x)
            assert output.shape == x.shape

            # Check that RMS is approximately 1 after scaling by weight
            # (when weight is initialized to ones)
            output_last_dim = output.view(-1, 64)
            rms = torch.sqrt(output_last_dim.pow(2).mean(-1))

            # RMS should be close to 1 (since weight is initialized to ones)
            assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_weight_scaling(self, device):
        """Test that weight parameter properly scales the output"""
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)

        # Set custom weight values
        with torch.no_grad():
            rms_norm.weight.fill_(2.0)

        x = torch.randn(5, 10, 64, device=device)
        output = rms_norm(x)

        # Output should be scaled by the weight
        normalized = rms_norm._norm(x.float()).type_as(x)
        expected_output = normalized * 2.0

        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_different_data_types(self, device):
        """Test RMSNorm with different input data types"""
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)

        # Test different dtypes
        dtypes = [torch.float32, torch.float16]
        if device.type == "cuda":
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            x = torch.randn(5, 10, 64, dtype=dtype, device=device)
            output = rms_norm(x)

            assert output.shape == x.shape
            # RMSNorm implementation converts to float internally, then back to original type
            # For half precision types, this may result in float32 output
            if dtype in [torch.float16, torch.bfloat16]:
                # Allow either the original dtype or float32
                assert output.dtype in [dtype, torch.float32]
            else:
                assert output.dtype == x.dtype

    def test_eps_stability(self, device):
        """Test numerical stability with different eps values"""
        x = torch.randn(5, 10, 64, device=device)

        # Test with very small values that might cause instability
        small_x = x * 1e-8

        eps_values = [1e-4, 1e-6, 1e-8, 1e-10]

        for eps in eps_values:
            rms_norm = RMSNorm(dim=64, eps=eps).to(device)
            output = rms_norm(small_x)

            # Should not produce NaN or Inf
            assert torch.isfinite(output).all()
            assert output.shape == small_x.shape

    def test_zero_input_handling(self, device):
        """Test behavior with zero or near-zero inputs"""
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)

        # Test zero input
        zero_x = torch.zeros(5, 10, 64, device=device)
        output = rms_norm(zero_x)

        # Should not produce NaN or Inf
        assert torch.isfinite(output).all()
        assert output.shape == zero_x.shape

        # Near-zero input
        tiny_x = torch.full((5, 10, 64), 1e-10, device=device)
        output_tiny = rms_norm(tiny_x)

        assert torch.isfinite(output_tiny).all()

    def test_gradient_flow(self, device):
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)

        x = torch.randn(5, 10, 64, requires_grad=True, device=device)
        output = rms_norm(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert rms_norm.weight.grad is not None
        assert x.grad is not None

        # Gradients should have correct shapes
        assert rms_norm.weight.grad.shape == (64,)
        assert x.grad.shape == x.shape

        # Gradients should be finite
        assert torch.isfinite(rms_norm.weight.grad).all()
        assert torch.isfinite(x.grad).all()

    def test_different_dimensions(self, device):
        """Test RMSNorm with various dimensions"""
        dimensions = [1, 16, 64, 256, 768, 1024, 2048, 4096]

        for dim in dimensions:
            rms_norm = RMSNorm(dim=dim, eps=1e-6).to(device)
            x = torch.randn(2, 10, dim, device=device)
            output = rms_norm(x)

            assert output.shape == (2, 10, dim)
            assert rms_norm.weight.shape == (dim,)

    def test_batch_independence(self, device):
        """Test that different samples in batch are normalized independently"""
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)

        # Create input with different scales for each batch element
        x = torch.randn(3, 10, 64, device=device)
        x[0] *= 10.0  # Large values
        x[1] *= 0.1  # Small values
        x[2] *= 1.0  # Normal values

        output = rms_norm(x)

        # Each batch element should be normalized independently
        for i in range(3):
            batch_output = output[i]  # Shape: (10, 64)
            rms = torch.sqrt(batch_output.pow(2).mean(-1))
            assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_comparison_with_manual_implementation(self, device):
        """Compare with manual RMSNorm implementation"""
        dim = 128
        eps = 1e-6

        rms_norm = RMSNorm(dim=dim, eps=eps).to(device)
        x = torch.randn(5, 10, dim, device=device)

        # Manual implementation
        def manual_rms_norm(x, weight, eps):
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            normalized = x / rms
            return normalized * weight

        # Get outputs
        module_output = rms_norm(x)
        manual_output = manual_rms_norm(x, rms_norm.weight, eps)

        assert torch.allclose(module_output, manual_output, atol=1e-6)

    def test_training_vs_eval_mode(self, device):
        """Test that RMSNorm behaves consistently in train/eval modes"""
        rms_norm = RMSNorm(dim=64, eps=1e-6).to(device)
        x = torch.randn(5, 10, 64, device=device)

        # Training mode
        rms_norm.train()
        train_output = rms_norm(x)

        # Eval mode
        rms_norm.eval()
        eval_output = rms_norm(x)

        # Outputs should be identical (RMSNorm has no mode-dependent behavior)
        assert torch.allclose(train_output, eval_output)


class TestLayerNormVsRMSNorm:
    """Comparative tests between LayerNorm and RMSNorm"""

    def test_output_differences(self, device):
        """Test that LayerNorm and RMSNorm produce different outputs"""
        dim = 64
        layer_norm = LayerNorm(ndim=dim, bias=False).to(device)
        rms_norm = RMSNorm(dim=dim, eps=1e-5).to(device)

        # Set same weight values
        with torch.no_grad():
            rms_norm.weight.copy_(layer_norm.weight)

        x = torch.randn(5, 10, dim, device=device)

        ln_output = layer_norm(x)
        rms_output = rms_norm(x)

        # Outputs should be different (LayerNorm centers around 0, RMSNorm doesn't)
        assert not torch.allclose(ln_output, rms_output, atol=1e-3)

    def test_normalization_properties(self, device):
        """Compare normalization properties of LayerNorm vs RMSNorm"""
        dim = 64
        layer_norm = LayerNorm(ndim=dim, bias=False).to(device)
        rms_norm = RMSNorm(dim=dim, eps=1e-5).to(device)

        x = torch.randn(5, 10, dim, device=device)

        ln_output = layer_norm(x)
        rms_output = rms_norm(x)

        # LayerNorm: mean ≈ 0, std ≈ 1
        ln_flat = ln_output.view(-1, dim)
        ln_mean = ln_flat.mean(dim=-1)
        ln_std = ln_flat.std(dim=-1, unbiased=False)

        assert torch.allclose(ln_mean, torch.zeros_like(ln_mean), atol=1e-5)
        assert torch.allclose(ln_std, torch.ones_like(ln_std), atol=1e-4)

        # RMSNorm: RMS ≈ 1 (but mean is not necessarily 0)
        rms_flat = rms_output.view(-1, dim)
        rms_rms = torch.sqrt(rms_flat.pow(2).mean(dim=-1))

        assert torch.allclose(rms_rms, torch.ones_like(rms_rms), atol=1e-4)
