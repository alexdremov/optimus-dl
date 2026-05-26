"""Tests for Transformer Engine model transform."""

import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

import torch
import torch.nn as nn

from optimus_dl.modules.model_transforms.transformer_engine import (
    TEDotProductAttention,
    TELinear,
    TENorm,
    TransformerEngineTransform,
    TransformerEngineTransformConfig,
    _convert_layernorm,
    _convert_linear,
    _convert_rmsnorm,
    is_transformer_engine_available,
)


class TestTEAvailability(unittest.TestCase):
    """Test TE availability detection."""

    def test_availability(self):
        """Test that availability is correctly detected."""
        available = is_transformer_engine_available()
        # This will be False since we only have stubs/meta package
        # The actual availability depends on the environment
        self.assertIsInstance(available, bool)


class TestTEModuleWrappers(unittest.TestCase):
    """Test TE module wrapper classes."""

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    @patch.dict(
        "sys.modules",
        {"transformer_engine": MagicMock(), "transformer_engine.pytorch": MagicMock()},
    )
    def test_te_linear_creation(self):
        """Test TELinear wrapper creation."""
        mock_te_module = MagicMock()
        mock_te_layer = MagicMock()
        mock_te_layer.weight = nn.Parameter(torch.randn(10, 20))
        mock_te_layer.bias = nn.Parameter(torch.randn(10))
        mock_te_module.Linear.return_value = mock_te_layer

        with patch.dict("sys.modules", {"transformer_engine.pytorch": mock_te_module}):
            linear = TELinear(20, 10, bias=True)
            self.assertEqual(linear.in_features, 20)
            self.assertEqual(linear.out_features, 10)
            self.assertTrue(linear.use_bias)
            mock_te_module.Linear.assert_called_once()

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    @patch.dict(
        "sys.modules",
        {"transformer_engine": MagicMock(), "transformer_engine.pytorch": MagicMock()},
    )
    def test_te_norm_rmsnorm_creation(self):
        """Test TENorm wrapper creation with RMSNorm."""
        mock_te_module = MagicMock()
        mock_norm = MagicMock()
        mock_te_module.RMSNorm.return_value = mock_norm

        with patch.dict("sys.modules", {"transformer_engine.pytorch": mock_te_module}):
            norm = TENorm(normalized_shape=100, eps=1e-6, normalization="RMSNorm")
            self.assertEqual(norm.normalized_shape, 100)
            self.assertEqual(norm.eps, 1e-6)
            self.assertEqual(norm.normalization, "RMSNorm")
            mock_te_module.RMSNorm.assert_called_once()

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    @patch.dict(
        "sys.modules",
        {"transformer_engine": MagicMock(), "transformer_engine.pytorch": MagicMock()},
    )
    def test_te_norm_layernorm_creation(self):
        """Test TENorm wrapper creation with LayerNorm."""
        mock_te_module = MagicMock()
        mock_norm = MagicMock()
        mock_te_module.LayerNorm.return_value = mock_norm

        with patch.dict("sys.modules", {"transformer_engine.pytorch": mock_te_module}):
            norm = TENorm(normalized_shape=100, eps=1e-5, normalization="LayerNorm")
            self.assertEqual(norm.normalized_shape, 100)
            self.assertEqual(norm.eps, 1e-5)
            self.assertEqual(norm.normalization, "LayerNorm")
            mock_te_module.LayerNorm.assert_called_once()

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    @patch.dict(
        "sys.modules",
        {"transformer_engine": MagicMock(), "transformer_engine.pytorch": MagicMock()},
    )
    def test_te_attention_creation(self):
        """Test TEDotProductAttention wrapper creation."""
        mock_te_module = MagicMock()
        mock_attn = MagicMock()
        mock_te_module.DotProductAttention.return_value = mock_attn

        with patch.dict("sys.modules", {"transformer_engine.pytorch": mock_te_module}):
            attn = TEDotProductAttention(num_heads=32, num_kv_heads=8, head_dim=128)
            self.assertEqual(attn.num_heads, 32)
            self.assertEqual(attn.num_kv_heads, 8)
            self.assertEqual(attn.head_dim, 128)
            mock_te_module.DotProductAttention.assert_called_once()


class TestModuleConversion(unittest.TestCase):
    """Test module conversion functions."""

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    def test_convert_linear(self):
        """Test Linear to TELinear conversion."""
        # Mock TELinear class
        mock_te_linear = MagicMock()
        mock_te_linear._te_linear.weight = nn.Parameter(torch.randn(10, 20))
        mock_te_linear._te_linear.bias = nn.Parameter(torch.randn(10))

        with patch(
            "optimus_dl.modules.model_transforms.transformer_engine.TELinear",
            return_value=mock_te_linear,
        ):
            original = nn.Linear(20, 10, bias=True)
            original.weight.data = torch.randn(10, 20)
            original.bias.data = torch.randn(10)

            converted = _convert_linear(original)
            self.assertIsNotNone(converted)

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    def test_convert_rmsnorm(self):
        """Test RMSNorm to TENorm conversion."""

        # Create a simple RMSNorm-like module
        class SimpleRMSNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = 1e-6

        original = SimpleRMSNorm(100)
        mock_te_norm = MagicMock()

        with patch(
            "optimus_dl.modules.model_transforms.transformer_engine.TENorm",
            return_value=mock_te_norm,
        ):
            converted = _convert_rmsnorm(original)
            self.assertIsNotNone(converted)

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    def test_convert_layernorm(self):
        """Test LayerNorm to TENorm conversion."""
        original = nn.LayerNorm(100, eps=1e-5)
        mock_te_norm = MagicMock()

        with patch(
            "optimus_dl.modules.model_transforms.transformer_engine.TENorm",
            return_value=mock_te_norm,
        ):
            converted = _convert_layernorm(original)
            self.assertIsNotNone(converted)


class TestConfig(unittest.TestCase):
    """Test TransformerEngineTransformConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = TransformerEngineTransformConfig()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.replace_linear)
        self.assertTrue(cfg.replace_norm)
        self.assertFalse(cfg.replace_attention)
        self.assertTrue(cfg.replace_inplace)
        self.assertFalse(cfg.verbose)
        self.assertEqual(cfg.linear_kwargs, {})
        self.assertEqual(cfg.norm_kwargs, {})
        self.assertEqual(cfg.attention_kwargs, {})

    def test_config_with_kwargs(self):
        """Test configuration with custom kwargs."""
        cfg = TransformerEngineTransformConfig(
            enabled=True,
            replace_linear=True,
            replace_norm=False,
            linear_kwargs={"sequence_parallel": True},
            norm_kwargs={"zero_centered_gamma": False},
        )
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.replace_linear)
        self.assertFalse(cfg.replace_norm)
        self.assertEqual(cfg.linear_kwargs, {"sequence_parallel": True})
        self.assertEqual(cfg.norm_kwargs, {"zero_centered_gamma": False})


class TestTransform(unittest.TestCase):
    """Test TransformerEngineTransform."""

    def test_disabled_transform(self):
        """Test that disabled transform does nothing."""
        cfg = TransformerEngineTransformConfig(enabled=False)
        transform = TransformerEngineTransform(cfg)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 20)
                self.norm = nn.LayerNorm(20)

            def forward(self, x):
                return self.norm(self.linear(x))

        model = SimpleModel()
        original_linear_type = type(model.linear)
        original_norm_type = type(model.norm)

        result = transform.apply(model)

        # Model should be unchanged
        self.assertIs(result, model)
        self.assertEqual(type(result.linear), original_linear_type)
        self.assertEqual(type(result.norm), original_norm_type)

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    def test_transform_replaces_modules(self):
        """Test that transform replaces modules when TE is available."""

        # Create mock TE modules that are actual nn.Module instances
        class MockTELinear(nn.Module):
            def __init__(self, in_features, out_features, bias=True, **kwargs):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.use_bias = bias
                self._te_linear = MagicMock()

            def forward(self, x):
                return self._te_linear(x)

        class MockTENorm(nn.Module):
            def __init__(
                self, normalized_shape, eps=1e-6, normalization="RMSNorm", **kwargs
            ):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps
                self.normalization = normalization
                self._te_norm = MagicMock()

            def forward(self, x):
                return self._te_norm(x)

        with (
            patch(
                "optimus_dl.modules.model_transforms.transformer_engine.TELinear",
                MockTELinear,
            ),
            patch(
                "optimus_dl.modules.model_transforms.transformer_engine.TENorm",
                MockTENorm,
            ),
        ):

            cfg = TransformerEngineTransformConfig(
                enabled=True,
                replace_linear=True,
                replace_norm=True,
                replace_inplace=True,
            )
            transform = TransformerEngineTransform(cfg)

            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 20)
                    self.norm = nn.LayerNorm(20)

                def forward(self, x):
                    return self.norm(self.linear(x))

            model = SimpleModel()
            result = transform.apply(model)

            # Should have replaced modules
            self.assertIs(result, model)
            # Check that replacements were made
            self.assertIsInstance(result.linear, MockTELinear)
            self.assertIsInstance(result.norm, MockTENorm)

    def test_transform_verbose(self):
        """Test verbose logging during transform."""
        cfg = TransformerEngineTransformConfig(
            enabled=True,
            replace_linear=True,
            replace_norm=True,
            replace_inplace=True,
            verbose=True,
        )
        transform = TransformerEngineTransform(cfg)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 20)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        # This should work without errors even if TE is not available
        result = transform.apply(model)
        self.assertIs(result, model)


class TestNestedModuleReplacement(unittest.TestCase):
    """Test module replacement in nested structures."""

    @patch("optimus_dl.modules.model_transforms.transformer_engine._HAVE_TE", True)
    def test_nested_replacement(self):
        """Test that nested modules are also replaced."""

        # Create mock TE modules that are actual nn.Module instances
        class MockTELinear(nn.Module):
            def __init__(self, in_features, out_features, bias=True, **kwargs):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.use_bias = bias
                self._te_linear = MagicMock()

            def forward(self, x):
                return self._te_linear(x)

        class MockTENorm(nn.Module):
            def __init__(
                self, normalized_shape, eps=1e-6, normalization="RMSNorm", **kwargs
            ):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps
                self.normalization = normalization
                self._te_norm = MagicMock()

            def forward(self, x):
                return self._te_norm(x)

        with (
            patch(
                "optimus_dl.modules.model_transforms.transformer_engine.TELinear",
                MockTELinear,
            ),
            patch(
                "optimus_dl.modules.model_transforms.transformer_engine.TENorm",
                MockTENorm,
            ),
        ):

            cfg = TransformerEngineTransformConfig(
                enabled=True,
                replace_linear=True,
                replace_norm=True,
                replace_inplace=True,
            )
            transform = TransformerEngineTransform(cfg)

            class InnerBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(10, 20)
                    self.norm = nn.LayerNorm(20)
                    self.linear2 = nn.Linear(20, 10)

                def forward(self, x):
                    return self.linear2(self.norm(self.linear1(x)))

            class OuterModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.block1 = InnerBlock()
                    self.block2 = InnerBlock()

                def forward(self, x):
                    return self.block2(self.block1(x))

            model = OuterModel()
            result = transform.apply(model)

            # Should have traversed all nested modules and replaced them
            self.assertIs(result, model)
            # Check nested replacements
            self.assertIsInstance(result.block1.linear1, MockTELinear)
            self.assertIsInstance(result.block1.norm, MockTENorm)
            self.assertIsInstance(result.block1.linear2, MockTELinear)
            self.assertIsInstance(result.block2.linear1, MockTELinear)
            self.assertIsInstance(result.block2.norm, MockTENorm)
            self.assertIsInstance(result.block2.linear2, MockTELinear)


if __name__ == "__main__":
    unittest.main()
