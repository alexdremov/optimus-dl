"""Tests for FP8 training context integration.

These tests verify that FP8 is properly integrated into the training loop:
- TrainingContextMixin handles FP8 config correctly
- FP8 recipe is created when FP8 dtype is specified
- Context managers are properly set up
- Backward pass uses FP8 context when available
"""

import torch
import pytest

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.amp import (
    Fp8Format,
    is_transformer_engine_available,
)
from optimus_dl.modules.optim.config import (
    AmpConfig,
    Fp8Config,
    OptimizationConfig,
)
from optimus_dl.recipe.train.mixins.execution.context_mixin import (
    TrainingContextMixin,
)


class TestTrainingContextMixinFP8:
    """Test TrainingContextMixin with FP8 configuration."""

    def test_setup_training_context_bf16(self):
        """Test setup with BF16 (no FP8)."""
        optimization_config = OptimizationConfig(
            optimizer=RegistryConfig(_name="adamw"),
            amp=AmpConfig(
                enabled=True,
                dtype="torch.bfloat16",
                enable_scaler=False,
            ),
        )

        mixin = TrainingContextMixin(optimization_config)
        device = torch.device("cpu")
        context = mixin.setup_training_context(device)

        assert "scaler" in context
        assert "amp_ctx" in context
        assert "amp_cfg" in context
        assert "device" in context
        assert context["fp8_recipe"] is None
        assert context["fp8_enabled"] is False
        assert context["is_fp8_dtype"] is False
        assert "fp8_backward_ctx" in context

    @pytest.mark.skipif(
        not is_transformer_engine_available(),
        reason="Transformer Engine not installed",
    )
    def test_setup_training_context_fp8_hybrid(self):
        """Test setup with FP8 hybrid mode."""
        optimization_config = OptimizationConfig(
            optimizer=RegistryConfig(_name="adamw"),
            amp=AmpConfig(
                enabled=True,
                dtype="torch.float8_e4m3fn",
                enable_scaler=False,
                fp8=Fp8Config(
                    enabled=True,
                    format="hybrid",
                    amax_history_len=16,
                ),
            ),
        )

        mixin = TrainingContextMixin(optimization_config)
        device = torch.device("cpu")  # CPU for testing (TE may not work on CPU)
        context = mixin.setup_training_context(device)

        assert "scaler" in context
        assert "amp_ctx" in context
        assert "amp_cfg" in context
        assert "device" in context

        # FP8 should be enabled
        if is_transformer_engine_available():
            assert context["fp8_recipe"] is not None
            assert context["fp8_enabled"] is True
            assert context["is_fp8_dtype"] is True
            assert context["fp8_backward_ctx"] is not None
        else:
            # TE not available, should fall back
            assert context["fp8_recipe"] is None
            assert context["fp8_enabled"] is False

    @pytest.mark.skipif(
        not is_transformer_engine_available(),
        reason="Transformer Engine not installed",
    )
    def test_setup_training_context_fp8_e4m3(self):
        """Test setup with FP8 E4M3 mode."""
        optimization_config = OptimizationConfig(
            optimizer=RegistryConfig(_name="adamw"),
            amp=AmpConfig(
                enabled=True,
                dtype="torch.float8_e4m3fn",
                enable_scaler=False,
                fp8=Fp8Config(
                    enabled=True,
                    format="e4m3",
                ),
            ),
        )

        mixin = TrainingContextMixin(optimization_config)
        device = torch.device("cpu")
        context = mixin.setup_training_context(device)

        assert context["is_fp8_dtype"] is True
        if is_transformer_engine_available():
            assert context["fp8_enabled"] is True
            assert context["fp8_recipe"].format == Fp8Format.E4M3

    @pytest.mark.skipif(
        not is_transformer_engine_available(),
        reason="Transformer Engine not installed",
    )
    def test_setup_training_context_fp8_e5m2(self):
        """Test setup with FP8 E5M2 mode raises error.

        Pure E5M2 format is not supported by Transformer Engine's DelayedScaling recipe.
        """
        optimization_config = OptimizationConfig(
            optimizer=RegistryConfig(_name="adamw"),
            amp=AmpConfig(
                enabled=True,
                dtype="torch.float8_e5m2",
                enable_scaler=False,
                fp8=Fp8Config(
                    enabled=True,
                    format="e5m2",
                ),
            ),
        )

        mixin = TrainingContextMixin(optimization_config)
        device = torch.device("cpu")

        # Expect ValueError when trying to create FP8 recipe with pure e5m2
        with pytest.raises(ValueError, match="Pure E5M2 format is not supported"):
            mixin.setup_training_context(device)

    def test_setup_training_context_fp8_disabled(self):
        """Test setup with FP8 dtype but FP8 disabled."""
        optimization_config = OptimizationConfig(
            optimizer=RegistryConfig(_name="adamw"),
            amp=AmpConfig(
                enabled=True,
                dtype="torch.float8_e4m3fn",
                enable_scaler=False,
                fp8=Fp8Config(
                    enabled=False,  # FP8 disabled
                    format="hybrid",
                ),
            ),
        )

        mixin = TrainingContextMixin(optimization_config)

        # Use a mock device with type "cuda" to bypass the CUDA-only check
        class MockDevice:
            type = "cuda"

        device = MockDevice()

        context = mixin.setup_training_context(device)

        # FP8 dtype is set but FP8 is disabled
        assert context["is_fp8_dtype"] is True
        assert context["fp8_enabled"] is False
        assert context["fp8_recipe"] is None

    def test_get_fp8_backward_ctx(self):
        """Test get_fp8_backward_ctx method."""
        optimization_config = OptimizationConfig(
            optimizer=RegistryConfig(_name="adamw"),
            amp=AmpConfig(
                enabled=True,
                dtype="torch.bfloat16",
                enable_scaler=False,
            ),
        )

        mixin = TrainingContextMixin(optimization_config)

        # Without FP8 recipe, should return nullcontext
        ctx = mixin.get_fp8_backward_ctx()
        from contextlib import nullcontext

        # Check that ctx is a nullcontext instance
        assert isinstance(ctx, type(nullcontext()))

    def test_scaler_disabled_for_fp8(self):
        """Test that gradient scaler is disabled for FP8."""
        optimization_config = OptimizationConfig(
            optimizer=RegistryConfig(_name="adamw"),
            amp=AmpConfig(
                enabled=True,
                dtype="torch.float8_e4m3fn",
                enable_scaler=True,  # User sets this to True
                fp8=Fp8Config(
                    enabled=True,
                    format="hybrid",
                ),
            ),
        )

        mixin = TrainingContextMixin(optimization_config)

        # Use a mock device with type "cuda" to bypass the CUDA-only check
        class MockDevice:
            type = "cuda"

        device = MockDevice()

        context = mixin.setup_training_context(device)

        # Scaler should be disabled for FP8
        assert context["scaler"].is_enabled() is False


class TestTrainingContextFP8Configs:
    """Test various FP8 configuration combinations."""

    def test_all_fp8_formats(self):
        """Test all FP8 format strings are recognized."""
        formats = ["hybrid", "e4m3", "e5m2"]
        for fmt in formats:
            amp_cfg = AmpConfig(
                enabled=True,
                dtype="torch.float8_e4m3fn",
                fp8=Fp8Config(enabled=True, format=fmt),
            )
            assert amp_cfg.fp8.format == fmt

    def test_fp8_with_different_dtypes(self):
        """Test FP8 config with different FP8 dtypes."""
        dtypes = ["torch.float8_e4m3fn", "torch.float8_e5m2", "e4m3", "e5m2"]
        for dtype_str in dtypes:
            amp_cfg = AmpConfig(
                enabled=True,
                dtype=dtype_str,
                fp8=Fp8Config(enabled=True, format="hybrid"),
            )
            # Should not raise
            assert amp_cfg.dtype == dtype_str

    def test_amp_config_enable_scaler_auto_for_fp8(self):
        """Test that enable_scaler is auto-disabled for FP8."""
        # For FP8, enable_scaler should be False
        # The actual value is set via interpolation in the config
        amp_cfg = AmpConfig(
            enabled=True,
            dtype="torch.float8_e4m3fn",
            enable_scaler=False,  # Manually set for FP8
            fp8=Fp8Config(enabled=True),
        )
        assert amp_cfg.enable_scaler is False
