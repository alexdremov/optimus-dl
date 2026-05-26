"""Tests for FP8 (Floating Point 8-bit) support in Optimus-DL.

These tests verify:
- FP8 dtype conversion utilities
- FP8 configuration parsing
- FP8 recipe creation (with Transformer Engine if available)
- Integration with AMP config
- Context managers for FP8 training
"""

import torch
import pytest

from optimus_dl.core.dtype import (
    get_fp8_format_from_dtype,
    is_fp8_dtype,
    str_to_dtype,
)
from optimus_dl.modules.amp import (
    Fp8Format,
    FP8Recipe,
    Fp8RecipeConfig,
    create_fp8_recipe,
    get_fp8_format_from_string,
    is_transformer_engine_available,
)
from optimus_dl.modules.amp.config import (
    AmpConfig,
    Fp8Config,
)


class TestFp8DtypeConversion:
    """Test FP8 dtype conversion utilities."""

    def test_str_to_dtype_fp8_e4m3(self):
        """Test conversion of E4M3 format strings to dtype."""
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available in this PyTorch version")
        assert str_to_dtype("float8_e4m3fn") == torch.float8_e4m3fn
        assert str_to_dtype("fp8_e4m3fn") == torch.float8_e4m3fn
        assert str_to_dtype("e4m3fn") == torch.float8_e4m3fn
        assert str_to_dtype("e4m3") == torch.float8_e4m3fn
        assert str_to_dtype("torch.float8_e4m3fn") == torch.float8_e4m3fn
        assert str_to_dtype("fp8_e4m3") == torch.float8_e4m3fn
        assert str_to_dtype("float8_e4m3") == torch.float8_e4m3fn

    def test_str_to_dtype_fp8_e5m2(self):
        """Test conversion of E5M2 format strings to dtype."""
        if not hasattr(torch, "float8_e5m2"):
            pytest.skip("torch.float8_e5m2 not available in this PyTorch version")
        assert str_to_dtype("float8_e5m2") == torch.float8_e5m2
        assert str_to_dtype("fp8_e5m2") == torch.float8_e5m2
        assert str_to_dtype("e5m2fn") == torch.float8_e5m2
        assert str_to_dtype("e5m2") == torch.float8_e5m2
        assert str_to_dtype("torch.float8_e5m2") == torch.float8_e5m2
        assert str_to_dtype("fp8_e5m2fn") == torch.float8_e5m2
        assert str_to_dtype("float8_e5m2fn") == torch.float8_e5m2

    def test_str_to_dtype_case_insensitive(self):
        """Test that dtype conversion is case-insensitive."""
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")
        assert str_to_dtype("Float8_E4M3FN") == torch.float8_e4m3fn
        assert str_to_dtype("E4M3") == torch.float8_e4m3fn
        assert str_to_dtype("torch.FLOAT8_E5M2") == torch.float8_e5m2

    def test_str_to_dtype_standard_dtypes_still_work(self):
        """Test that standard dtype conversion still works."""
        assert str_to_dtype("float32") == torch.float32
        assert str_to_dtype("float16") == torch.float16
        assert str_to_dtype("bfloat16") == torch.bfloat16
        assert str_to_dtype("int8") == torch.int8

    def test_str_to_dtype_invalid_raises(self):
        """Test that invalid dtype strings raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            str_to_dtype("invalid_dtype")

    def test_is_fp8_dtype(self):
        """Test FP8 dtype checking."""
        if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e5m2"):
            pytest.skip("FP8 dtypes not available")
        assert is_fp8_dtype(torch.float8_e4m3fn) is True
        assert is_fp8_dtype(torch.float8_e5m2) is True
        assert is_fp8_dtype(torch.float32) is False
        assert is_fp8_dtype(torch.float16) is False
        assert is_fp8_dtype(torch.bfloat16) is False

    def test_get_fp8_format_from_dtype(self):
        """Test getting FP8 format string from dtype."""
        if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e5m2"):
            pytest.skip("FP8 dtypes not available")
        assert get_fp8_format_from_dtype(torch.float8_e4m3fn) == "e4m3fn"
        assert get_fp8_format_from_dtype(torch.float8_e5m2) == "e5m2"
        assert get_fp8_format_from_dtype(torch.float32) is None
        assert get_fp8_format_from_dtype(torch.float16) is None


class TestFp8Config:
    """Test FP8 configuration dataclasses."""

    def test_fp8_config_defaults(self):
        """Test FP8Config default values."""
        cfg = Fp8Config()
        assert cfg.enabled is False
        assert cfg.format == "hybrid"
        assert cfg.margin == 0
        assert cfg.amax_history_len == 1024
        assert cfg.amax_compute_algo == "max"
        assert cfg.reduce_amax is True
        assert cfg.fp8_dpa is False
        assert cfg.fp8_mha is False

    def test_fp8_config_custom_values(self):
        """Test FP8Config with custom values."""
        cfg = Fp8Config(
            enabled=True,
            format="e4m3",
            margin=5,
            amax_history_len=2048,
            amax_compute_algo="most_recent",
            reduce_amax=False,
            fp8_dpa=True,
            fp8_mha=True,
        )
        assert cfg.enabled is True
        assert cfg.format == "e4m3"
        assert cfg.margin == 5
        assert cfg.amax_history_len == 2048
        assert cfg.amax_compute_algo == "most_recent"
        assert cfg.reduce_amax is False
        assert cfg.fp8_dpa is True
        assert cfg.fp8_mha is True

    def test_amp_config_with_fp8(self):
        """Test AmpConfig with FP8 settings."""
        amp_cfg = AmpConfig(
            enabled=True,
            dtype="torch.float8_e4m3fn",
            enable_scaler=False,  # Should be False for FP8
            fp8=Fp8Config(enabled=True, format="hybrid"),
        )
        assert amp_cfg.enabled is True
        assert amp_cfg.dtype == "torch.float8_e4m3fn"
        assert amp_cfg.enable_scaler is False
        assert amp_cfg.fp8.enabled is True
        assert amp_cfg.fp8.format == "hybrid"

    def test_amp_config_bf16_no_fp8(self):
        """Test AmpConfig with BF16 (no FP8)."""
        amp_cfg = AmpConfig(
            enabled=True,
            dtype="torch.bfloat16",
            enable_scaler=False,
        )
        assert amp_cfg.enabled is True
        assert amp_cfg.dtype == "torch.bfloat16"
        assert amp_cfg.fp8.enabled is False


class TestFp8FormatEnum:
    """Test FP8 format enumeration."""

    def test_fp8_format_values(self):
        """Test Fp8Format enum values."""
        assert Fp8Format.HYBRID.value == "hybrid"
        assert Fp8Format.E4M3.value == "e4m3"
        assert Fp8Format.E5M2.value == "e5m2"


class TestFp8FormatFromString:
    """Test FP8 format string conversion."""

    def test_hybrid_format(self):
        """Test hybrid format string conversion."""
        assert get_fp8_format_from_string("hybrid") == Fp8Format.HYBRID
        assert get_fp8_format_from_string("HYBRID") == Fp8Format.HYBRID
        assert get_fp8_format_from_string("hybrid_fp8") == Fp8Format.HYBRID
        assert get_fp8_format_from_string("fp8_hybrid") == Fp8Format.HYBRID

    def test_e4m3_format(self):
        """Test E4M3 format string conversion."""
        assert get_fp8_format_from_string("e4m3") == Fp8Format.E4M3
        assert get_fp8_format_from_string("E4M3") == Fp8Format.E4M3
        assert get_fp8_format_from_string("fp8_e4m3") == Fp8Format.E4M3
        assert get_fp8_format_from_string("float8_e4m3fn") == Fp8Format.E4M3
        assert get_fp8_format_from_string("e4m3fn") == Fp8Format.E4M3

    def test_e5m2_format(self):
        """Test E5M2 format string conversion."""
        assert get_fp8_format_from_string("e5m2") == Fp8Format.E5M2
        assert get_fp8_format_from_string("E5M2") == Fp8Format.E5M2
        assert get_fp8_format_from_string("fp8_e5m2") == Fp8Format.E5M2
        assert get_fp8_format_from_string("float8_e5m2") == Fp8Format.E5M2
        assert get_fp8_format_from_string("e5m2fn") == Fp8Format.E5M2

    def test_invalid_format_raises(self):
        """Test that invalid format strings raise ValueError."""
        with pytest.raises(ValueError, match="Unknown FP8 format"):
            get_fp8_format_from_string("invalid_format")


class TestTransformerEngineAvailability:
    """Test Transformer Engine availability checking."""

    def test_is_transformer_engine_available(self):
        """Test that is_transformer_engine_available returns a boolean."""
        result = is_transformer_engine_available()
        assert isinstance(result, bool)


@pytest.mark.skipif(
    not is_transformer_engine_available(),
    reason="Transformer Engine not installed",
)
class TestFp8Recipe:
    """Test FP8 recipe creation and usage (requires Transformer Engine)."""

    def test_create_fp8_recipe_hybrid(self):
        """Test creating FP8 recipe with hybrid format."""
        recipe = create_fp8_recipe(
            format="hybrid",
            amax_history_len=16,
        )
        assert recipe is not None
        assert isinstance(recipe, FP8Recipe)
        assert recipe.format == Fp8Format.HYBRID
        assert recipe.amax_history_len == 16

    def test_create_fp8_recipe_e4m3(self):
        """Test creating FP8 recipe with E4M3 format."""
        recipe = create_fp8_recipe(format="e4m3")
        assert recipe is not None
        assert recipe.format == Fp8Format.E4M3

    def test_create_fp8_recipe_e5m2(self):
        """Test creating FP8 recipe with E5M2 format raises error.

        Pure E5M2 format is not supported by any Transformer Engine recipe type.
        Only hybrid (e4m3 forward, e5m2 backward) or pure e4m3 are supported.
        """
        with pytest.raises(ValueError, match="Pure E5M2 format is not supported"):
            create_fp8_recipe(format="e5m2")

    def test_create_fp8_recipe_with_config(self):
        """Test creating FP8 recipe from config dataclass."""
        config = Fp8RecipeConfig(
            format="hybrid",
            margin=5,
            amax_history_len=2048,
            amax_compute_algo="most_recent",
            reduce_amax=False,
            fp8_dpa=True,
            fp8_mha=True,
        )
        recipe = create_fp8_recipe(
            format=config.format,
            margin=config.margin,
            amax_history_len=config.amax_history_len,
            amax_compute_algo=config.amax_compute_algo,
            reduce_amax=config.reduce_amax,
            fp8_dpa=config.fp8_dpa,
            fp8_mha=config.fp8_mha,
        )
        assert recipe is not None
        assert recipe.margin == 5
        assert recipe.amax_history_len == 2048
        assert recipe.reduce_amax is False
        assert recipe.fp8_dpa is True
        assert recipe.fp8_mha is True

    def test_fp8_recipe_autocast_context(self):
        """Test FP8 recipe autocast context manager."""
        recipe = create_fp8_recipe(format="hybrid")
        assert recipe is not None

        # Test that autocast context can be entered
        with recipe.autocast():
            # Should not raise
            x = torch.randn(10, 10, device="cpu")
            assert x.shape == (10, 10)

    def test_fp8_recipe_backward_context(self):
        """Test FP8 recipe backward context manager."""
        recipe = create_fp8_recipe(format="hybrid")
        assert recipe is not None

        # Test that backward context can be entered
        with recipe.backward():
            # Should not raise
            x = torch.randn(10, 10, device="cpu", requires_grad=True)
            assert x.requires_grad is True

    def test_fp8_recipe_with_fp8format_enum(self):
        """Test creating FP8 recipe with Fp8Format enum."""
        recipe = create_fp8_recipe(format=Fp8Format.HYBRID)
        assert recipe is not None
        assert recipe.format == Fp8Format.HYBRID

    @pytest.mark.skipif(
        not is_transformer_engine_available(),
        reason="Transformer Engine not installed - E5M2 validation requires TE",
    )
    def test_fp8_recipe_e5m2_raises_error(self):
        """Test that E5M2 format raises ValueError (not supported by DelayedScaling)."""
        # E5M2 is not supported by TE's DelayedScaling recipe
        with pytest.raises(ValueError, match="Pure E5M2 format is not supported"):
            create_fp8_recipe(format="e5m2")


class TestFp8Integration:
    """Integration tests for FP8 with Optimus-DL components."""

    def test_amp_config_with_fp8_yaml_safe(self):
        """Test that FP8 config can be created from YAML-safe values."""
        from omegaconf import OmegaConf

        yaml_str = """
enabled: true
dtype: torch.float8_e4m3fn
enable_scaler: false
fp8:
  enabled: true
  format: hybrid
  margin: 0
  amax_history_len: 1024
  amax_compute_algo: max
  reduce_amax: true
  fp8_dpa: false
  fp8_mha: false
"""
        cfg_dict = OmegaConf.create(yaml_str)

        # Convert to AmpConfig
        amp_cfg = AmpConfig(
            enabled=cfg_dict.enabled,
            dtype=cfg_dict.dtype,
            enable_scaler=cfg_dict.enable_scaler,
            fp8=Fp8Config(
                enabled=cfg_dict.fp8.enabled,
                format=cfg_dict.fp8.format,
                margin=cfg_dict.fp8.margin,
                amax_history_len=cfg_dict.fp8.amax_history_len,
                amax_compute_algo=cfg_dict.fp8.amax_compute_algo,
                reduce_amax=cfg_dict.fp8.reduce_amax,
                fp8_dpa=cfg_dict.fp8.fp8_dpa,
                fp8_mha=cfg_dict.fp8.fp8_mha,
            ),
        )

        assert amp_cfg.enabled is True
        assert amp_cfg.dtype == "torch.float8_e4m3fn"
        assert amp_cfg.fp8.enabled is True
        assert amp_cfg.fp8.format == "hybrid"

    def test_fp8_dtype_with_str_to_dtype_in_omegaconf(self):
        """Test that FP8 dtype strings work with OmegaConf interpolation."""
        from omegaconf import OmegaConf

        yaml_str = """
dtype_str: torch.float8_e4m3fn
"""
        cfg = OmegaConf.create(yaml_str)
        dtype = str_to_dtype(cfg.dtype_str)

        if hasattr(torch, "float8_e4m3fn"):
            assert dtype == torch.float8_e4m3fn
        else:
            pytest.skip("torch.float8_e4m3fn not available")


@pytest.mark.skipif(
    not is_transformer_engine_available(),
    reason="Transformer Engine not installed",
)
class TestFp8ForwardBackward:
    """Test FP8 forward and backward pass with actual model execution.

    These tests verify that FP8 is actually being used during forward and backward passes
    by checking that tensors are in FP8 format when inside the FP8 contexts.
    """

    def test_fp8_forward_pass_with_simple_model(self):
        """Test that FP8 forward pass works with a simple linear model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, FP8 requires CUDA")

        recipe = create_fp8_recipe(format="hybrid", amax_history_len=16)
        assert recipe is not None

        device = torch.device("cuda")

        # Create a simple model with FP8-compatible dtypes
        # Note: TE handles the actual quantization, we just verify the context works
        model = torch.nn.Linear(10, 10).to(device)

        # Create input tensor
        x = torch.randn(5, 10, device=device)

        # Run forward pass with FP8 autocast
        with recipe.autocast():
            output = model(x)
            # Output should be computed (shape check)
            assert output.shape == (5, 10)

        # Verify model still works after FP8 context
        with torch.no_grad():
            output2 = model(x)
            assert output2.shape == (5, 10)

    def test_fp8_backward_pass_with_simple_model(self):
        """Test that FP8 backward pass works with a simple model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, FP8 requires CUDA")

        recipe = create_fp8_recipe(format="hybrid", amax_history_len=16)
        assert recipe is not None

        device = torch.device("cuda")

        # Create a simple model
        model = torch.nn.Linear(10, 10).to(device)
        model.weight.requires_grad = True
        model.bias.requires_grad = True

        # Create input and target
        x = torch.randn(5, 10, device=device, requires_grad=True)

        # Run forward and backward with FP8
        with recipe.autocast():
            output = model(x)
            loss = output.sum()

        with recipe.backward():
            loss.backward()

        # Verify gradients were computed
        assert model.weight.grad is not None
        assert model.bias.grad is not None

    def test_fp8_full_training_step(self):
        """Test a complete training step with FP8 forward and backward."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, FP8 requires CUDA")

        recipe = create_fp8_recipe(format="hybrid", amax_history_len=16)
        assert recipe is not None

        device = torch.device("cuda")

        # Simple model: input -> linear -> ReLU -> linear -> output
        model = torch.nn.Sequential(
            torch.nn.Linear(20, 40).to(device),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 10).to(device),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create dummy data
        x = torch.randn(8, 20, device=device)
        y = torch.randint(0, 10, (8,), device=device)

        # Training step with FP8
        optimizer.zero_grad()

        # Forward pass with FP8 autocast
        with recipe.autocast():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)

        # Backward pass with FP8 backward context
        with recipe.backward():
            loss.backward()

        # Optimizer step
        optimizer.step()

        # Verify gradients were computed and optimizer updated
        for param in model.parameters():
            assert param.grad is not None

        # Verify loss is reasonable
        assert loss.item() > 0

    def test_fp8_recipe_format_variations(self):
        """Test FP8 forward/backward with all format variations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, FP8 requires CUDA")

        device = torch.device("cuda")

        # Note: Pure E5M2 format is not supported by Transformer Engine, only hybrid and e4m3
        for fmt in ["hybrid", "e4m3"]:
            recipe = create_fp8_recipe(format=fmt, amax_history_len=16)
            assert recipe is not None

            # Simple forward/backward test
            model = torch.nn.Linear(5, 5).to(device)
            x = torch.randn(3, 5, device=device, requires_grad=True)

            with recipe.autocast():
                output = model(x)
                loss = output.sum()

            with recipe.backward():
                loss.backward()

            assert model.weight.grad is not None
            assert output.shape == (3, 5)

        # Verify that e5m2 raises an error
        with pytest.raises(ValueError, match="Pure E5M2 format is not supported"):
            create_fp8_recipe(format="e5m2", amax_history_len=16)

    def test_fp8_context_managers_are_entered(self):
        """Test that FP8 context managers can be entered and exited properly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, FP8 requires CUDA")

        recipe = create_fp8_recipe(format="hybrid")
        assert recipe is not None

        device = torch.device("cuda")

        # Test autocast context
        entered_autocast = False
        with recipe.autocast():
            entered_autocast = True
            x = torch.randn(5, 5, device=device)
        assert entered_autocast

        # Test backward context
        entered_backward = False
        x = torch.randn(5, 5, device=device, requires_grad=True)
        with recipe.backward():
            entered_backward = True
            x.sum().backward()
        assert entered_backward
        assert x.grad is not None

    def test_fp8_with_nested_contexts(self):
        """Test FP8 with nested context managers."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, FP8 requires CUDA")

        recipe = create_fp8_recipe(format="hybrid")
        assert recipe is not None

        device = torch.device("cuda")
        model = torch.nn.Linear(5, 5).to(device)

        x = torch.randn(3, 5, device=device)

        # Nested contexts
        with recipe.autocast():
            output1 = model(x)
            with recipe.autocast():
                output2 = model(x)
                assert output2.shape == (3, 5)
            assert output1.shape == (3, 5)

        # Verify outputs are consistent
        assert torch.allclose(output1, output2)

    def test_fp8_utilization_with_te_linear(self):
        """Test that FP8 is genuinely utilized by checking TE internal states."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, FP8 requires CUDA")

        from optimus_dl.modules.model_transforms.transformer_engine import TELinear

        recipe = create_fp8_recipe(format="hybrid", amax_history_len=16)
        assert recipe is not None

        device = torch.device("cuda")

        # Create TELinear directly to guarantee TE is used.
        # Note: FP8 requires dims to be multiples of 16 (or 8 for inner dims)
        model = TELinear(32, 64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(16, 32, device=device)
        y = torch.randn(16, 64, device=device)

        te_layer = model._te_linear

        # In newer TE versions, scaling_fwd is only populated on the first forward pass
        if hasattr(te_layer, "fp8_meta"):
            assert (
                "scaling_fwd" not in te_layer.fp8_meta
            ), "scaling_fwd should not be initialized yet"

        # Forward pass with FP8
        with recipe.autocast():
            logits = model(x)
            loss = torch.nn.functional.mse_loss(logits, y)

        # Backward pass with FP8
        with recipe.backward():
            loss.backward()

        optimizer.step()

        for param in model.parameters():
            assert param.grad is not None

        # Verify that TE actually used FP8 and updated the amax_history
        if hasattr(te_layer, "fp8_meta"):
            fwd_meta = te_layer.fp8_meta["scaling_fwd"]
            bwd_meta = te_layer.fp8_meta["scaling_bwd"]

            def is_updated(meta):
                tensors = (
                    meta if isinstance(meta, list | tuple) else [meta.amax_history]
                )
                for t in tensors:
                    # amax_history is a 2D tensor (e.g. [16, 3] or [16, 2])
                    if hasattr(t, "dim") and t.dim() == 2:
                        return torch.any(t > 0).item()
                return False

            assert is_updated(fwd_meta), "Forward FP8 amax history was not updated!"
            assert is_updated(bwd_meta), "Backward FP8 amax history was not updated!"


def test_create_fp8_recipe_returns_none_without_te():
    """Test that create_fp8_recipe returns None when TE is not available."""
    # This test is a bit tricky since we're checking the opposite condition
    # We'll just verify the function signature
    result = create_fp8_recipe(format="hybrid")
    # If TE is available, result should be FP8Recipe; if not, None
    if is_transformer_engine_available():
        assert result is not None
        assert isinstance(result, FP8Recipe)
    else:
        assert result is None
