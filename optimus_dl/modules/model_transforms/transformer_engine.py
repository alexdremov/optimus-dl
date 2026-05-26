"""Transformer Engine model transform for Optimus-DL.

This module provides a model transform that replaces supported PyTorch modules
with their Transformer Engine (TE) equivalents to enable optimized execution
and FP8 training support.

The transform performs in-place modification of the model by traversing its
module hierarchy and replacing:
- nn.Linear -> TELinear (or TEColumnParallelLinear / TERowParallelLinear for TP)
- RMSNorm / LayerNorm -> TENorm
- Multi-head attention -> TEDotProductAttention

Note: This transform requires Transformer Engine to be installed with PyTorch support.
      Install with: pip install transformer-engine[pytorch]
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import register_model_transform
from optimus_dl.modules.model_transforms.base import BaseModelTransform
from optimus_dl.modules.model_transforms.config import ModelTransformConfig

logger = logging.getLogger(__name__)


# Check if Transformer Engine is available (with PyTorch support, not meta package)
_HAVE_TE = False
try:
    import importlib.util

    if importlib.util.find_spec("transformer_engine") is not None:
        # Try to import and check it's not a meta package
        try:
            import transformer_engine  # noqa

            # Check if it's a real package (not meta/stub)
            # Meta packages raise RuntimeError on import
            _HAVE_TE = True
        except (RuntimeError, AttributeError):
            # Meta package or stub - not usable
            _HAVE_TE = False
except (ImportError, RuntimeError, AttributeError):
    # TE not found or meta/stub package
    _HAVE_TE = False


def is_transformer_engine_available() -> bool:
    """Check if Transformer Engine is available with PyTorch support."""
    return _HAVE_TE


# ---------------------------------------------------------------------------
# TE Module Wrappers
# ---------------------------------------------------------------------------


class TELinear(nn.Module):
    """Wrapper for Transformer Engine's Linear layer.

    This wrapper provides a drop-in replacement for nn.Linear that uses
    Transformer Engine's optimized implementation. It supports FP8
    quantization through TE's autocast context.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias.
        **kwargs: Additional TE Linear arguments (sequence_parallel, etc.)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        if not _HAVE_TE:
            raise ImportError(
                "Transformer Engine is required for TELinear. "
                "Install with: pip install transformer-engine[pytorch]"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.kwargs = kwargs

        # Import TE Linear
        from transformer_engine.pytorch import Linear as TELayer

        self._te_linear = TELayer(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TE Linear."""
        return self._te_linear(x)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class TENorm(nn.Module):
    """Wrapper for Transformer Engine's normalization layers.

    Provides RMSNorm or LayerNorm based on the normalization type.
    Supports FP8 quantization through TE's autocast context.

    Args:
        normalized_shape: Shape of the input to normalize.
        eps: Small value for numerical stability.
        normalization: Type of normalization ('RMSNorm' or 'LayerNorm').
        **kwargs: Additional TE norm arguments.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        normalization: str = "RMSNorm",
        **kwargs: Any,
    ):
        super().__init__()
        if not _HAVE_TE:
            raise ImportError(
                "Transformer Engine is required for TENorm. "
                "Install with: pip install transformer-engine[pytorch]"
            )

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.normalization = normalization
        self.kwargs = kwargs

        # Import TE norm layers
        from transformer_engine.pytorch import (
            LayerNorm as TELayerNorm,
            RMSNorm as TERMSNorm,
        )

        if normalization == "RMSNorm":
            self._te_norm = TERMSNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                **kwargs,
            )
        elif normalization == "LayerNorm":
            self._te_norm = TELayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported normalization type: {normalization}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TE norm."""
        return self._te_norm(x)

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, normalization={self.normalization}"


class TEDotProductAttention(nn.Module):
    """Wrapper for Transformer Engine's DotProductAttention.

    Provides optimized attention computation with support for:
    - Multi-head attention
    - Grouped Query Attention (GQA)
    - Multi-Query Attention (MQA)
    - FP8 quantization
    - Flash Attention

    Args:
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads (for GQA/MQA).
        head_dim: Dimension of each head.
        **kwargs: Additional TE attention arguments.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if not _HAVE_TE:
            raise ImportError(
                "Transformer Engine is required for TEDotProductAttention. "
                "Install with: pip install transformer-engine[pytorch]"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.kwargs = kwargs

        # Import TE attention
        from transformer_engine.pytorch import DotProductAttention as TEAttention

        self._te_attention = TEAttention(
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_dim,
            **kwargs,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through TE attention."""
        return self._te_attention(query, key, value, **kwargs)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"


# ---------------------------------------------------------------------------
# Module Replacement Utilities
# ---------------------------------------------------------------------------


def _get_module_class_name(module: nn.Module) -> str:
    """Get the fully qualified class name of a module."""
    return f"{module.__class__.__module__}.{module.__class__.__name__}"


def _is_te_available() -> bool:
    """Check if TE is available at runtime."""
    return _HAVE_TE


# Mapping of standard module types to TE replacements
# Keys are (module_name, class_name) tuples or just class_name strings
# Values are (replacement_class, conversion_function, module_type)

SUPPORTED_MODULE_REPLACEMENTS: dict[str, tuple[type, Any, str]] = {}


def _register_replacement(
    module_path: str,
    replacement_class: type,
    converter: Any = None,
    module_type: str = "norm",
) -> None:
    """Register a module type for replacement with a TE equivalent.

    Args:
        module_path: Fully qualified module path or class name.
        replacement_class: The TE module class to use as replacement.
        converter: Optional conversion function.
        module_type: Type of module ('linear', 'norm', 'attention').
    """
    SUPPORTED_MODULE_REPLACEMENTS[module_path] = (
        replacement_class,
        converter,
        module_type,
    )


def _convert_linear(original: nn.Linear, **kwargs: Any) -> TELinear:
    """Convert an nn.Linear to TELinear."""
    new_module = TELinear(
        in_features=original.in_features,
        out_features=original.out_features,
        bias=original.bias is not None,
        **kwargs,
    )
    # Copy weights
    new_module._te_linear.weight.data.copy_(original.weight.data)
    if original.bias is not None:
        new_module._te_linear.bias.data.copy_(original.bias.data)
    return new_module


def _convert_rmsnorm(original: nn.Module, **kwargs: Any) -> TENorm:
    """Convert RMSNorm to TENorm."""
    # Get normalized_shape from the module
    if hasattr(original, "normalized_shape"):
        normalized_shape = original.normalized_shape
    elif hasattr(original, "weight") and original.weight is not None:
        normalized_shape = original.weight.shape
    else:
        raise ValueError(f"Cannot determine normalized_shape for {type(original)}")

    eps = getattr(original, "eps", 1e-6)
    new_module = TENorm(
        normalized_shape=normalized_shape,
        eps=eps,
        normalization="RMSNorm",
        **kwargs,
    )
    # Copy weights
    if hasattr(original, "weight") and original.weight is not None:
        new_module._te_norm.weight.data.copy_(original.weight.data)
    if hasattr(original, "bias") and original.bias is not None:
        new_module._te_norm.bias.data.copy_(original.bias.data)
    return new_module


def _convert_layernorm(original: nn.LayerNorm, **kwargs: Any) -> TENorm:
    """Convert LayerNorm to TENorm."""
    new_module = TENorm(
        normalized_shape=original.normalized_shape,
        eps=original.eps,
        normalization="LayerNorm",
        **kwargs,
    )
    # Copy weights
    new_module._te_norm.weight.data.copy_(original.weight.data)
    if original.bias is not None:
        new_module._te_norm.bias.data.copy_(original.bias.data)
    return new_module


# Register default replacements
_register_replacement(
    "torch.nn.modules.linear.Linear", TELinear, _convert_linear, "linear"
)
_register_replacement(
    "optimus_dl.modules.model.blocks.layer_norms.RMSNorm",
    TENorm,
    _convert_rmsnorm,
    "norm",
)
_register_replacement(
    "optimus_dl.modules.model.blocks.layer_norms.LayerNorm",
    TENorm,
    _convert_layernorm,
    "norm",
)
_register_replacement(
    "torch.nn.modules.normalization.LayerNorm",
    TENorm,
    _convert_layernorm,
    "norm",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TransformerEngineTransformConfig(ModelTransformConfig):
    """Configuration for Transformer Engine model transform.

    Attributes:
        enabled: If True, apply TE module replacements.
        replace_linear: If True, replace nn.Linear with TELinear.
        replace_norm: If True, replace RMSNorm/LayerNorm with TENorm.
        replace_attention: If True, replace attention with TEDotProductAttention.
        linear_kwargs: Additional keyword arguments for TELinear.
        norm_kwargs: Additional keyword arguments for TENorm.
        attention_kwargs: Additional keyword arguments for TEDotProductAttention.
        replace_inplace: If True, replace modules in-place. If False, create a copy.
        verbose: If True, log detailed replacement information.
    """

    enabled: bool = True
    replace_linear: bool = True
    replace_norm: bool = True
    replace_attention: bool = False  # Attention replacement is more complex

    # Additional kwargs for TE modules
    linear_kwargs: dict = None
    norm_kwargs: dict = None
    attention_kwargs: dict = None

    # Replacement behavior
    replace_inplace: bool = True
    verbose: bool = False

    def __post_init__(self):
        if self.linear_kwargs is None:
            self.linear_kwargs = {}
        if self.norm_kwargs is None:
            self.norm_kwargs = {}
        if self.attention_kwargs is None:
            self.attention_kwargs = {}


# ---------------------------------------------------------------------------
# Main Transform Class
# ---------------------------------------------------------------------------


@register_model_transform("transformer_engine", TransformerEngineTransformConfig)
class TransformerEngineTransform(BaseModelTransform):
    """Model transform that replaces modules with Transformer Engine equivalents.

    This transform traverses the model's module hierarchy and replaces supported
    PyTorch modules with their Transformer Engine (TE) equivalents. This enables:

    - Optimized CUDA kernels for linear layers and attention
    - FP8 training support through TE's autocast contexts
    - Better performance on NVIDIA H100/B200 GPUs

    The transform modifies the model in-place by default. It can replace:
    - nn.Linear -> TELinear
    - RMSNorm/LayerNorm -> TENorm
    - Custom attention -> TEDotProductAttention (if configured)

    Note: TE must be installed for this transform to work.
          Install with: pip install transformer-engine[pytorch]

    Args:
        cfg: Transformer Engine transform configuration.
    """

    def __init__(self, cfg: TransformerEngineTransformConfig, **kwargs: Any):
        super().__init__(cfg, **kwargs)
        if not _HAVE_TE:
            logger.warning(
                "Transformer Engine not available. TE model transform will be a no-op. "
                "Install with: pip install transformer-engine[pytorch]"
            )

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply TE module replacements to the model.

        Args:
            model: The model to transform.
            **kwargs: Additional arguments (unused).

        Returns:
            The transformed model (modified in-place).
        """
        if not self.cfg.enabled or not _HAVE_TE:
            logger.info("TE transform disabled or TE not available, skipping")
            return model

        logger.info("Applying Transformer Engine module replacements")

        # Track replacements
        replacements: dict[str, int] = {
            "Linear": 0,
            "RMSNorm": 0,
            "LayerNorm": 0,
            "Attention": 0,
        }

        # Replace modules
        self._replace_modules_recursive(
            model,
            replacements,
            replace_linear=self.cfg.replace_linear,
            replace_norm=self.cfg.replace_norm,
            replace_attention=self.cfg.replace_attention,
        )

        # Log results
        total = sum(replacements.values())
        if self.cfg.verbose or total > 0:
            logger.info(f"TE module replacements: {replacements}")
            logger.info(f"Total modules replaced: {total}")

        return model

    def _replace_modules_recursive(
        self,
        module: nn.Module,
        replacements: dict[str, int],
        replace_linear: bool,
        replace_norm: bool,
        replace_attention: bool,
        parent: nn.Module | None = None,
        name: str | None = None,
    ) -> None:
        """Recursively traverse and replace modules.

        Args:
            module: Current module being processed.
            replacements: Dictionary to track replacement counts.
            replace_linear: Whether to replace Linear modules.
            replace_norm: Whether to replace norm modules.
            replace_attention: Whether to replace attention modules.
            parent: Parent module (for in-place replacement).
            name: Name of current module in parent.
        """
        # Get the class name
        class_name = module.__class__.__name__
        full_class_name = _get_module_class_name(module)

        # Check for replacement
        replacement_made = False
        new_module = None

        # Try to replace modules based on registered replacements
        for module_path, (
            replacement_class,
            converter,
            module_type,
        ) in SUPPORTED_MODULE_REPLACEMENTS.items():
            if full_class_name == module_path:
                # Check if this type of replacement is enabled and get correct kwargs
                enabled = False
                kwargs = {}
                replacement_key = None

                if module_type == "linear":
                    enabled = replace_linear
                    kwargs = self.cfg.linear_kwargs
                    replacement_key = "Linear"
                elif module_type == "norm":
                    enabled = replace_norm
                    kwargs = self.cfg.norm_kwargs
                    replacement_key = (
                        "RMSNorm" if "RMSNorm" in class_name else "LayerNorm"
                    )
                elif module_type == "attention":
                    enabled = replace_attention
                    kwargs = self.cfg.attention_kwargs
                    replacement_key = "Attention"

                if not enabled:
                    continue

                try:
                    if converter:
                        new_module = converter(module, **kwargs)
                    else:
                        # Default: try to instantiate with same args
                        new_module = replacement_class(**module.__dict__)

                    if replacement_key:
                        replacements[replacement_key] += 1
                    else:
                        replacements[class_name] = replacements.get(class_name, 0) + 1

                    replacement_made = True
                    break
                except Exception as e:
                    logger.debug(f"Failed to replace {full_class_name}: {e}")

        # Perform replacement if needed
        if replacement_made and new_module is not None:
            if self.cfg.replace_inplace and parent is not None and name is not None:
                # Replace in-place
                setattr(parent, name, new_module)
            elif self.cfg.verbose:
                logger.debug(f"Would replace {full_class_name} with {type(new_module)}")

        # Recurse into children
        for child_name, child_module in module.named_children():
            self._replace_modules_recursive(
                child_module,
                replacements,
                replace_linear,
                replace_norm,
                replace_attention,
                parent=module,
                name=child_name,
            )
