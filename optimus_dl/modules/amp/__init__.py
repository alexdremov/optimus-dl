"""AMP (Automatic Mixed Precision) and FP8 utilities for Optimus-DL.

This module provides:
- FP8 (Floating Point 8-bit) training support via Transformer Engine
- FP8 recipe configurations for delayed scaling
- FP8-aware autocast context managers
- Utility functions for FP8 dtype handling
"""

from optimus_dl.core.bootstrap import bootstrap_module

from .config import (
    AmpConfig,
    Fp8Config,
    OptimizationConfig,
)

# Use lazy imports for FP8 utilities to avoid circular import issues
# These will be imported on first access


def __getattr__(name: str):
    """Lazy import for FP8 utilities to avoid circular imports."""
    _fp8_exports = {
        "Fp8Format": "Fp8Format",
        "Fp8RecipeType": "Fp8RecipeType",
        "FP8Recipe": "FP8Recipe",
        "Fp8RecipeConfig": "Fp8RecipeConfig",
        "create_fp8_recipe": "create_fp8_recipe",
        "fp8_autocast": "fp8_autocast",
        "fp8_backward": "fp8_backward",
        "get_fp8_format_from_string": "get_fp8_format_from_string",
        "get_fp8_recipe_type_from_string": "get_fp8_recipe_type_from_string",
        "get_transformer_engine_version": "get_transformer_engine_version",
        "is_transformer_engine_available": "is_transformer_engine_available",
    }
    if name in _fp8_exports:
        from . import fp8

        return getattr(fp8, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


bootstrap_module(__name__)

__all__ = [
    # Config classes
    "AmpConfig",
    "Fp8Config",
    "OptimizationConfig",
    "Fp8RecipeConfig",
    # FP8 classes and utilities
    "Fp8Format",
    "Fp8RecipeType",
    "FP8Recipe",
    "create_fp8_recipe",
    "fp8_autocast",
    "fp8_backward",
    # Utility functions
    "get_fp8_format_from_string",
    "get_fp8_recipe_type_from_string",
    "get_transformer_engine_version",
    "is_transformer_engine_available",
]
