"""Configuration dataclasses for AMP and FP8.

This module re-exports the AMP and FP8 configuration classes from
optimus_dl.modules.optim.config for convenience.
"""

from optimus_dl.modules.optim.config import (
    AmpConfig,
    Fp8Config,
    OptimizationConfig,
)

__all__ = [
    "AmpConfig",
    "Fp8Config",
    "OptimizationConfig",
]
