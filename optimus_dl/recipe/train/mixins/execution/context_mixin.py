"""Training context mixin for AMP and gradient scaler setup.

This module provides FP8 (Floating Point 8-bit) and standard AMP support for training.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import (
    Any,
)

import torch

from optimus_dl.core.dtype import (
    get_fp8_format_from_dtype,
    is_fp8_dtype,
    str_to_dtype,
)
from optimus_dl.modules.amp import (
    FP8Recipe,
    create_fp8_recipe,
    is_transformer_engine_available,
)
from optimus_dl.recipe.train.config import OptimizationConfig

logger = logging.getLogger(__name__)


class TrainingContextMixin:
    """Mixin for setting up the training context (precision, scaling, devices).

    Responsible for initializing PyTorch's AMP (Automatic Mixed Precision),
    GradScaler, and FP8 (Floating Point 8-bit) support based on the optimization
    configuration. This ensures consistent precision settings across the training loop.

    FP8 Support:
        - Uses Transformer Engine for FP8 training when available
        - Supports hybrid (E4M3 forward, E5M2 backward), E4M3, and E5M2 formats
        - Falls back to standard AMP when FP8 is not available

    Args:
        optimization_config: Configuration containing AMP and FP8 settings.
    """

    def __init__(self, optimization_config: OptimizationConfig):
        self.optimization_config = optimization_config
        self._fp8_recipe: FP8Recipe | None = None

    def setup_training_context(self, device: torch.device) -> dict[str, Any]:
        """Initialize AMP context, Gradient Scaler, and FP8 recipe.

        Args:
            device: The target compute device.

        Returns:
            A dictionary containing:

            - "scaler": The torch.cuda.amp.GradScaler instance.
            - "amp_ctx": The autocast context manager (standard AMP or FP8).
            - "amp_cfg": The raw AMP configuration object.
            - "device": The device being used.
            - "fp8_recipe": The FP8Recipe instance (if FP8 is enabled and available).
            - "fp8_enabled": Whether FP8 training is active.
            - "is_fp8_dtype": Whether the target dtype is FP8.
        """
        amp_cfg = self.optimization_config.amp

        # Check if FP8 is requested
        dtype = str_to_dtype(amp_cfg.dtype)
        is_fp8 = is_fp8_dtype(dtype)
        get_fp8_format_from_dtype(dtype)

        # For FP8, disable gradient scaler (handled by Transformer Engine)
        enable_scaler = amp_cfg.enable_scaler and not is_fp8

        scaler = torch.GradScaler(
            device=device.type,
            enabled=amp_cfg.enabled and enable_scaler,
            init_scale=amp_cfg.init_scale,
            growth_factor=amp_cfg.growth_factor,
            backoff_factor=amp_cfg.backoff_factor,
            growth_interval=amp_cfg.growth_interval,
        )
        logger.info(f"Using grad scaler: {scaler.is_enabled()}")

        # Setup FP8 recipe if requested
        fp8_recipe = None
        fp8_enabled = False

        if is_fp8 and amp_cfg.enabled and amp_cfg.fp8.enabled:
            if not is_transformer_engine_available():
                logger.warning(
                    "FP8 training requested but Transformer Engine is not available. "
                    "Falling back to standard AMP. "
                    "Install with: pip install transformer-engine"
                )
            else:
                # Create FP8 recipe from config
                fp8_recipe = create_fp8_recipe(
                    format=amp_cfg.fp8.format,
                    margin=amp_cfg.fp8.margin,
                    amax_history_len=amp_cfg.fp8.amax_history_len,
                    amax_compute_algo=amp_cfg.fp8.amax_compute_algo,
                    reduce_amax=amp_cfg.fp8.reduce_amax,
                    fp8_dpa=amp_cfg.fp8.fp8_dpa,
                    fp8_mha=amp_cfg.fp8.fp8_mha,
                )
                fp8_enabled = fp8_recipe is not None
                self._fp8_recipe = fp8_recipe

        # Create the appropriate autocast context manager
        if fp8_enabled and device.type == "cuda":
            # FP8 autocast via Transformer Engine
            amp_ctx = self._create_fp8_autocast(device, fp8_recipe)
        else:
            # Standard AMP autocast
            amp_ctx = torch.autocast(
                device.type, dtype=dtype, enabled=amp_cfg.enabled and not is_fp8
            )

        return {
            "scaler": scaler,
            "amp_ctx": amp_ctx,
            "amp_cfg": amp_cfg,
            "device": device,
            "fp8_recipe": fp8_recipe,
            "fp8_enabled": fp8_enabled,
            "is_fp8_dtype": is_fp8,
            "fp8_backward_ctx": self.get_fp8_backward_ctx(),
        }

    def _create_fp8_autocast(
        self, device: torch.device, fp8_recipe: FP8Recipe | None
    ) -> Any:
        """Create FP8 autocast context manager.

        Args:
            device: The target device.
            fp8_recipe: The FP8Recipe instance.

        Returns:
            A context manager for FP8 autocast.
        """

        @contextmanager
        def fp8_autocast_ctx() -> Iterator[None]:
            if fp8_recipe is not None:
                with fp8_recipe.autocast():
                    yield
            else:
                yield

        return fp8_autocast_ctx()

    def get_fp8_backward_ctx(self) -> Any:
        """Get FP8 backward context manager.

        Returns:
            A context manager for FP8 backward pass, or nullcontext if not enabled.
        """
        if self._fp8_recipe is not None:
            return self._fp8_recipe.backward()
        from contextlib import nullcontext

        return nullcontext()
