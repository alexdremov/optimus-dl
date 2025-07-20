"""Training context mixin for AMP and gradient scaler setup."""

import logging
from typing import Any

import torch

from optimus_dl.recipe.train.config import OptimizationConfig

logger = logging.getLogger(__name__)


class TrainingContextMixin:
    """Mixin for setting up training context (AMP, scaler, etc.)."""

    def __init__(self, optimization_config: OptimizationConfig):
        self.optimization_config = optimization_config

    def setup_training_context(self, device: torch.device) -> dict[str, Any]:
        """Setup training context (AMP, scaler, etc.).

        Args:
            device: Target device

        Returns:
            Dictionary containing training context objects
        """
        amp_cfg = self.optimization_config.amp
        scaler = torch.GradScaler(
            device=device.type,
            enabled=amp_cfg.enabled and amp_cfg.enable_scaler,
            init_scale=amp_cfg.init_scale,
            growth_factor=amp_cfg.growth_factor,
            backoff_factor=amp_cfg.backoff_factor,
            growth_interval=amp_cfg.growth_interval,
        )
        logger.info(f"Using grad scaler: {scaler.is_enabled()}")
        # Safe dtype conversion without eval()
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        dtype = dtype_map.get(amp_cfg.dtype, torch.float16)
        if amp_cfg.dtype not in dtype_map:
            logger.warning(f"Unknown dtype '{amp_cfg.dtype}', defaulting to float16")

        amp_ctx = torch.autocast(device.type, dtype=dtype, enabled=amp_cfg.enabled)

        return {
            "scaler": scaler,
            "amp_ctx": amp_ctx,
            "amp_cfg": amp_cfg,
            "device": device,
        }
