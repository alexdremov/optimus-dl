"""
Weights & Biases (wandb) metrics logger implementation.

This logger integrates with Weights & Biases for experiment tracking,
supporting both online and offline modes.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from optimus_dl.modules.loggers import register_metrics_logger
from optimus_dl.modules.loggers.base import BaseMetricsLogger
from optimus_dl.modules.loggers.config import MetricsLoggerConfig

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available - install with 'pip install wandb'")


@dataclass
class WandbLoggerConfig(MetricsLoggerConfig):
    """Configuration for Weights & Biases logger."""

    # WandB specific settings
    project: Optional[str] = None
    entity: Optional[str] = None
    mode: str = "online"  # "online", "offline", or "disabled"
    save_code: bool = True

    # Run configuration
    group: Optional[str] = None
    job_type: Optional[str] = "train"
    name: Optional[str] = None

    # Logging settings
    log_model: bool = False
    log_gradients: bool = False


@register_metrics_logger("wandb", WandbLoggerConfig)
class WandbLogger(BaseMetricsLogger):
    """Weights & Biases metrics logger.

    Logs training metrics, configuration, and optionally model artifacts
    to Weights & Biases for experiment tracking and visualization.
    """

    def __init__(self, cfg: WandbLoggerConfig, state_dict=None, **kwargs):
        """Initialize WandB logger.

        Args:
            cfg: WandB logger configuration
            **kwargs: Additional keyword arguments
        """
        super().__init__(cfg, **kwargs)

        if not WANDB_AVAILABLE:
            self.enabled = False
            logger.error("WandB logger disabled - wandb package not available")
            return

        if cfg.mode == "disabled":
            self.enabled = False
            logger.info("WandB logger disabled via mode setting")
            return

        self.run_id = (state_dict or {}).get("run_id")
        self.run = None

    def setup(self, experiment_name: str, config: Dict[str, Any]) -> None:
        """Setup WandB run with experiment configuration."""
        if not self.enabled:
            return

        try:
            # Initialize wandb run
            if OmegaConf.is_config(config):
                config = OmegaConf.to_container(config, resolve=True)
            self.run = wandb.init(
                project=self.cfg.project,
                entity=self.cfg.entity,
                mode=self.cfg.mode,
                name=self.cfg.name or experiment_name,
                group=self.cfg.group,
                job_type=self.cfg.job_type,
                tags=list(self.cfg.tags.keys()) if self.cfg.tags else None,
                notes=self.cfg.notes,
                save_code=self.cfg.save_code,
                config=config,
                id=self.run_id,  # Resume from preemption etc
                resume="allow",  # Allow resuming if run exists
            )

            logger.info(f"WandB run initialized: {self.run.name} ({self.run.id})")

        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}", exc_info=True)
            self.enabled = False

    def log_metrics(
        self, metrics: dict[str, Any], step: int, group: str = "train"
    ) -> None:
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step/iteration number
            group: Metrics group (e.g., 'train', 'eval/validation')
        """
        if not self.enabled:
            return

        if self.run is None:
            logger.warning("WandB run not initialized, skipping metrics logging")
            return

        try:
            # Flatten nested metrics and add group prefix
            flattened_metrics = {}

            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Handle nested metrics
                    for nested_key, nested_value in value.items():
                        full_key = f"{group}/{key}/{nested_key}"
                        flattened_metrics[full_key] = nested_value
                else:
                    # Simple metric
                    full_key = f"{group}/{key}"
                    flattened_metrics[full_key] = value

            # Log to wandb
            self.run.log(flattened_metrics, step=step)

        except Exception as e:
            logger.error(f"Failed to log metrics to WandB: {e}")

    def close(self) -> None:
        """Close WandB run."""
        if self.run is not None:
            try:
                self.run.finish()
                logger.info("WandB run finished successfully")
            except Exception as e:
                logger.error(f"Error finishing WandB run: {e}")
            finally:
                self.run = None

    def state_dict(self):
        return {
            "run_id": self.run.id if self.run is not None else None,
        }
