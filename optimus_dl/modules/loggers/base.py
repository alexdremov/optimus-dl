"""
Base class for metrics loggers in the LLM baselines framework.

This module provides the abstract interface that all metrics logging
backends must implement to integrate with the existing metrics system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseMetricsLogger(ABC):
    """Base class for metrics logging backends.

    All metrics loggers should inherit from this class and implement
    the required methods for logging metrics from different groups.

    The logger integrates with the existing metrics system and receives
    computed metrics from different training phases (train, eval, etc.).
    """

    def __init__(self, cfg, state_dict=None, **kwargs):
        """Initialize the metrics logger with configuration.

        Args:
            cfg: Logger configuration (MetricsLoggerConfig or subclass)
            **kwargs: Additional keyword arguments
        """
        self.cfg = cfg
        self.enabled = cfg.enabled if hasattr(cfg, "enabled") else True

        if not self.enabled:
            logger.info(f"{self.__class__.__name__} disabled via configuration")

    @abstractmethod
    def setup(self, experiment_name: str, config: dict[str, Any]) -> None:
        """Setup the logger with experiment configuration.

        Called once at the beginning of training with full experiment context.

        Args:
            experiment_name: Name of the current experiment
            config: Full training configuration dictionary
        """
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, Any], step: int, group: str = "train"
    ) -> None:
        """Log metrics for a specific training step.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step/iteration number
            group: Metrics group (e.g., 'train', 'eval/validation')
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up and close the logger.

        Called at the end of training to properly shut down the logger.
        """
        pass

    def state_dict(self):
        return {}
