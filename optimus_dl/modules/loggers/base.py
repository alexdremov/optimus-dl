"""Base class for metrics loggers in the Optimus-DL framework.

This module provides the abstract interface that all metrics logging
backends must implement to integrate with the metrics system.
"""

import logging
from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

logger = logging.getLogger(__name__)


class BaseMetricsLogger(ABC):
    """Abstract base class for metrics logging backends.

    All metrics loggers in the framework should inherit from this class.
    The logger receives computed metrics from various training phases (train,
    eval, etc.) and is responsible for persisting them (e.g., to a file,
    a database, or a cloud service).

    Attributes:
        cfg: Configuration object for the logger.
        enabled: Whether the logger is active.
    """

    def __init__(self, cfg, state_dict=None, **kwargs):
        """Initialize the metrics logger.

        Args:
            cfg: Logger configuration (subclass of MetricsLoggerConfig).
            state_dict: Optional state for resuming.
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg
        self.enabled = cfg.enabled if hasattr(cfg, "enabled") else True

        if not self.enabled:
            logger.info(f"{self.__class__.__name__} disabled via configuration")

    @abstractmethod
    def setup(self, experiment_name: str, config: dict[str, Any]) -> None:
        """Setup the logger with experiment metadata and config.

        This is typically called once at the start of a training run.

        Args:
            experiment_name: A unique name for the experiment.
            config: The full training configuration (as a dictionary).
        """
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, Any], step: int, group: str = "train"
    ) -> None:
        """Record a set of metrics for a specific training step.

        Args:
            metrics: Dictionary mapping metric names to values.
            step: The current training iteration or step.
            group: The metrics group (e.g., 'train', 'eval').
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Perform any necessary cleanup and flush remaining logs.

        Called at the end of the training or evaluation process.
        """
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return the logger's internal state for checkpointing.

        Returns:
            A dictionary containing any state needed to resume the logger
            (e.g., a run ID for WandB).
        """
        return {}
