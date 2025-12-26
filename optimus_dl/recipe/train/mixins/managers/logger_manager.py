"""Logger mixin for handling metrics logging."""

import logging
from dataclasses import dataclass
from typing import Any

from optimus_dl.core.registry import RegistryConfig, build, make_registry
from optimus_dl.modules.loggers import BaseMetricsLogger, MetricsLoggerConfig

logger = logging.getLogger(__name__)


@dataclass
class LoggerManagerConfig(RegistryConfig):
    """Configuration for LoggerManager."""

    pass


class LoggerManager:
    """Manager for multiple metrics loggers.

    This class instantiates and orchestrates a list of logging backends (e.g.,
    JSONL, WandB). It provides a unified interface for setting up, logging to,
    and closing all configured loggers.

    Args:
        cfg: Manager configuration.
        loggers_config: List of configurations for individual loggers.
    """

    def __init__(
        self,
        cfg: LoggerManagerConfig,
        loggers_config: list[MetricsLoggerConfig] | None,
        **kwargs: Any,
    ):
        self.loggers_config = loggers_config
        self.previous_state = {}
        self.loggers: list[BaseMetricsLogger] | None = None

    def build_loggers(self, **kwargs):
        """Instantiate all configured loggers.

        Uses the registry to build logger instances. If previous state is available
        (from a checkpoint), it is passed to the logger builders for resumption.

        Returns:
            List of active logger instances.
        """
        if self.loggers_config is None:
            logger.info("No loggers configuration found, metrics logging disabled")
            return
        assert self.loggers is None, "Loggers already built"

        loggers = []
        for logger_config in self.loggers_config:
            try:
                logger_instance = build(
                    "metrics_logger",
                    logger_config,
                    state_dict=self.previous_state.get(logger_config.id),
                    **kwargs,
                )
                loggers.append(logger_instance)
                logger.info(f"Built logger: {logger_instance.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to build logger from config {logger_config}: {e}")
                raise

        self.loggers = loggers

    def setup_loggers(self, experiment_name: str, full_config: dict):
        """Initialize all loggers with experiment context.

        Args:
            experiment_name: Name of the experiment.
            full_config: Complete training configuration dictionary.
        """
        for logger_instance in self.loggers or []:
            try:
                logger_instance.setup(experiment_name, full_config)
            except Exception as e:
                logger.error(
                    f"Failed to setup logger {logger_instance.__class__.__name__}: {e}"
                )

    def log_metrics_to_loggers(self, metrics, step: int, group: str = "train"):
        """Dispatch metrics to all active loggers.

        Args:
            metrics: Dictionary of metric values.
            step: Current iteration.
            group: Metric group name.
        """
        for logger_instance in self.loggers or []:
            try:
                logger_instance.log_metrics(metrics, step, group)
            except Exception as e:
                logger.error(
                    f"Failed to log metrics with {logger_instance.__class__.__name__}: {e}"
                )

    def close_loggers(self):
        """Clean up all loggers."""
        for logger_instance in self.loggers or []:
            try:
                logger_instance.close()
            except Exception as e:
                logger.error(
                    f"Failed to close logger {logger_instance.__class__.__name__}: {e}"
                )

    def state_dict(self):
        """Collect state from all loggers for checkpointing."""
        return {
            logger_instance.cfg.id: logger_instance.state_dict()
            for logger_instance in self.loggers or []
        }

    def load_state_dict(self, state_dict):
        """Load logger states from a checkpoint."""
        self.previous_state = state_dict


_, register_logger_manager, build_logger_manager = make_registry(
    "logger_manager", LoggerManager
)
register_logger_manager("base", LoggerManagerConfig)(LoggerManager)
