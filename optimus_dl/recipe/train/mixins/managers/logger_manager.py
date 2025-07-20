"""Logger mixin for handling metrics logging."""

import logging
from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfig, build, make_registry
from optimus_dl.modules.loggers import BaseMetricsLogger, MetricsLoggerConfig

logger = logging.getLogger(__name__)


@dataclass
class LoggerManagerConfig(RegistryConfig):
    pass


class LoggerManager:
    """Mixin for handling metrics logging."""

    def __init__(
        self,
        cfg: LoggerManagerConfig,
        loggers_config: list[MetricsLoggerConfig] | None,
        **kwargs,
    ):
        self.loggers_config = loggers_config
        self.previous_state = {}
        self.loggers: list[BaseMetricsLogger] | None = None

    def build_loggers(self, **kwargs):
        """Build metrics loggers from configuration.

        Returns:
            List of configured logger instances
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
        """Setup all loggers with experiment configuration.

        Args:
            loggers: List of logger instances
            experiment_name: Name of the current experiment
            full_config: Full configuration dict for logger setup
        """
        for logger_instance in self.loggers or []:
            try:
                logger_instance.setup(experiment_name, full_config)
            except Exception as e:
                logger.error(
                    f"Failed to setup logger {logger_instance.__class__.__name__}: {e}"
                )

    def log_metrics_to_loggers(self, metrics, step: int, group: str = "train"):
        """Log metrics to all provided loggers.

        Args:
            loggers: List of logger instances
            metrics: Dictionary of metric names to values
            step: Training step/iteration number
            group: Metrics group (e.g., 'train', 'eval/validation')
        """
        for logger_instance in self.loggers or []:
            try:
                logger_instance.log_metrics(metrics, step, group)
            except Exception as e:
                logger.error(
                    f"Failed to log metrics with {logger_instance.__class__.__name__}: {e}"
                )

    def close_loggers(self):
        """Close all loggers.

        Args:
            loggers: List of logger instances to close
        """
        for logger_instance in self.loggers or []:
            try:
                logger_instance.close()
            except Exception as e:
                logger.error(
                    f"Failed to close logger {logger_instance.__class__.__name__}: {e}"
                )

    def state_dict(self):
        return {
            logger_instance.cfg.id: logger_instance.state_dict()
            for logger_instance in self.loggers or []
        }

    def load_state_dict(self, state_dict):
        self.previous_state = state_dict


_, register_logger_manager, build_logger_manager = make_registry(
    "logger_manager", LoggerManager
)
register_logger_manager("base", LoggerManagerConfig)(LoggerManager)
