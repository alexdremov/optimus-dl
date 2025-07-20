from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import BaseMetricsLogger
from .config import MetricsLoggerConfig

_LOGGERS_REGISTRY, register_metrics_logger, build_metrics_logger = make_registry(
    "metrics_logger", BaseMetricsLogger
)

bootstrap_module(__name__)
