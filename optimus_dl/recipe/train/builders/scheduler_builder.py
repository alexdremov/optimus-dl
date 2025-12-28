"""Scheduler builder mixin for building learning rate schedulers."""

import logging
from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer

from optimus_dl.core.registry import (
    RegistryConfig,
    build,
    make_registry,
)
from optimus_dl.modules.optim import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class SchedulerBuilderConfig(RegistryConfig):
    """Configuration for SchedulerBuilder."""

    pass


class SchedulerBuilder:
    """Builder class responsible for creating the learning rate scheduler.

    Instantiates a scheduler (e.g., CosineAnnealing, WSD) and associates it
    with the optimizer. It ensures the scheduler is aware of the total training
    iterations.

    Args:
        cfg: Builder configuration.
        lr_scheduler_config: Configuration for the scheduler itself (can be None).
        optimization_config: Optimization settings (needed for total iterations).
    """

    def __init__(
        self,
        cfg: SchedulerBuilderConfig,
        lr_scheduler_config: RegistryConfig | None,
        optimization_config: OptimizationConfig,
        **kwargs: Any,
    ):
        self.lr_scheduler_config = lr_scheduler_config
        self.optimization_config = optimization_config

    def build_lr_scheduler(self, optimizer: Optimizer, **kwargs):
        """Build and validate the learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule.
            **kwargs: Additional arguments.

        Returns:
            Instantiated LR Scheduler or None if not configured.
        """
        if self.lr_scheduler_config is None:
            return None
        lr_scheduler = build(
            "lr_scheduler",
            cfg=self.lr_scheduler_config,
            optimizer=optimizer,
            iterations=self.optimization_config.iterations,
            **kwargs,
        )
        if lr_scheduler is not None:
            logger.info(f"LR Scheduler \n{lr_scheduler}")
        return lr_scheduler


_, register_scheduler_builder, build_scheduler_builder = make_registry(
    "scheduler_builder", SchedulerBuilder
)
register_scheduler_builder("base", SchedulerBuilderConfig)(SchedulerBuilder)
