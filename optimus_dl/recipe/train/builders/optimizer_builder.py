"""Optimizer builder mixin for building optimizers."""

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
class OptimizerBuilderConfig(RegistryConfig):
    """Configuration for OptimizerBuilder."""

    pass


class OptimizerBuilder:
    """Builder class responsible for creating the optimizer.

    Takes parameter groups from the model and instantiates the configured
    optimizer (e.g., AdamW). It also logs the total number of optimized
    parameters.

    Args:
        cfg: Builder configuration.
        optimization_config: Optimization settings including the optimizer config.
    """

    def __init__(
        self,
        cfg: OptimizerBuilderConfig,
        optimization_config: OptimizationConfig,
        **kwargs: Any,
    ):
        self.optimization_config = optimization_config

    def build_optimizer(self, params, **kwargs) -> Optimizer:
        """Build and validate the optimizer.

        Args:
            params: Iterable of parameters or dicts defining parameter groups
                (typically from `model.make_parameter_groups()`).
            **kwargs: Additional arguments passed to the optimizer constructor.

        Returns:
            Instantiated Optimizer.
        """
        optimizer = build(
            "optimizer", self.optimization_config.optimizer, params=params, **kwargs
        )
        assert isinstance(optimizer, Optimizer)
        logger.info(f"Optimizer \n{optimizer}")
        optimized_params = []
        for param_group in optimizer.param_groups:
            optimized_params.append(
                sum([p.numel() for p in param_group["params"] if p.requires_grad])
            )
        logger.info(
            f"Optimized {sum(optimized_params):,} parameters. Per group: {[f'{i:,}' for i in optimized_params]}"
        )

        return optimizer


_, register_optimizer_builder, build_optimizer_builder = make_registry(
    "optimizer_builder", OptimizerBuilder
)
register_optimizer_builder("base", OptimizerBuilderConfig)(OptimizerBuilder)
