"""Optimizer builder mixin for building optimizers."""

import logging
from dataclasses import dataclass

from torch.optim import Optimizer

from optimus_dl.core.registry import RegistryConfig, build, make_registry
from optimus_dl.modules.optim import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizerBuilderConfig(RegistryConfig):
    pass


class OptimizerBuilder:
    """Mixin for building optimizers."""

    def __init__(
        self,
        cfg: OptimizerBuilderConfig,
        optimization_config: OptimizationConfig,
        **kwargs,
    ):
        self.optimization_config = optimization_config

    def build_optimizer(self, params, **kwargs) -> Optimizer:
        """Build and validate the optimizer."""
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
