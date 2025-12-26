"""Criterion builder mixin for building loss criteria."""

import logging
from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfig, build, make_registry
from optimus_dl.modules.criterion import BaseCriterion, CriterionConfig

logger = logging.getLogger(__name__)


@dataclass
class CriterionBuilderConfig(RegistryConfig):
    """Configuration for CriterionBuilder (usually just a registry name)."""

    pass


class CriterionBuilder:
    """Builder class responsible for creating the loss criterion instance.

    Uses the `criterion` registry to instantiate the specified loss function
    (e.g., CrossEntropy) based on the training configuration.

    Args:
        cfg: Builder configuration.
        criterion_config: Configuration for the criterion itself.
    """

    def __init__(
        self, cfg: CriterionBuilderConfig, criterion_config: CriterionConfig, **kwargs
    ):
        self.criterion_config = criterion_config

    def build_criterion(self, **kwargs) -> BaseCriterion:
        """Instantiate and return the configured loss criterion."""
        criterion = build("criterion", self.criterion_config, **kwargs)
        assert isinstance(criterion, BaseCriterion)
        logger.info(f"Criterion \n{criterion}")
        return criterion


_, register_criterion_builder, build_criterion_builder = make_registry(
    "criterion_builder", CriterionBuilder
)
register_criterion_builder("base", CriterionBuilderConfig)(CriterionBuilder)
