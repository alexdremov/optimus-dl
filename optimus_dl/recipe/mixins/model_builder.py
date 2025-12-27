"""Model builder mixin for building and transforming models with checkpoint loading."""

import logging
from dataclasses import dataclass
from typing import Any

from optimus_dl.core.model_utils import get_num_parameters
from optimus_dl.core.registry import RegistryConfig, make_registry
from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.model import BaseModel, ModelConfig, build_model
from optimus_dl.modules.model_transforms import (
    ModelTransformConfig,
    build_model_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBuilderConfig(RegistryConfig):
    """Configuration for ModelBuilder."""

    pass


class ModelBuilder:
    """Mixin for building models and applying transformations.

    Encapsulates the logic for:
    1.  Instantiating a `BaseModel` from a configuration object.
    2.  Sequentially applying a list of `ModelTransforms` (e.g., FSDP, DDP, compile).
    3.  Logging model statistics (parameter count).

    Args:
        cfg: Builder configuration.
        model_transforms: List of configurations for transforms to apply.
    """

    def __init__(
        self,
        cfg: ModelBuilderConfig,
        model_transforms: list[ModelTransformConfig] | None = None,
        **kwargs: Any,
    ):
        self.model_transforms = model_transforms or []

    def build_model(
        self, model_config: ModelConfig | None, collective: Collective, **kwargs
    ) -> BaseModel:
        """Build the model and apply all configured transforms.

        Args:
            model_config: Configuration for the model architecture.
            collective: Distributed collective for transforms that need it.
            **kwargs: Additional arguments passed to model constructor and transforms.

        Returns:
            The fully constructed and transformed model.
        """
        if model_config is None:
            raise ValueError(
                "model_config is None. Use build_model_from_checkpoint for evaluation."
            )

        model = build_model(model_config, **kwargs)
        logger.info(
            f"Params num (before model transforms): {get_num_parameters(model):,}"
        )
        assert isinstance(model, BaseModel)

        # Apply model transforms (including distributed setup)
        model = self._apply_model_transforms(
            model, collective=collective, device=collective.default_device, **kwargs
        )
        logger.info(f"Model \n{model}")
        logger.info(
            f"Params num (after model transforms): {get_num_parameters(model):,}"
        )

        return model

    def _apply_model_transforms(self, model: BaseModel, **kwargs) -> BaseModel:
        """Iteratively apply all configured model transforms.

        Args:
            model: The base model.
            **kwargs: Context arguments (device, collective, etc.).

        Returns:
            The transformed model.
        """
        for transform_cfg in self.model_transforms:
            try:
                transform = build_model_transform(transform_cfg, **kwargs)
                if transform is not None:
                    logger.info(f"Applying model transform: {transform}")
                    model = transform.apply(model, **kwargs)
                else:
                    logger.warning(
                        f"Failed to build model transform from config: {transform_cfg}"
                    )
            except Exception as e:
                logger.error(f"Failed to apply model transform {transform_cfg}: {e}")
                raise

        return model


_, register_model_builder, build_model_builder = make_registry(
    "model_builder", ModelBuilder
)
register_model_builder("base", ModelBuilderConfig)(ModelBuilder)
