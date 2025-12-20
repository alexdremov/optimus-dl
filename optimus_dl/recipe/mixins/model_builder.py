"""Model builder mixin for building and transforming models with checkpoint loading."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from optimus_dl.core.model_utils import get_num_parameters
from optimus_dl.core.registry import RegistryConfig, build, make_registry
from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.model import ModelConfig
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import (
    ModelTransformConfig,
    build_model_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBuilderConfig(RegistryConfig):
    pass


class ModelBuilder:
    """Mixin for building models and applying transforms, with checkpoint loading support."""

    def __init__(
        self,
        cfg: ModelBuilderConfig,
        model_transforms: list[ModelTransformConfig] | None = None,
        **kwargs,
    ):
        self.model_transforms = model_transforms or []

    def build_model(
        self, model_config: ModelConfig | None, collective: Collective, **kwargs
    ) -> BaseModel:
        """Build and validate the model."""
        if model_config is None:
            raise ValueError(
                "model_config is None. Use build_model_from_checkpoint for evaluation."
            )

        model = build("model", model_config, **kwargs)
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
        """Apply configured model transforms to the model.

        Args:
            model: The model to transform
            **kwargs: Additional arguments to pass to transforms

        Returns:
            The transformed model
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
