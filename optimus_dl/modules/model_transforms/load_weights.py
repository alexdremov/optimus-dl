import logging
from dataclasses import dataclass, field

from omegaconf import MISSING

from optimus_dl.modules.checkpoint.checkpoint_manager import CheckpointManager
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import register_model_transform
from optimus_dl.modules.model_transforms.base import BaseModelTransform
from optimus_dl.modules.model_transforms.config import ModelTransformConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadWeightsTransformConfig(ModelTransformConfig):
    """
    Configuration for load weights model transform.
    """

    checkpoint_path: str = field(
        default=MISSING,
        metadata={
            "description": "Path to checkpoint to load weights from.",
        },
    )

    skip_on_restart: bool = field(
        default=True,
        metadata={
            "description": "Skip loading weights if this run is a restart.",
        },
    )


@register_model_transform("load_weights", LoadWeightsTransformConfig)
class LoadWeightsTransform(BaseModelTransform):
    def apply(
        self,
        model: BaseModel,
        is_restart: bool,
        checkpoint_manager: CheckpointManager,
        **kwargs,
    ) -> BaseModel:
        """Load weights from a specified checkpoint path into the model.

        Args:
            model: The model to load weights into.
            is_restart: A boolean indicating if the current run is a restart.
            checkpoint_manager: The checkpoint manager instance.
            **kwargs: Additional keyword arguments.

        Returns:
            The model with loaded weights.
        """
        if self.cfg.skip_on_restart and is_restart:
            logger.info(
                "Skipping 'load_weights' transform because it's a restart and "
                "'skip_on_restart' is True."
            )
            return model

        if self.cfg.checkpoint_path is None:
            logger.warning(
                "No 'checkpoint_path' specified for 'load_weights' transform. Skipping."
            )
            return model

        logger.info(f"Loading weights from: {self.cfg.checkpoint_path}")

        checkpoint_manager.load_model_state_dict(model, self.cfg.checkpoint_path)

        logger.info("Successfully loaded weights.")
        return model
