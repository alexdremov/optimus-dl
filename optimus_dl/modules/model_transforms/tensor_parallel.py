"""Tensor Parallelism Transform."""

import logging
from dataclasses import dataclass

from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.distributed.mesh import MeshCollective
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import register_model_transform
from optimus_dl.modules.model_transforms.base import BaseModelTransform
from optimus_dl.modules.model_transforms.config import ModelTransformConfig

logger = logging.getLogger(__name__)


@dataclass
class TensorParallelConfig(ModelTransformConfig):
    """Configuration for Tensor Parallelism."""

    pass


@register_model_transform("tensor_parallel", TensorParallelConfig)
class TensorParallelTransform(BaseModelTransform):
    """Applies Tensor Parallelism to the model."""

    def __init__(
        self,
        cfg: TensorParallelConfig,
        collective: Collective,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)
        self.collective = collective

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        if not isinstance(self.collective, MeshCollective):
            logger.warning("TensorParallel requires MeshCollective. Skipping.")
            return model

        tp_mesh = self.collective.tp_mesh
        if tp_mesh is None:
            logger.info("No TP mesh found (tp_size=1). Skipping Tensor Parallelism.")
            return model

        logger.info(f"Applying Tensor Parallelism with mesh: {tp_mesh}")

        # Get the parallelization plan from the model
        model.apply_tp(tp_mesh)

        logger.info("Tensor Parallelism applied successfully.")
        return model
