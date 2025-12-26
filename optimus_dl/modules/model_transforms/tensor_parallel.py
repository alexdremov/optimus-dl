"""Tensor Parallelism Transform."""

import logging
from dataclasses import dataclass, field

from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.distributed.mesh import MeshCollective
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import register_model_transform
from optimus_dl.modules.model_transforms.base import BaseModelTransform
from optimus_dl.modules.model_transforms.config import ModelTransformConfig

logger = logging.getLogger(__name__)


@dataclass
class TensorParallelConfig(ModelTransformConfig):
    """Configuration for Tensor Parallelism.

    Attributes:
        custom_model_kwargs: Additional keyword arguments passed to the model's
            `apply_tp` method (e.g., sequence_parallel=True).
    """

    custom_model_kwargs: dict = field(default_factory=dict)


@register_model_transform("tensor_parallel", TensorParallelConfig)
class TensorParallelTransform(BaseModelTransform):
    """Transform that applies Tensor Parallelism to a model.

    This transform delegates the actual sharding logic to the model's `apply_tp`
    method, providing it with the appropriate Tensor Parallel device mesh from
     the global collective.

    Args:
        cfg: Tensor parallel configuration.
        collective: Distributed collective (MeshCollective required).
    """

    def __init__(
        self,
        cfg: TensorParallelConfig,
        collective: Collective,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)
        self.collective = collective

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply the tensor parallel plan to the model."""
        if not isinstance(self.collective, MeshCollective):
            logger.warning("TensorParallel requires MeshCollective. Skipping.")
            return model

        tp_mesh = self.collective.tp_mesh
        if tp_mesh is None:
            logger.info("No TP mesh found (tp_size=1). Skipping Tensor Parallelism.")
            return model

        logger.info(f"Applying Tensor Parallelism with mesh: {tp_mesh}")

        # Get the parallelization plan from the model
        model.apply_tp(tp_mesh, **self.cfg.custom_model_kwargs)

        logger.info("Tensor Parallelism applied successfully.")
        return model
