import logging
from abc import ABC, abstractmethod
from typing import Any

from optimus_dl.modules.model.base import BaseModel

logger = logging.getLogger(__name__)


class BaseModelTransform(ABC):
    """Abstract base class for all model transformations.

    Model transforms are applied after the model is built but before training
    begins. They modify the model's structure, wrapping it with distributed
    wrappers (DDP, FSDP), applying graph compilation (torch.compile), or
    injecting activation checkpointing.

    Transforms are registered in the `model_transform` registry and can be
    chained together in the configuration.
    """

    def __init__(self, cfg: Any = None, **kwargs):
        """Initialize the transform.

        Args:
            cfg: Configuration object for the transform.
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg

    @abstractmethod
    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply the transformation to the given model.

        Args:
            model: The model to transform.
            **kwargs: Additional arguments (e.g., collective, device).

        Returns:
            The transformed model (either modified in-place or a new wrapper).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cfg={self.cfg})"
