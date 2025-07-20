import logging
from abc import ABC, abstractmethod
from typing import Any

from optimus_dl.modules.model.base import BaseModel

logger = logging.getLogger(__name__)


class BaseModelTransform(ABC):
    """Base class for model transformations.

    Model transforms are applied after the model is built to modify or enhance it.
    Examples include torch.compile, quantization, pruning, etc.
    """

    def __init__(self, cfg: Any = None, **kwargs):
        """Initialize the model transform.

        Args:
            cfg: Configuration for the transform
            **kwargs: Additional keyword arguments
        """
        self.cfg = cfg

    @abstractmethod
    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply the transformation to the model.

        Args:
            model: The model to transform
            **kwargs: Additional arguments that may be needed for transformation

        Returns:
            The transformed model (may be the same instance or a new wrapper)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cfg={self.cfg})"
