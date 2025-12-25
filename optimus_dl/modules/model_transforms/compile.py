import logging
from dataclasses import dataclass, field

from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import register_model_transform
from optimus_dl.modules.model_transforms.base import BaseModelTransform
from optimus_dl.modules.model_transforms.config import ModelTransformConfig

logger = logging.getLogger(__name__)


@dataclass
class CompileTransformConfig(ModelTransformConfig):
    """Configuration for torch.compile model transform."""

    compile_kwargs: dict = field(
        default_factory=dict,
        metadata={
            "help": "Arguments for torch.compile. See https://pytorch.org/docs/stable/generated/torch.compile.html"
        },
    )
    activation_memory_budget: float | None = field(
        default=None,
        metadata={
            "help": "Activation memory budget for torch.compile. See https://pytorch.org/blog/activation-checkpointing-techniques/"
        },
    )


@register_model_transform("compile", CompileTransformConfig)
class CompileTransform(BaseModelTransform):
    """Model transform that applies torch.compile to the model's forward method."""

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply torch.compile to the model's forward method.

        Args:
            model: The model to compile
            **kwargs: Additional arguments (unused)

        Returns:
            The model with compiled forward method
        """
        import torch._functorch.config

        compile_kwargs = self.cfg.compile_kwargs if self.cfg else {}
        torch._functorch.config.activation_memory_budget = (
            self.cfg.activation_memory_budget
        )

        logger.info(f"Applying torch.compile with args: {compile_kwargs}")
        model = torch.compile(model, **compile_kwargs)

        return model
