"""Activation checkpointing (gradient checkpointing) transform using public PyTorch API."""

import logging
import functools
from dataclasses import dataclass

import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import register_model_transform
from optimus_dl.modules.model_transforms.base import BaseModelTransform
from optimus_dl.modules.model_transforms.config import ModelTransformConfig

logger = logging.getLogger(__name__)


@dataclass
class ActivationCheckpointConfig(ModelTransformConfig):
    """Configuration for activation checkpointing."""

    # List of layer class names to wrap (e.g. ["LlamaBlock", "GPTBlock"])
    layer_classes: list[str] | None = None

    # Whether to use reentrant checkpointing.
    # False is generally recommended for newer PyTorch versions and FSDP.
    use_reentrant: bool = False

    save_ops: list[str] | None = None


class CheckpointWrapper(nn.Module):
    """Wraps a module to apply activation checkpointing during forward pass."""

    def __init__(self, module: nn.Module, ops_to_save: list, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

        def policy_fn(ctx, op, *args, **kwargs):
            if op in ops_to_save:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        self.policy_fn = policy_fn

    def forward(self, *args, **kwargs):
        # torch.utils.checkpoint.checkpoint requires a function as the first argument.
        # We pass self.module (which is callable).
        # Note: 'use_reentrant' argument is available in modern PyTorch.
        return checkpoint(
            self.module,
            *args,
            use_reentrant=self.use_reentrant,
            context_fn=functools.partial(create_selective_checkpoint_contexts, self.policy_fn.policy_fn),
            **kwargs
        )


@register_model_transform("activation_checkpoint", ActivationCheckpointConfig)
class ActivationCheckpointTransform(BaseModelTransform):
    """Applies activation checkpointing to the model using torch.utils.checkpoint."""

    def __init__(
        self,
        cfg: ActivationCheckpointConfig,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        logger.info("Applying activation checkpointing (torch.utils.checkpoint)")

        if not self.cfg.layer_classes:
            logger.warning(
                "No layer classes specified for activation checkpointing. "
                "Please specify 'layer_classes' in the config (e.g. ['LlamaBlock'])."
            )
            return model

        target_classes = set(self.cfg.layer_classes)
        ops_to_save = [
            eval(op, {"__builtins__": None}, {"torch": torch}) for op in (self.cfg.ops_to_save or [])
        ]
        replaced_count = self._replace_modules(
            model, target_classes, use_reentrant=self.cfg.use_reentrant, ops_to_save=ops_to_save
        )

        if replaced_count == 0:
            logger.warning(f"No modules matching {target_classes} found to checkpoint.")
        else:
            logger.info(
                f"Applied activation checkpointing to {replaced_count} layers of types: {target_classes}"
            )

        return model

    def _replace_modules(
        self, model: nn.Module, target_classes: set, use_reentrant: bool, ops_to_save: list
    ) -> int:
        """Recursively replace target modules with CheckpointWrapper."""
        count = 0
        for name, child in model.named_children():
            if child.__class__.__name__ in target_classes:
                # Replace the module
                logger.debug(f"Wrapping {name} ({child.__class__.__name__})")
                wrapped_child = CheckpointWrapper(child, use_reentrant=use_reentrant, ops_to_save=ops_to_save)
                setattr(model, name, wrapped_child)
                count += 1
            else:
                # Recurse
                count += self._replace_modules(child, target_classes, use_reentrant, ops_to_save)
        return count
