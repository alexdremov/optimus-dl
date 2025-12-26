"""Base model class for all language models in Optimus-DL.

This module defines the BaseModel class that all model architectures must inherit from.
It provides common functionality for parameter grouping, distributed sharding, and
tensor parallelism.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch.nn

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module):
    """Base class for all language model architectures.

    All model implementations in Optimus-DL should inherit from this class. It provides:
    - Parameter grouping for optimizers (e.g., different learning rates per layer)
    - FSDP2 sharding support
    - Tensor parallelism support

    Subclasses should implement:
    - `forward()`: The forward pass of the model
    - Optionally `fully_shard()`: Custom FSDP2 sharding strategy
    - Optionally `apply_tp()`: Tensor parallelism plan

    Example:
        >>> @register_model("my_model", MyModelConfig)
        >>> class MyModel(BaseModel):
        ...     def __init__(self, cfg: MyModelConfig):
        ...         super().__init__()
        ...         self.embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        ...
        ...     def forward(self, input_ids):
        ...         return self.embedding(input_ids)
    """

    def __init__(self):
        """Initialize the base model.

        Subclasses should call super().__init__() in their __init__ methods.
        """
        super().__init__()

    @classmethod
    def register_arch(cls, arch_name: str) -> Callable[[Callable[[], Any]], Any]:
        """Register an architecture variant of this model.

        This decorator is automatically added to model classes when they are
        registered via the registry system. It allows registering multiple
        size variants (e.g., "llama-7b", "llama-13b", "llama-70b").

        Args:
            arch_name: Name of the architecture variant.

        Returns:
            A decorator function that takes a config factory method.

        Note:
            This is a placeholder that gets replaced during model registration.
            Do not call this directly.
        """
        raise NotImplementedError(
            "This is a placeholder for the register_arch decorator. Populated on model class registration"
        )

    def make_parameter_groups(self) -> dict[str, Any]:
        """Create parameter groups for optimizer configuration.

        Returns a dictionary of parameter groups that can be passed to an optimizer.
        By default, returns all parameters in a single group. Subclasses can override
        this to create custom parameter groups (e.g., different learning rates for
        embeddings vs. transformer layers).

        Returns:
            Dictionary with "params" key containing an iterator over named parameters.
            Can be extended with additional keys like "lr", "weight_decay", etc.

        Example:
            >>> # Default: single parameter group
            >>> groups = model.make_parameter_groups()
            >>> optimizer = torch.optim.AdamW(groups["params"])
            >>>
            >>> # Custom: different LR for embeddings
            >>> def make_parameter_groups(self):
            ...     embed_params = list(self.embedding.parameters())
            ...     other_params = [p for n, p in self.named_parameters()
            ...                     if 'embedding' not in n]
            ...     return [
            ...         {"params": embed_params, "lr": 1e-4},
            ...         {"params": other_params, "lr": 1e-3},
            ...     ]
        """
        return {"params": self.named_parameters()}

    def fully_shard(self, **fsdp_kwargs) -> None:
        """Define how the model should be fully sharded for FSDP2.

        This method is called by the FSDP2 transform to apply model sharding.
        By default, it logs a warning. Subclasses should override this to
        implement custom sharding strategies if needed.

        Args:
            **fsdp_kwargs: Keyword arguments passed from FSDP2 transform,
                including mesh, mp_policy, offload_policy, etc.

        Note:
            Most models don't need to override this. The FSDP2 transform will
            automatically shard the model. Override only if you need custom
            sharding behavior (e.g., different sharding for different layers).

        Example:
            >>> def fully_shard(self, **fsdp_kwargs):
            ...     # Custom sharding: don't shard embeddings
            ...     from torch.distributed.fsdp import fully_shard
            ...     fully_shard(self.embedding)  # Don't shard
            ...     for layer in self.layers:
            ...         fully_shard(layer, **fsdp_kwargs)  # Shard each layer
        """
        logger.warning(
            "Model does not support fully sharding. Define this method or performance will be impacted."
        )

    def apply_tp(self, mesh, **kwargs):
        """Apply Tensor Parallelism plan for this model.

        This method defines how the model should be sharded across devices for
        tensor parallelism. It returns a mapping from fully qualified parameter
        names (or regex patterns) to parallel styles.

        Args:
            mesh: The DeviceMesh for tensor parallelism, defining which devices
                participate in the parallel group.
            **kwargs: Additional model-specific TP parameters
        """
        return {}
