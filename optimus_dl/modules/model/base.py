"""Base model class for all language models in Optimus-DL.

This module defines the BaseModel class that all model architectures must inherit from.
It provides common functionality for parameter grouping, distributed sharding, and
tensor parallelism.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch.nn

from optimus_dl.modules.metrics.source import StandardProtocols

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module):
    """Base class for all language model architectures in the framework.

    All model implementations should inherit from this class. It provides a
    standardized interface for:

    - **Forward Pass**: Standard PyTorch forward method.
    - **Optimizer Integration**: Custom parameter grouping (e.g., weight decay
      exclusion for norms/biases).
    - **FSDP2 Sharding**: Support for fully sharded data parallelism via a custom
      `fully_shard` method.
    - **Tensor Parallelism**: Support for sharding parameters across multiple
      devices via `apply_tp`.

    Subclasses must implement:

    - `forward()`: The main computation loop.

    Example:
        ```python
        @register_model("my_model", MyModelConfig)
        class MyModel(BaseModel):
            def __init__(self, cfg: MyModelConfig):
                super().__init__()
                self.embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)

            def forward(self, input_ids):
                return {"logits": self.embedding(input_ids)}

        ```"""

    def __init__(self):
        """Initialize the base model."""
        super().__init__()

    @property
    def provides(self) -> set[str]:
        """Returns the set of protocol strings this model natively provides.
        
        By default, language models return LOGITS. Subclasses can override this
        if they provide additional or different protocols (like generations).
        """
        return {StandardProtocols.LOGITS}

    @classmethod
    def register_arch(cls, arch_name: str) -> Callable[[Callable[[], Any]], Any]:
        """Decorator for registering an architecture variant of this model.

        This method is dynamically populated on the class during registration
        in the model registry. It allows registering variants like '7b', '13b', etc.

        Args:
            arch_name: Name of the architecture variant.

        Returns:
            A decorator function.
        """
        raise NotImplementedError(
            "This is a placeholder for the register_arch decorator. Populated on model class registration"
        )

    def make_parameter_groups(self) -> dict[str, Any]:
        """Create parameter groups for optimizer configuration.

        This method allows models to specify which parameters should have
        weight decay applied, or to use different learning rates for different
        sub-modules.

        Returns:
            Dictionary with a 'params' key, or a list of such dictionaries,
            compatible with PyTorch optimizers.
        """
        return {"params": self.named_parameters()}

    def fully_shard(self, **fsdp_kwargs) -> None:
        """Define the FSDP2 sharding strategy for this model.

        This method should wrap sub-modules (e.g., transformer blocks) with
        `fully_shard` to enable efficient distributed training.

        Args:
            **fsdp_kwargs: Arguments for the FSDP sharding process (e.g., mesh).
        """
        logger.warning(
            "Model does not support fully sharding. Define this method or performance will be impacted."
        )

    def apply_tp(self, mesh, **kwargs):
        """Apply Tensor Parallelism (sharding) to the model's parameters.

        This method should use `parallelize_module` or similar utilities to
        shard specific linear or embedding layers across the provided mesh.

        Args:
            mesh: The DeviceMesh for tensor parallelism.
            **kwargs: Additional model-specific TP flags (e.g., sequence_parallel).
        """
        ...

