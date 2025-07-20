import logging
from collections.abc import Callable
from typing import Any

import torch.nn
from omegaconf import DictConfig  # Import for tokenizer_config type hint

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def register_arch(cls, arch_name: str) -> Callable[[Callable[[], Any]], Any]:
        raise NotImplementedError(
            "This is a placeholder for the register_arch decorator. Populated on model class registration"
        )

    def make_parameter_groups(self):
        return {"params": self.named_parameters()}

    def fully_shard(self, **fsdp_kwargs):
        """Defines how the model should be fully sharded."""
        logger.warning(
            "Model does not support fully sharding. Define this method or performance will be impacted."
        )
