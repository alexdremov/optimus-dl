from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer

from optimus_dl.core.registry import RegistryConfig


@dataclass
class BaseLRSchedulerConfig(RegistryConfig):
    """Base configuration for learning rate schedulers"""

    pass


class BaseLRScheduler(ABC):
    """Base class for learning rate schedulers"""

    def __init__(self, optimizer: Optimizer, **kwargs):
        self.optimizer = optimizer
        self._step_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    @abstractmethod
    def get_lr(self) -> list[float]:
        """Calculate learning rates for current step"""
        pass

    def step(self) -> None:
        """Update learning rates"""
        self._step_count += 1
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values, strict=True):
            param_group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        """Get the last computed learning rates"""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state"""
        return {
            "step_count": self._step_count,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scheduler state"""
        self._step_count = state_dict["step_count"]
        self.base_lrs = state_dict["base_lrs"]

    @property
    def last_epoch(self) -> int:
        """Compatibility property with PyTorch schedulers"""
        return self._step_count
