from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer

from optimus_dl.core.registry import RegistryConfig


@dataclass
class BaseLRSchedulerConfig(RegistryConfig):
    """Base configuration for learning rate schedulers."""

    pass


class BaseLRScheduler(ABC):
    """Abstract base class for learning rate schedulers.

    This class provides a uniform interface for learning rate scheduling that
    is decoupled from specific optimizer implementations. It manages the
    stepping of learning rates across multiple parameter groups and handles
    state serialization for checkpointing.

    Attributes:
        optimizer: The PyTorch optimizer whose learning rates are managed.
        base_lrs: Initial learning rates for each parameter group.
    """

    def __init__(self, optimizer: Optimizer, **kwargs):
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer to manage.
            **kwargs: Additional keyword arguments.
        """
        self.optimizer = optimizer
        self._step_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.set()

    @abstractmethod
    def get_lr(self) -> list[float]:
        """Calculate the target learning rates for the current step.

        Returns:
            List of floats representing the new learning rates for each
            parameter group in the optimizer.
        """
        pass

    def step(self) -> None:
        """Update the optimizer's learning rates based on the current step count.

        This should be called at the end of each training iteration.
        """
        self._step_count += 1
        self.set()

    def set(self) -> None:
        """Set the learning rates of the optimizer to the current values."""
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values, strict=True):
            param_group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        """Return the most recently computed learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Return the scheduler's state for checkpointing."""
        return {
            "step_count": self._step_count,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the scheduler's state from a checkpoint."""
        self._step_count = state_dict["step_count"]
        self.base_lrs = state_dict["base_lrs"]
        self.set()

    @property
    def last_epoch(self) -> int:
        """The current step count (for compatibility with PyTorch schedulers)."""
        return self._step_count
