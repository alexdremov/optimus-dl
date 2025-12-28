import math
from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer

from . import register_lr_scheduler
from .base import (
    BaseLRScheduler,
    BaseLRSchedulerConfig,
)


@dataclass
class CosineAnnealingLRConfig(BaseLRSchedulerConfig):
    """Configuration for cosine annealing learning rate scheduler.

    Attributes:
        T_max: Maximum number of iterations for the decay.
        eta_min: Minimum learning rate (bottom of the cosine curve).
        last_epoch: Initial step count for resuming.
    """

    T_max: int = 1000  # Maximum number of iterations
    eta_min: float = 0.0  # Minimum learning rate
    last_epoch: int = -1  # Last epoch (for resuming)


@register_lr_scheduler("cosine_annealing", CosineAnnealingLRConfig)
class CosineAnnealingLR(BaseLRScheduler):
    """Cosine annealing learning rate scheduler.

    Decays the learning rate using a cosine curve following:
    ```
    lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * step / T_max)) / 2
    ```
    This implementation allows the learning rate to smoothly transition from
    its initial value down to `eta_min` over `T_max` steps.

    Args:
        cfg: Scheduler configuration.
        optimizer: Managed optimizer.
        iterations: Total training iterations (used if T_max is not specified).
    """

    def __init__(
        self,
        cfg: CosineAnnealingLRConfig,
        optimizer: Optimizer,
        iterations: int,
        **kwargs: Any,
    ):
        super().__init__(optimizer)
        self.T_max = iterations
        self.eta_min = cfg.eta_min
        self._step_count = cfg.last_epoch + 1

    def get_lr(self) -> list[float]:
        """Calculate learning rates using the cosine annealing formula."""
        if self._step_count == 0:
            return self.base_lrs

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self._step_count / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]

    def state_dict(self) -> dict[str, any]:
        """Return the scheduler's state, including cosine-specific parameters."""
        state = super().state_dict()
        state.update(
            {
                "T_max": self.T_max,
                "eta_min": self.eta_min,
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, any]) -> None:
        """Restore the scheduler's state."""
        super().load_state_dict(state_dict)
        self.T_max = state_dict["T_max"]
        self.eta_min = state_dict["eta_min"]
