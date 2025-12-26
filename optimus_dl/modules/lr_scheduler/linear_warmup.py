from dataclasses import dataclass

from torch.optim import Optimizer

from . import register_lr_scheduler
from .base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class LinearWarmupLRConfig(BaseLRSchedulerConfig):
    """Configuration for linear warmup learning rate scheduler.

    Attributes:
        warmup_steps: Number of iterations for the linear warmup.
        warmup_percent: Fraction of total iterations for warmup (used if
            warmup_steps is None).
        target_lr: Final learning rate after warmup (defaults to base_lr).
        start_lr: Initial learning rate at step 0.
    """

    warmup_steps: int | None = None
    warmup_percent: float | None = 0.05  # Percentage of total steps for warmup
    target_lr: float | None = None  # Target learning rate (defaults to base_lr)
    start_lr: float = 0.0  # Starting learning rate


@register_lr_scheduler("linear_warmup", LinearWarmupLRConfig)
class LinearWarmupLR(BaseLRScheduler):
    """Linear warmup learning rate scheduler.

    Linearly increases the learning rate from `start_lr` to `target_lr` over a
    specified number of steps. Once the warmup phase is complete, the learning
    rate is held constant at `target_lr`.

    Args:
        cfg: Scheduler configuration.
        optimizer: Managed optimizer.
        iterations: Total training iterations (used to calculate warmup steps
            if configured by percentage).
    """

    def __init__(
        self, cfg: LinearWarmupLRConfig, optimizer: Optimizer, iterations: int, **kwargs
    ):
        super().__init__(optimizer)
        if cfg.warmup_steps is None:
            if cfg.warmup_percent is not None:
                cfg.warmup_steps = int(iterations * cfg.warmup_percent)
            else:
                raise ValueError("Either warmup_steps or warmup_percent must be set")
        self.warmup_steps = cfg.warmup_steps
        self.start_lr = cfg.start_lr
        self.target_lrs = [cfg.target_lr or base_lr for base_lr in self.base_lrs]

    def get_lr(self) -> list[float]:
        """Calculate learning rates using the linear warmup formula."""
        if self.warmup_steps == 0 or self._step_count > self.warmup_steps:
            # No warmup or post-warmup: maintain target learning rate
            return self.target_lrs.copy()
        else:
            # Linear warmup phase
            warmup_factor = self._step_count / self.warmup_steps
            return [
                self.start_lr + (target_lr - self.start_lr) * warmup_factor
                for target_lr in self.target_lrs
            ]

    def state_dict(self) -> dict[str, any]:
        """Return the scheduler's state, including warmup-specific parameters."""
        state = super().state_dict()
        state.update(
            {
                "warmup_steps": self.warmup_steps,
                "start_lr": self.start_lr,
                "target_lrs": self.target_lrs,
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, any]) -> None:
        """Restore the scheduler's state."""
        super().load_state_dict(state_dict)
        self.warmup_steps = state_dict["warmup_steps"]
        self.start_lr = state_dict["start_lr"]
        self.target_lrs = state_dict["target_lrs"]
