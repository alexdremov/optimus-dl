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
class WSDSchedulerConfig(BaseLRSchedulerConfig):
    """Configuration for WSD (Warmup, Sustain, Decay) learning rate scheduler.

    Attributes:
        final_lr_factor: Factor of base_lr for the final learning rate.
        warmup_steps: Number of iterations for linear warmup.
        warmup_steps_fraction: Fraction of total iterations for warmup.
        init_div_factor: Initial division factor for start of warmup (1/X).
        fract_decay: Fraction of total iterations dedicated to decay phase.
        decay_type: Strategy for decay ('linear', 'cosine', 'exp', etc.).
        sqrt_power: Power for 'sqrt' decay strategy.
        linear_pw_subdivisions: Intermediate factors for piecewise linear decay.
        cooldown_start_lr_factor: LR factor at the start of decay phase.
    """

    final_lr_factor: float = 0.0  # factor by which to reduce max_lr at the end
    warmup_steps: int | None = 300  # number of warmup iterations
    warmup_steps_fraction: float | None = None  # fraction of iterations used for warmup
    init_div_factor: int = 100  # initial division factor for warmup
    fract_decay: float = 0.1  # fraction of iterations used for decay
    decay_type: str = (
        "linear"  # type of decay: linear, linear_pw, exp, cosine, miror_cosine, square, sqrt
    )
    sqrt_power: float = 0.5  # power for sqrt decay type
    linear_pw_subdivisions: list[float] | None = (
        None  # subdivisions for linear_pw decay
    )
    cooldown_start_lr_factor: float = 1.0  # starting factor for cooldown phase


@register_lr_scheduler("wsd", WSDSchedulerConfig)
class WSDScheduler(BaseLRScheduler):
    """WSD (Warmup, Sustain, Decay) learning rate scheduler.

    This scheduler is designed for pre-training large models and consists of:
    1. **Warmup**: Linear increase from `base_lr / init_div_factor` to `base_lr`.
    2. **Sustain**: Constant learning rate at `base_lr`.
    3. **Decay (Cooldown)**: Decrease from `base_lr` to `base_lr * final_lr_factor`.

    The decay phase supports multiple shapes, including linear, cosine, and
    piecewise linear.

    Args:
        cfg: Scheduler configuration.
        optimizer: Managed optimizer.
        iterations: Total training iterations.
    """

    def __init__(
        self, cfg: WSDSchedulerConfig, optimizer: Optimizer, iterations: int, **kwargs
    ):
        super().__init__(optimizer)

        assert (
            cfg.warmup_steps is not None or cfg.warmup_steps_fraction is not None
        ), "Either warmup_steps or warmup_steps_fraction must be specified"
        if cfg.warmup_steps is None:
            assert cfg.warmup_steps_fraction is not None
            cfg.warmup_steps = int(cfg.warmup_steps_fraction * iterations)

        self.iterations = iterations
        self.final_lr_factor = cfg.final_lr_factor
        self.warmup_steps = cfg.warmup_steps
        self.init_div_factor = cfg.init_div_factor
        self.fract_decay = cfg.fract_decay
        self.decay_type = cfg.decay_type
        self.sqrt_power = cfg.sqrt_power
        self.linear_pw_subdivisions = cfg.linear_pw_subdivisions or []
        self.cooldown_start_lr_factor = cfg.cooldown_start_lr_factor

        # Calculate phase boundaries
        self.n_anneal_steps = int(self.fract_decay * iterations)
        self.n_hold = iterations - self.n_anneal_steps

        # Validate decay type
        valid_decay_types = [
            "linear",
            "linear_pw",
            "exp",
            "cosine",
            "miror_cosine",
            "square",
            "sqrt",
        ]
        if self.decay_type not in valid_decay_types:
            raise ValueError(
                f"decay_type {self.decay_type} is not in {valid_decay_types}"
            )

        self.set()

    def get_lr(self) -> list[float]:
        """Calculate learning rates using the WSD formula for the current step."""
        step = self._step_count
        lr_factor = self._get_lr_factor(step)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def _get_lr_factor(self, step: int) -> float:
        """Identify current phase and compute the corresponding LR factor."""
        if step < self.warmup_steps:
            # Warmup phase: linear interpolation from 1/init_div_factor to 1.0
            return (step / self.warmup_steps) + (
                1 - step / self.warmup_steps
            ) / self.init_div_factor
        elif step < self.n_hold:
            # Hold phase: constant at 1.0
            return 1.0
        elif step < self.iterations:
            # Decay phase: various decay strategies
            return self._get_decay_factor(step)
        else:
            # Past end: final learning rate factor
            return self.final_lr_factor

    def _get_decay_factor(self, step: int) -> float:
        """Compute decay factor shape based on configuration."""
        if self.decay_type == "linear":
            progress = (step - self.n_hold) / self.n_anneal_steps
            return self.final_lr_factor + (
                self.cooldown_start_lr_factor - self.final_lr_factor
            ) * (1 - progress)

        elif self.decay_type == "linear_pw":
            subdivisions = (
                [self.cooldown_start_lr_factor]
                + self.linear_pw_subdivisions
                + [self.final_lr_factor]
            )
            division_step = 1 / (len(subdivisions) - 1)

            cooldown_fraction = (step - self.n_hold) / self.n_anneal_steps
            now_subdivision = math.floor(cooldown_fraction / division_step)
            now_subdivision = min(
                now_subdivision, len(subdivisions) - 2
            )  # Ensure we don't go out of bounds

            left_frac, right_frac = (
                subdivisions[now_subdivision],
                subdivisions[now_subdivision + 1],
            )
            local_fraction = (
                cooldown_fraction - division_step * now_subdivision
            ) / division_step
            return left_frac + (right_frac - left_frac) * local_fraction

        elif self.decay_type == "exp":
            progress = (step - self.n_hold) / self.n_anneal_steps
            return self.final_lr_factor**progress

        elif self.decay_type == "cosine":
            progress = (step - self.n_hold) / self.n_anneal_steps
            return (
                self.final_lr_factor
                + (self.cooldown_start_lr_factor - self.final_lr_factor)
                * (1 + math.cos(math.pi * progress))
                * 0.5
            )

        elif self.decay_type == "miror_cosine":
            progress = (step - self.n_hold) / self.n_anneal_steps
            cosine_value = (
                self.final_lr_factor
                + (self.cooldown_start_lr_factor - self.final_lr_factor)
                * (1 + math.cos(math.pi * progress))
                * 0.5
            )
            linear_value = self.final_lr_factor + (
                self.cooldown_start_lr_factor - self.final_lr_factor
            ) * (1 - progress)
            return linear_value * 2 - cosine_value

        elif self.decay_type == "square":
            progress = (step - self.n_hold) / self.n_anneal_steps
            return self.final_lr_factor + (
                self.cooldown_start_lr_factor - self.final_lr_factor
            ) * (1 - progress**2)

        elif self.decay_type == "sqrt":
            progress = (step - self.n_hold) / self.n_anneal_steps
            return self.final_lr_factor + (
                self.cooldown_start_lr_factor - self.final_lr_factor
            ) * (1 - progress**self.sqrt_power)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")

    def state_dict(self) -> dict[str, Any]:
        """Return the scheduler's state, including WSD-specific parameters."""
        state = super().state_dict()
        state.update(
            {
                "iterations": self.iterations,
                "final_lr_factor": self.final_lr_factor,
                "warmup_steps": self.warmup_steps,
                "init_div_factor": self.init_div_factor,
                "fract_decay": self.fract_decay,
                "decay_type": self.decay_type,
                "sqrt_power": self.sqrt_power,
                "linear_pw_subdivisions": self.linear_pw_subdivisions,
                "cooldown_start_lr_factor": self.cooldown_start_lr_factor,
                "n_anneal_steps": self.n_anneal_steps,
                "n_hold": self.n_hold,
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the scheduler's state."""
        super().load_state_dict(state_dict)
        self.iterations = state_dict["iterations"]
        self.final_lr_factor = state_dict["final_lr_factor"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.init_div_factor = state_dict["init_div_factor"]
        self.fract_decay = state_dict["fract_decay"]
        self.decay_type = state_dict["decay_type"]
        self.sqrt_power = state_dict["sqrt_power"]
        self.linear_pw_subdivisions = state_dict["linear_pw_subdivisions"]
        self.cooldown_start_lr_factor = state_dict["cooldown_start_lr_factor"]
        self.n_anneal_steps = state_dict["n_anneal_steps"]
        self.n_hold = state_dict["n_hold"]
        self.set()
