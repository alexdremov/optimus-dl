"""
Muon optimizer
"""

from dataclasses import dataclass

import torch
import torch.optim
from torch.optim._muon import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
)

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.optim import register_optimizer


@dataclass
class MuonConfig(RegistryConfigStrict):
    """Configuration for Muon optimizer.

    Muon is a momentum-based optimizer that uses Newton-Schulz iteration for
    preconditioning. It's designed for efficient training of large models.

    Attributes:
        lr: Learning rate for parameter updates.
        weight_decay: Weight decay (L2 penalty) coefficient applied to parameters.
        momentum: Momentum factor for the moving average of gradients.
        nesterov: Whether to use Nesterov momentum.
        ns_coefficients: Coefficients (a, b, c) for Newton-Schulz iteration algorithm.
            These control the convergence and stability of the preconditioning step.
            Default values are tuned for typical use cases.
        eps: Small constant added for numerical stability in computations.
        ns_steps: Number of Newton-Schulz iteration steps to perform for preconditioning.
            Higher values can improve accuracy but increase computational cost.
        adjust_lr_fn: Optional learning rate adjustment function name. If provided,
            applies dynamic learning rate scaling during training.
    """

    lr: float = 1e-3
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C)
    eps: float = EPS
    ns_steps: int = DEFAULT_NS_STEPS
    adjust_lr_fn: str | None = None


@register_optimizer("muon", MuonConfig)
def make_muon(cfg, params, **_):
    return torch.optim.Muon(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
        ns_coefficients=cfg.ns_coefficients,
        eps=cfg.eps,
        ns_steps=cfg.ns_steps,
        adjust_lr_fn=cfg.adjust_lr_fn,
    )
