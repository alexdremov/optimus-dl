"""
AdamW optimizer
"""

from dataclasses import dataclass

import torch

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.optim import register_optimizer


@dataclass
class AdamWConfig(RegistryConfigStrict):
    """Configuration for AdamW optimizer.

    Attributes:
        lr: Learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to denominator for numerical stability.
        weight_decay: Weight decay (L2 penalty) coefficient.
        amsgrad: Whether to use the AMSGrad variant.
        maximize: Whether to maximize the objective or minimize it.
        foreach: Whether to use the faster 'foreach' implementation.
        capturable: Whether this instance is safe to capture in a CUDA graph.
        differentiable: Whether autograd should occur through the optimizer step.
        fused: Whether to use the fused kernel implementation (recommended for GPU).
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool | None = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool = True


@register_optimizer("adamw", AdamWConfig)
def make_adamw(cfg, params, **_):
    """Instantiate a PyTorch AdamW optimizer from the given configuration."""
    return torch.optim.AdamW(
        params=params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        amsgrad=cfg.amsgrad,
        maximize=cfg.maximize,
        foreach=cfg.foreach,
        capturable=cfg.capturable,
        differentiable=cfg.differentiable,
        fused=cfg.fused,
    )
