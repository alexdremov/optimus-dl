from dataclasses import dataclass

import torch

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.optim import register_optimizer


@dataclass
class AdamWConfig(RegistryConfig):
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
def make_adamw(cfg, params, **kwargs):
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
