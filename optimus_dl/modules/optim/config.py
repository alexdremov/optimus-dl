"""
General optimizer config
"""

from dataclasses import dataclass, field

from optimus_dl.core.registry import RegistryConfig


@dataclass
class AmpConfig:
    enabled: bool = False
    dtype: str = "torch.bfloat16"

    enable_scaler: bool = '${eval: \'"${.dtype}" == "torch.float16"\'}'
    init_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000


@dataclass
class OptimizationConfig:
    optimizer: RegistryConfig

    iterations: int = field(default=1000, metadata={"description": "Total train steps"})
    acc_steps: int = field(
        default=1, metadata={"description": "Steps to accumulate gradient"}
    )
    clip_grad_norm: float | None = field(
        default=None, metadata={"description": "Clip gradient norm"}
    )
    amp: AmpConfig = field(default_factory=AmpConfig)
