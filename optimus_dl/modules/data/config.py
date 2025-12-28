from dataclasses import (
    dataclass,
    field,
)
from typing import Any

from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfig


@dataclass
class DataPipelineConfig:
    source: RegistryConfig
    transform: RegistryConfig | None = None


@dataclass
class DataConfig:
    train_datasets: DataPipelineConfig = MISSING
    eval_datasets: dict[str, DataPipelineConfig] = field(default_factory=dict)

    scratch: Any = None
