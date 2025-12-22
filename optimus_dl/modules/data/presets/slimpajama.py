from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.data import register_dataset
from optimus_dl.modules.data.datasets import (
    HuggingFaceDataset,
    HuggingFaceDatasetConfig,
)


@dataclass
class Config(RegistryConfig):
    split: str = "train"
    streaming: bool = True


@register_dataset("preset_slimpajama6b", Config)
def make_dataset_6b(cfg, rank: int, world_size: int, **_):
    config = HuggingFaceDatasetConfig(
        dataset_load_kwargs={
            "path": "DKYoon/SlimPajama-6B",
            "split": cfg.split,
            "streaming": cfg.streaming,
        }
    )
    return HuggingFaceDataset(config, rank=rank, world_size=world_size)


@register_dataset("preset_slimpajama", Config)
def make_dataset_full(cfg, rank: int, world_size: int, **_):
    config = HuggingFaceDatasetConfig(
        dataset_load_kwargs={
            "path": "cerebras/SlimPajama-627B",
            "split": cfg.split,
            "streaming": cfg.streaming,
        }
    )
    return HuggingFaceDataset(config, rank=rank, world_size=world_size)
