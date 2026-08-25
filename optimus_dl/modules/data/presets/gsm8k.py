from dataclasses import dataclass

from optimus_dl.modules.data import register_dataset
from optimus_dl.modules.data.datasets.huggingface import (
    HuggingFaceDataset,
    HuggingFaceDatasetConfig,
)


@dataclass
class GSM8kConfig(HuggingFaceDatasetConfig):
    def __post_init__(self):
        # Provide default kwargs if none are set
        if self.dataset_load_kwargs is None or self.dataset_load_kwargs == "???":
            self.dataset_load_kwargs = {}

        self.dataset_load_kwargs.setdefault("path", "gsm8k")
        self.dataset_load_kwargs.setdefault("name", "main")


@register_dataset("preset_gsm8k", GSM8kConfig)
def make_gsm8k(cfg, rank=0, world_size=1, **_):
    return HuggingFaceDataset(cfg, rank=rank, world_size=world_size)
