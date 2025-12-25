from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data import register_dataset
from optimus_dl.modules.data.datasets import (
    HuggingFaceDataset,
    HuggingFaceDatasetConfig,
)


@dataclass
class Config(RegistryConfigStrict):
    subset: str = "sample-10BT"


@register_dataset("preset_fineweb_edu", Config)
def make_dataset(cfg, rank=0, world_size=1, **_):
    config = HuggingFaceDatasetConfig(
        dataset_load_kwargs={
            "path": "HuggingFaceFW/fineweb-edu",
            "split": "train",
            "name": cfg.subset,
        }
    )
    return HuggingFaceDataset(config, rank=rank, world_size=world_size)
