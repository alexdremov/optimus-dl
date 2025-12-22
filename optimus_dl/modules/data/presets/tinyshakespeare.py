from dataclasses import dataclass

from optimus_dl.modules.data import register_dataset
from optimus_dl.modules.data.datasets import (
    TxtLinesDataset,
    TxtLinesDatasetConfig,
)


@dataclass
class Config(TxtLinesDatasetConfig):
    def __post_init__(self):
        self.file_link = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@register_dataset("preset_tinyshakespeare", Config)
def make_dataset(cfg, rank=0, world_size=1, **_):
    return TxtLinesDataset(cfg, rank=rank, world_size=world_size)
