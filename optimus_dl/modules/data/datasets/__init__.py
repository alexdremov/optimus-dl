from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import BaseDataset

_DATASETS_REGISTRY, register_dataset, build_dataset = make_registry(
    "dataset", BaseDataset
)

from .huggingface import HuggingFaceDataset, HuggingFaceDatasetConfig
from .loop_dataset import LoopDataset, LoopDatasetConfig
from .tokenized_dataset import TokenizedDataset, TokenizedDatasetConfig
from .tokenized_flat_dataset import TokenizedFlatDataset, TokenizedFlatDatasetConfig
from .txt_lines import TxtLinesDataset, TxtLinesDatasetConfig

bootstrap_module(__name__)
