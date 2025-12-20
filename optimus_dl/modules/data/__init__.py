from typing import NamedTuple

import torchdata
import torchdata.nodes
from omegaconf import DictConfig

from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.modules.data.datasets import build_dataset, register_dataset
from optimus_dl.modules.data.transforms.composite import (
    build_transform,
    register_transform,
)

from .config import DataConfig, DataPipelineConfig


class DataPipeline(NamedTuple):
    datasets: torchdata.nodes.BaseNode
    dataloader: torchdata.nodes.BaseNode


def build_data_pipeline(cfg: DataPipelineConfig, **kwargs) -> DataPipeline | None:
    if cfg is None:
        return None
    dataset = build_dataset(cfg.source, **kwargs)
    pipeline = dataset
    if cfg.transform is not None:
        transform = build_transform(cfg.transform, **kwargs)
        pipeline = transform.build(dataset)
    assert isinstance(pipeline, torchdata.nodes.BaseNode)
    return DataPipeline(dataset, pipeline)


def build_data_pipeline_dict(
    cfg: dict[str, DataPipelineConfig], **kwargs
) -> dict[str, DataPipeline | None]:
    return {k: build_data_pipeline(v, **kwargs) for k, v in cfg.items()}


bootstrap_module(__name__)
