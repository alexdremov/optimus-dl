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


def build_data_pipeline(
    cfg: DataPipelineConfig | dict | None, **kwargs
) -> torchdata.nodes.BaseNode:
    if cfg is None:
        return None
    dataset = build_dataset(cfg.source, **kwargs)
    pipeline = dataset
    if cfg.transform is not None:
        transform = build_transform(cfg.transform, **kwargs)
        pipeline = transform.build(dataset)
    assert isinstance(pipeline, torchdata.nodes.BaseNode)
    return pipeline


def build_data_pipeline_dict(cfg: DataPipelineConfig | dict | None, **kwargs):
    return {k: build_data_pipeline(v, **kwargs) for k, v in cfg.items()}


bootstrap_module(__name__)
