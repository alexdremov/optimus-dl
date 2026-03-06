from contextlib import nullcontext
from typing import NamedTuple

import torchdata
import torchdata.nodes
from omegaconf import DictConfig

from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.modules.data.datasets import (
    build_dataset,
    register_dataset,
)
from optimus_dl.modules.data.profiling import (
    PipelineProfiler,
    ProfilingProxyNode,
    _active_profiler,
)
from optimus_dl.modules.data.transforms.composite import (
    build_transform,
    register_transform,
)

from .config import (
    DataConfig,
    DataPipelineConfig,
    EvalDataPipelineConfig,
)


class DataPipeline(NamedTuple):
    datasets: torchdata.nodes.BaseNode
    dataloader: torchdata.nodes.BaseNode | torchdata.nodes.Loader


class EvalDataPipeline(NamedTuple):
    datasets: torchdata.nodes.BaseNode
    dataloader: torchdata.nodes.BaseNode | torchdata.nodes.Loader
    eval_freq: int | None
    eval_iterations: int | None


def build_data_pipeline(
    cfg: DataPipelineConfig | EvalDataPipelineConfig, profile_name: str, **kwargs
) -> DataPipeline | EvalDataPipeline | None:
    if cfg is None:
        return None
    dataset = build_dataset(cfg.source, **kwargs)
    pipeline = dataset

    profiler = None
    if cfg.profile and cfg.report_freq > 0:
        profiler = PipelineProfiler(name=profile_name, report_freq=cfg.report_freq)
        pipeline = ProfilingProxyNode(pipeline, name=repr(dataset), profiler=profiler)
        profiler.root_nodes.append(pipeline)

    profiler_token = _active_profiler.set(profiler)
    try:

        if cfg.transform is not None:
            transform = build_transform(cfg.transform, **kwargs)
            pipeline = transform.build(pipeline)

        # Mark the final node as root for periodic reporting
        if isinstance(pipeline, ProfilingProxyNode):
            pipeline._is_root = True
    finally:
        _active_profiler.reset(profiler_token)

    assert isinstance(pipeline, torchdata.nodes.BaseNode)
    if hasattr(cfg, "eval_freq") or hasattr(cfg, "eval_iterations"):
        return EvalDataPipeline(
            datasets=dataset,
            dataloader=pipeline,
            eval_freq=getattr(cfg, "eval_freq", None),
            eval_iterations=getattr(cfg, "eval_iterations", None),
        )
    else:
        return DataPipeline(datasets=dataset, dataloader=pipeline)


def build_data_pipeline_dict(
    cfg: dict[str, DataPipelineConfig], profile_name: str, **kwargs
) -> dict[str, DataPipeline | EvalDataPipeline | None]:
    return {
        k: build_data_pipeline(v, profile_name=f"{profile_name}-{k}", **kwargs)
        for k, v in cfg.items()
    }


bootstrap_module(__name__)
