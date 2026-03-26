import logging
from typing import (
    Any,
    NamedTuple,
)

import torchdata
import torchdata.nodes

from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.modules.data.datasets import (
    build_dataset,
    register_dataset,
)
from optimus_dl.modules.data.profiling import (
    PipelineProfiler,
    ProfilingProxyNode,
    scope_profiler,
)
from optimus_dl.modules.data.transforms import (
    build_transform,
    register_transform,
)

from .config import (
    DataConfig,
    DataPipelineConfig,
    EvalDataPipelineConfig,
)

logger = logging.getLogger(__name__)


class DataPipeline(NamedTuple):
    datasets: torchdata.nodes.BaseNode
    dataloader: torchdata.nodes.BaseNode | torchdata.nodes.Loader


class EvalDataPipeline(NamedTuple):
    datasets: torchdata.nodes.BaseNode
    dataloader: torchdata.nodes.BaseNode | torchdata.nodes.Loader
    eval_freq: int | None
    eval_iterations: int | None
    eval_guaranteed_same_batches: bool | None


class LoggingDataNode(torchdata.nodes.BaseNode):
    """A simple node that logs reset calls and delegates to a source node. Useful for debugging data pipelines."""

    def __init__(self, name: str, source: torchdata.nodes.BaseNode):
        super().__init__()
        self.name = name
        self.source = source

    def reset(self, initial_state=None):
        logger.info(f"Resetting data node {self.name}")
        super().reset(initial_state)
        self.source.reset(initial_state)

    def get_state(self) -> dict[str, Any]:
        return self.source.get_state()

    def next(self):
        return self.source.next()


def build_data_pipeline(
    cfg: DataPipelineConfig | EvalDataPipelineConfig,
    profile_name: str | None = None,
    **kwargs,
) -> DataPipeline | EvalDataPipeline | None:
    if cfg is None:
        return None
    dataset = build_dataset(cfg.source, **kwargs)
    dataset = LoggingDataNode(name=profile_name or repr(dataset), source=dataset)
    pipeline = dataset

    profiler = None
    if cfg.profile:
        profiler = PipelineProfiler(
            name=profile_name or repr(dataset), report_freq=cfg.report_freq
        )
        pipeline = ProfilingProxyNode(pipeline, name=repr(dataset), profiler=profiler)
        # Only add the dataset as a root if no transforms will be applied.
        # If transforms exist, the outermost transform will register itself as the root.
        if cfg.transform is None:
            profiler.root_nodes.append(pipeline)

    with scope_profiler(profiler):
        if cfg.transform is not None:
            transform = build_transform(cfg.transform, **kwargs)
            pipeline = transform.build(pipeline)

        # Mark the final node as root for periodic reporting
        if isinstance(pipeline, ProfilingProxyNode):
            pipeline._is_root = True

    assert isinstance(pipeline, torchdata.nodes.BaseNode)
    if (
        hasattr(cfg, "eval_freq")
        or hasattr(cfg, "eval_iterations")
        or hasattr(cfg, "eval_guaranteed_same_batches")
    ):
        return EvalDataPipeline(
            datasets=dataset,
            dataloader=pipeline,
            eval_freq=getattr(cfg, "eval_freq", None),
            eval_iterations=getattr(cfg, "eval_iterations", None),
            eval_guaranteed_same_batches=getattr(
                cfg, "eval_guaranteed_same_batches", None
            ),
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
