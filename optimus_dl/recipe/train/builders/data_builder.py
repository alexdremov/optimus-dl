"""Data builder mixin for building data pipelines."""

import logging
from dataclasses import dataclass

import torchdata.nodes

from optimus_dl.core.registry import RegistryConfig, make_registry
from optimus_dl.modules.data import (
    DataConfig,
    DataPipeline,
    build_data_pipeline,
    build_data_pipeline_dict,
)
from optimus_dl.modules.distributed.base import Collective

logger = logging.getLogger(__name__)


@dataclass
class DataBuilderConfig(RegistryConfig):
    pass


class DataBuilder:
    """Mixin for building data pipelines."""

    def __init__(self, cfg: DataBuilderConfig, data_config: DataConfig, **kwargs):
        self.data_config = data_config

    def build_train_data(self, collective: Collective, **kwargs) -> DataPipeline | None:
        """Build training data pipeline."""
        kwargs["rank"] = collective.dp_rank
        kwargs["world_size"] = collective.dp_world_size
        train_data = build_data_pipeline(self.data_config.train_datasets, **kwargs)
        if train_data is None:
            return None
        dataloader = torchdata.nodes.Loader(
            root=train_data.dataloader,
            restart_on_stop_iteration=True,
        )
        return DataPipeline(
            datasets=train_data.datasets,
            dataloader=dataloader,
        )

    def build_eval_data(
        self, collective: Collective, **kwargs
    ) -> dict[str, DataPipeline | None]:
        """Build evaluation data pipelines."""
        kwargs["rank"] = collective.dp_rank
        kwargs["world_size"] = collective.dp_world_size
        eval_data = build_data_pipeline_dict(self.data_config.eval_datasets, **kwargs)
        eval_data = {
            k: (
                DataPipeline(
                    datasets=v.datasets,
                    dataloader=LoaderIterResettable(
                        root=v.dataloader,
                        restart_on_stop_iteration=False,
                    ),
                )
                if v is not None
                else None
            )
            for k, v in eval_data.items()
        }
        return eval_data


class LoaderIterResettable(torchdata.nodes.Loader):
    def __init__(self, root, restart_on_stop_iteration: bool = True):
        super().__init__(root=root, restart_on_stop_iteration=restart_on_stop_iteration)

    def __iter__(self):
        iter = super().__iter__()
        iter.reset()
        return iter


_, register_data_builder, build_data_builder = make_registry(
    "data_builder", DataBuilder
)
register_data_builder("base", DataBuilderConfig)(DataBuilder)
