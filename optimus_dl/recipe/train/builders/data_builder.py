"""Data builder mixin for building data pipelines."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torchdata.nodes

from optimus_dl.core.registry import (
    RegistryConfig,
    make_registry,
)
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
    """Configuration for DataBuilder."""

    pass


class DataBuilder:
    """Builder class for constructing training and evaluation data pipelines.

    Manages the creation of `DataPipeline` objects, ensuring correct distributed
    sharding and iterator behavior (e.g., infinite loop for training, resettable
    for evaluation).

    Args:
        cfg: Builder configuration.
        data_config: Configuration for datasets and transforms.
    """

    def __init__(
        self, cfg: DataBuilderConfig, data_config: DataConfig, data_seed: int, **kwargs
    ):
        self.data_config = data_config
        self.data_seed = data_seed

    @staticmethod
    def _get_rank_seed(seed: int, rank: int, world_size: int) -> int:
        """
        Generate a unique seed for each rank based on the base seed, rank, and epoch.
        """
        rng = torch.Generator()
        # Mix seed, epoch, and rank deterministically
        rng.manual_seed(seed + world_size * 10000 + rank)
        return torch.randint(0, 2**32, (1,), generator=rng).item()

    def build_train_data(self, collective: Collective, **kwargs) -> DataPipeline | None:
        """Build the training data pipeline.

        Automatically injects rank and world_size for sharding. The resulting
        loader is configured to restart automatically on StopIteration, creating
        an infinite stream.

        Args:
            collective: Distributed collective for sharding info.
            **kwargs: Additional arguments passed to dataset builders.

        Returns:
            A DataPipeline containing the dataset and loader.
        """
        kwargs["rank"] = collective.dp_rank
        kwargs["world_size"] = collective.dp_world_size
        kwargs["seed"] = self._get_rank_seed(
            self.data_seed, collective.dp_rank, collective.dp_world_size
        )
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
        self, collective: Collective, **kwargs: Any
    ) -> dict[str, DataPipeline | None]:
        """Build evaluation data pipelines.

        Constructs a dictionary of pipelines for multiple evaluation datasets.
        Uses `LoaderIterResettable` to allow repeated iteration over the same
        validation sets.

        Args:
            collective: Distributed collective.
            **kwargs: Additional arguments.

        Returns:
            Dictionary mapping dataset names to DataPipelines.
        """
        kwargs["rank"] = collective.dp_rank
        kwargs["world_size"] = collective.dp_world_size
        kwargs["seed"] = self._get_rank_seed(
            self.data_seed, collective.dp_rank, collective.dp_world_size
        )
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
    """A Loader that automatically resets its iterator on `__iter__`.

    This is essential for evaluation loops where the dataloader is re-used
    multiple times.
    """

    def __init__(self, root, restart_on_stop_iteration: bool = True):
        super().__init__(root=root, restart_on_stop_iteration=restart_on_stop_iteration)

    def __iter__(self):
        """Reset the iterator state and return a new iterator."""
        iter = super().__iter__()
        iter.reset()
        return iter


_, register_data_builder, build_data_builder = make_registry(
    "data_builder", DataBuilder
)
register_data_builder("base", DataBuilderConfig)(DataBuilder)
