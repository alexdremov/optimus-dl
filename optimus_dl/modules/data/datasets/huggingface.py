import logging
from dataclasses import dataclass

import datasets
import datasets.distributed
from datasets import load_dataset
from omegaconf import MISSING, OmegaConf

from optimus_dl.core.registry import RegistryConfig

from . import register_dataset
from .base import BaseDataset

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceDatasetConfig(RegistryConfig):
    dataset_load_kwargs: dict = MISSING


@register_dataset("huggingface_dataset", HuggingFaceDatasetConfig)
class HuggingFaceDataset(BaseDataset):
    def __init__(self, cfg, rank: int, world_size: int, **kwargs):
        super().__init__(cfg)
        self.rank = rank
        self.world_size = world_size
        self.position = 0

    def get_state(self):
        return {
            "cfg": self.cfg,
            "dataset_state": (
                self.dataset.state_dict()
                if hasattr(self.dataset, "state_dict")
                else None
            ),
            "world_size": self.world_size,
            "rank": self.rank,
            "position": self.position,
        }

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        if initial_state is not None:
            self.cfg = initial_state.get("cfg", self.cfg)
            self.cfg = OmegaConf.merge(
                OmegaConf.structured(HuggingFaceDatasetConfig), self.cfg
            )
            self.position = initial_state["position"]

            assert self.rank == initial_state.get("rank", self.rank)
            assert self.world_size == initial_state.get("world_size", self.world_size)

        if (
            "streaming" in self.cfg.dataset_load_kwargs
            and not self.cfg.dataset_load_kwargs["streaming"]
        ):
            logger.info("streaming=False is not recommended")
        else:
            self.cfg.dataset_load_kwargs["streaming"] = True

        if not self.cfg.dataset_load_kwargs.get("streaming"):
            self.cfg.dataset_load_kwargs.setdefault("num_proc", 4)
        self.dataset = load_dataset(**self.cfg.dataset_load_kwargs)

        logger.info(
            f"Sharding dataset... (num_shards={self.world_size}, index={self.rank})"
        )

        if self.world_size > 1:
            self.dataset = datasets.distributed.split_dataset_by_node(
                dataset=self.dataset,
                rank=self.rank,
                world_size=self.world_size,
            )

        if (
            initial_state is not None
            and "dataset_state" in initial_state
            and initial_state["dataset_state"] is not None
        ):
            self.dataset.load_state_dict(initial_state["dataset_state"])

        if not isinstance(self.dataset, datasets.IterableDataset):
            self.dataset = self.dataset.skip(self.position)
        self.iter = iter(self.dataset)

    def next(self):
        self.position += 1
        return next(self.iter)
