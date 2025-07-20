import logging
from dataclasses import dataclass
from typing import Any, Dict

import torchdata.nodes
from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfig

from . import build_dataset, register_dataset

logger = logging.getLogger(__name__)


@dataclass
class LoopDatasetConfig(RegistryConfig):
    inner: Any = MISSING


@register_dataset("loop", LoopDatasetConfig)
class LoopDataset(torchdata.nodes.BaseNode):
    def __init__(self, cfg: LoopDatasetConfig, rank: int, world_size: int, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.kwargs = kwargs

        self.inner_dataset: torchdata.nodes.BaseNode | None = None

    def _build_inner(self):
        self.inner_dataset = build_dataset(
            self.cfg.inner, rank=self.rank, world_size=self.world_size, **self.kwargs
        )

    def next(self):
        if self.inner_dataset is None:
            raise ValueError("Inner dataset not initialized")

        try:
            return self.inner_dataset.next()
        except StopIteration:
            logger.info("Inner dataset exhausted, recreating loop...")
            self._build_inner()
            return self.inner_dataset.next()

    def reset(self, initial_state: Dict | None = None):
        super().reset(initial_state)

        inner_state = None
        if initial_state is not None:
            self.rank = initial_state.get("rank", self.rank)
            self.world_size = initial_state.get("world_size", self.world_size)
            inner_state = initial_state.get("inner_state")

        if self.inner_dataset is None:
            self._build_inner()

        assert self.inner_dataset is not None, "Inner dataset not initialized"
        self.inner_dataset.reset(inner_state)

    def get_state(self):
        state = {
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if self.inner_dataset:
            state["inner_state"] = self.inner_dataset.get_state()
        return state
