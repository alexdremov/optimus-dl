import logging
from dataclasses import dataclass
from typing import Any

import torchdata.nodes
from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfigStrict

from . import build_dataset, register_dataset
from .base import BaseDataset

logger = logging.getLogger(__name__)


@dataclass
class LoopDatasetConfig(RegistryConfigStrict):
    inner: Any = MISSING


@register_dataset("loop", LoopDatasetConfig)
class LoopDataset(BaseDataset):
    """Dataset that infinitely loops over an inner dataset.

    When the inner dataset is exhausted, it is automatically re-initialized,
    creating an endless stream of data. This is useful for training loops where
    the model needs to see the data multiple times.

    Args:
        cfg: Loop dataset configuration.
        rank: Distributed rank for sharding.
        world_size: Total number of ranks.
    """

    def __init__(self, cfg: LoopDatasetConfig, rank: int, world_size: int, **kwargs):
        super().__init__(cfg)
        self.rank = rank
        self.world_size = world_size
        self.kwargs = kwargs

        self.inner_dataset: torchdata.nodes.BaseNode | None = None

    def _build_inner(self):
        """Build the inner dataset from configuration."""
        self.inner_dataset = build_dataset(
            self.cfg.inner, rank=self.rank, world_size=self.world_size, **self.kwargs
        )

    def next(self):
        """Yield the next item from the inner dataset, resetting it if exhausted."""
        if self.inner_dataset is None:
            raise ValueError("Inner dataset not initialized")

        try:
            return self.inner_dataset.next()
        except StopIteration:
            logger.info("Inner dataset exhausted, recreating loop...")
            self._build_inner()
            return self.inner_dataset.next()

    def reset(self, initial_state: dict | None = None):
        """Reset or restore the loop dataset state.

        Args:
            initial_state: Optional state dictionary for resuming.
        """
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
        """Return the current state for checkpointing."""
        state = {
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if self.inner_dataset:
            state["inner_state"] = self.inner_dataset.state_dict()
        return state
