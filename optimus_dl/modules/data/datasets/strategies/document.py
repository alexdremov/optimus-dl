from dataclasses import dataclass
from typing import (
    Any,
)

import numpy as np

from . import register_dataset_sampling_strategy
from .base import (
    BaseStrategy,
    BaseStrategyConfig,
)


@dataclass
class DocumentStrategyConfig(BaseStrategyConfig):
    shuffle: bool = False
    seed: int = 42


@register_dataset_sampling_strategy("document", DocumentStrategyConfig)
class DocumentStrategy(BaseStrategy):
    """
    Yields full documents from the dataset.
    Supports both sequential and random ordering via shuffling.
    """

    def __init__(self, cfg: DocumentStrategyConfig, rank: int, world_size: int):
        super().__init__(cfg, rank, world_size)
        self.cfg = cfg
        self.indices: np.ndarray | None = None
        self.ptr = 0

    def initialize(self, doc_lengths: np.ndarray):
        super().initialize(doc_lengths)
        self._setup_indices()

    def _setup_indices(self):
        total_docs = len(self.doc_lengths)

        if self.cfg.shuffle:
            rng = np.random.default_rng(seed=self.cfg.seed)
            perm = rng.permutation(total_docs)
            # Use striding for uniform distribution across ranks when shuffled
            self.indices = perm[self.rank :: self.world_size]
        else:
            # Use contiguous blocks for sequential mode
            docs_per_rank = total_docs // self.world_size
            start = docs_per_rank * self.rank
            end = (
                docs_per_rank * (self.rank + 1)
                if self.rank < self.world_size - 1
                else total_docs
            )
            self.indices = np.arange(start, end, dtype=np.int64)

    def next_sample(self) -> list[tuple[int, tuple[int, int]]]:
        if self.indices is None:
            raise RuntimeError("Strategy not initialized")

        if self.ptr >= len(self.indices):
            raise StopIteration

        doc_idx = self.indices[self.ptr]
        doc_len = self.doc_lengths[doc_idx]

        self.ptr += 1
        return [(int(doc_idx), (0, int(doc_len)))]

    def reset(self, initial_state: dict[str, Any] | None = None):
        if self.doc_lengths is not None:
            self._setup_indices()

        if initial_state:
            self.ptr = initial_state.get("ptr", 0)
        else:
            self.ptr = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "ptr": self.ptr,
            "rank": self.rank,
        }
