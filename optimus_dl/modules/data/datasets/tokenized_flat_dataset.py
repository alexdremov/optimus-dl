import math
from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfig

from . import register_dataset
from .base import BaseDataset


@dataclass
class TokenizedFlatDatasetConfig(RegistryConfig):
    dtype: str = "np.uint16"
    files: list[str] = field(
        default_factory=list,
    )
    seq_len: int = field(default=MISSING)
    batch_size: int = field(default=MISSING)


@register_dataset("tokenized_flat", TokenizedFlatDatasetConfig)
class TokenizedFlatDataset(BaseDataset):
    def __init__(
        self, cfg: TokenizedFlatDatasetConfig, rank: int, world_size: int, **kwargs
    ):
        super().__init__(cfg)
        self.files = cfg.files
        self.seq_len = cfg.seq_len
        self.batch_size = cfg.batch_size
        self.dtype = cfg.dtype
        self.rank = rank
        self.world_size = world_size

    def _remap_files(self):
        # Safe dtype conversion without eval()
        dtype_map = {
            "np.uint8": np.uint8,
            "np.uint16": np.uint16,
            "np.uint32": np.uint32,
            "np.int8": np.int8,
            "np.int16": np.int16,
            "np.int32": np.int32,
            "np.float32": np.float32,
            "np.float64": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
        }

        dtype = dtype_map.get(self.dtype, np.uint16)
        if self.dtype not in dtype_map:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Unknown dtype '{self.dtype}', defaulting to np.uint16")

        self.files_mapped = [np.memmap(i, dtype=dtype, mode="r") for i in self.files]
        self.cumlens = np.cumsum([len(i) for i in self.files_mapped])

        total_tokens = self.cumlens[-1]
        tokens_per_rank = math.floor(total_tokens / self.world_size)

        self.index = tokens_per_rank * (self.rank)
        self.limit = tokens_per_rank * (self.rank + 1)

    @property
    def file_index(self):
        if self.index >= self.cumlens[-1]:
            return None
        return np.min(np.arange(len(self.files_mapped))[self.index < self.cumlens])

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)

        initial_state = initial_state or {}

        old_files = self.files
        self.files = initial_state.get("files", self.files)
        self.seq_len = initial_state.get("seq_len", self.seq_len)
        self.batch_size = initial_state.get("batch_size", self.batch_size)

        assert initial_state.get("world_size", self.world_size) == self.world_size
        assert initial_state.get("rank", self.rank) == self.rank

        if self.files != old_files or not hasattr(self, "index"):
            self._remap_files()

        self.index = initial_state.get("index", self.index)
        self.limit = initial_state.get("limit", self.limit)

    def get_state(self):
        return {
            "files": self.files,
            "index": self.index,
            "limit": self.limit,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def _take_at_most(self, size):
        file_index = self.file_index
        if file_index is None:
            raise StopIteration

        infile_index = self.index
        if file_index > 0:
            infile_index = self.index - self.cumlens[file_index - 1]

        to_take = min(size, len(self.files_mapped[file_index]) - infile_index)
        chunk = self.files_mapped[file_index][infile_index : infile_index + to_take]
        self.index += to_take
        return chunk

    def next(self):
        target_size = self.batch_size * self.seq_len
        result = self._take_at_most(target_size)

        while len(result) != target_size:
            left = target_size - len(result)
            result = np.concatenate((result, self._take_at_most(left)))

        if self.index > self.limit:
            raise StopIteration

        return {"input_ids": result.reshape(self.batch_size, self.seq_len)}
