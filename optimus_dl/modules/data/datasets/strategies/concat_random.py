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
class ConcatAndChunkFullRandomConfig(BaseStrategyConfig):
    seed: int = 42
    chunk_size: int = 2048
    random_offset: bool = True  # Shift global start by random(0, chunk_size)


@register_dataset_sampling_strategy("concat_random", ConcatAndChunkFullRandomConfig)
class ConcatAndChunkFullRandom(BaseStrategy):
    """
    Treats the dataset as a single concatenated stream of tokens, splits it into
    fixed-size chunks, and yields them in a globally random order.

    Algorithm:
    1. Virtual Stream: [Doc0] [Doc1] [Doc2] ...
    2. Apply Global Offset (if random_offset): Skip first K tokens.
    3. Define Chunks: Chunk i = Stream[Offset + i*Size : Offset + (i+1)*Size]
    4. Shuffle Chunks: Permute indices 0..NumChunks.
    5. Partition: Rank r takes indices {p | p % world_size == r}.
    """

    def __init__(self, cfg: ConcatAndChunkFullRandomConfig, rank: int, world_size: int):
        super().__init__(cfg, rank, world_size)
        self.cfg = cfg

        # Schedule state
        self.my_chunk_indices: np.ndarray | None = None
        self.chunk_ptr = 0
        self.global_offset = 0

        # Indexing for seek
        self.cumulative_lengths: np.ndarray | None = None

    def initialize(self, doc_lengths: np.ndarray):
        super().initialize(doc_lengths)
        # We need cumulative lengths to map GlobalTokenIdx -> (DocIdx, Offset)
        # Prepending 0 simplifies the binary search logic:
        # intervals are [cum[i], cum[i+1])
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(doc_lengths)))
        self._setup_schedule()

    def _setup_schedule(self):
        """Create the chunk permutation and assign to this rank."""
        if self.cumulative_lengths is None:
            raise RuntimeError("Strategy not initialized with doc_lengths")

        total_tokens = self.cumulative_lengths[-1]

        # 1. Determine Global Offset
        # We use a seed-based RNG to ensure all ranks agree on the offset
        rng = np.random.default_rng(seed=self.cfg.seed)

        if self.cfg.random_offset:
            # Shift can be anything in [0, chunk_size).
            # This ensures we hit different "phases" of the documents across epochs/seeds.
            self.global_offset = rng.integers(0, self.cfg.chunk_size)
        else:
            self.global_offset = 0

        # 2. Calculate Number of Chunks
        available_tokens = total_tokens - self.global_offset
        if available_tokens <= 0:
            self.my_chunk_indices = np.array([], dtype=np.int64)
            return

        num_chunks = available_tokens // self.cfg.chunk_size

        # 3. Permute Chunks
        # We shuffle the *indices* of the chunks (0 to num_chunks-1)
        perm = rng.permutation(num_chunks)

        # 4. Partition (Stride)
        # Rank k gets perm[k, k+ws, k+2ws, ...]
        self.my_chunk_indices = perm[self.rank :: self.world_size]

    def _get_segments_for_range(
        self, start_token: int, end_token: int
    ) -> list[tuple[int, tuple[int, int]]]:
        """Finds which documents cover the global token range [start, end)."""
        segments = []

        # Binary search to find the document containing 'start_token'
        # searchsorted returns i such that cum[i-1] <= val < cum[i]
        start_doc_idx = (
            np.searchsorted(self.cumulative_lengths, start_token, side="right") - 1
        )

        current_doc_idx = start_doc_idx
        current_global_pos = start_token

        while current_global_pos < end_token:
            if current_doc_idx >= len(self.doc_lengths):
                break

            doc_start_global = self.cumulative_lengths[current_doc_idx]
            doc_end_global = self.cumulative_lengths[current_doc_idx + 1]

            seg_start_global = max(current_global_pos, doc_start_global)
            seg_end_global = min(end_token, doc_end_global)

            if seg_end_global > seg_start_global:
                local_start = seg_start_global - doc_start_global
                local_end = seg_end_global - doc_start_global

                segments.append(
                    (int(current_doc_idx), (int(local_start), int(local_end)))
                )

                current_global_pos = seg_end_global

            current_doc_idx += 1

        return segments

    def next_sample(self) -> list[tuple[int, tuple[int, int]]]:
        if self.my_chunk_indices is None:
            raise RuntimeError("Strategy not initialized")

        if self.chunk_ptr >= len(self.my_chunk_indices):
            raise StopIteration

        # 1. Get the abstract chunk index
        chunk_idx = self.my_chunk_indices[self.chunk_ptr]

        # 2. Map to global token coordinates
        global_start = self.global_offset + (chunk_idx * self.cfg.chunk_size)
        global_end = global_start + self.cfg.chunk_size

        # 3. Retrieve segments
        segments = self._get_segments_for_range(global_start, global_end)

        self.chunk_ptr += 1
        return segments

    def reset(self, initial_state: dict[str, Any] | None = None):
        if initial_state:
            # Restore state
            self.cfg.seed = initial_state.get("seed", self.cfg.seed)
            self.chunk_ptr = initial_state.get("chunk_ptr", 0)

            # Re-run setup to regenerate permutation and offset
            if self.cumulative_lengths is not None:
                self._setup_schedule()

        else:
            self.chunk_ptr = 0
            if self.cumulative_lengths is not None:
                self._setup_schedule()

    def get_state(self) -> dict[str, Any]:
        return {
            "chunk_ptr": self.chunk_ptr,
            "seed": self.cfg.seed,
            "rank": self.rank,
        }
