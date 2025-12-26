import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfigStrict

from . import register_dataset
from .base import BaseDataset

logger = logging.getLogger(__name__)


@dataclass
class TokenizedDatasetConfig(RegistryConfigStrict):
    """Configuration for pre-tokenized sharded datasets.

    Attributes:
        data_dir: Path to the directory containing shards and index file.
        index_file: Name of the JSON index file (defaults to index.json).
        limit: Optional maximum number of documents to read.
    """

    data_dir: str = MISSING
    index_file: str = "index.json"
    limit: int | None = None  # Optional limit on number of documents


@register_dataset("tokenized_dataset", TokenizedDatasetConfig)
class TokenizedDataset(BaseDataset):
    """Dataset that streams full tokenized documents from numpy shards.

    This dataset expects data prepared by `scripts/prepare_data.py`, consisting
    of multiple `.npy` shards and a global `index.json`. It provides:
    - **Memory Mapping**: Efficiently reads shards using `mmap_mode="r"`.
    - **Document Partitioning**: Automatically shards the document list across
      distributed ranks.
    - **Precise Seeking**: Can jump to any document index globally for resuming.

    Yields:
        Dictionary: {"input_ids": np.array([...]), "document_id": int}
    """

    def __init__(
        self, cfg: TokenizedDatasetConfig, rank: int, world_size: int, **kwargs
    ):
        super().__init__(cfg)
        self.data_dir = Path(cfg.data_dir)
        self.index_file = cfg.index_file
        self.rank = rank
        self.world_size = world_size
        self.limit = cfg.limit

        # Internal State
        self.shards = []
        self.shard_num_docs = []  # Number of documents per shard
        self.total_docs = 0

        # Pointers
        self.global_doc_idx = 0
        self.start_doc_idx = 0
        self.end_doc_idx = 0

        # Current Shard State
        self.current_shard_idx = -1
        self.current_shard_tokens: np.ndarray | None = None
        self.current_shard_doc_lens: np.ndarray | None = None
        self.shard_token_offset = 0
        self.shard_doc_offset = 0  # Index of document WITHIN the current shard

    def _resolve_dtype(self, type_str: str):
        """Map string dtype names to numpy dtypes."""
        dtype_map = {
            "np.uint8": np.uint8,
            "np.uint16": np.uint16,
            "np.uint32": np.uint32,
            "np.int32": np.int32,
            "np.int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "int32": np.int32,
            "int64": np.int64,
        }
        return dtype_map.get(type_str, np.uint16)

    def _load_index(self):
        """Load metadata and calculate rank-specific document boundaries."""
        index_path = self.data_dir / self.index_file
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path) as f:
            data = json.load(f)

        self.dtype = self._resolve_dtype(data["config"]["dtype"])

        files_meta = data.get("files", [])
        files_meta.sort(key=lambda x: x["shard_idx"])

        self.shards = []
        self.shard_num_docs = []
        self.total_docs = 0

        for meta in files_meta:
            token_file = self.data_dir / meta["file"]
            lens_file = self.data_dir / meta["lens_file"]
            num_docs = meta.get("num_docs", 0)

            if not token_file.exists():
                raise FileNotFoundError(f"Token file not found: {token_file}")

            if not lens_file.exists():
                raise FileNotFoundError(f"Lens file not found: {lens_file}")

            self.shards.append((token_file, lens_file))
            self.shard_num_docs.append(num_docs)
            self.total_docs += num_docs

        if self.limit is not None:
            self.total_docs = min(self.total_docs, self.limit)

        # Calculate Rank Boundaries (Partition by Documents)
        docs_per_rank = self.total_docs // self.world_size
        self.start_doc_idx = docs_per_rank * self.rank
        self.end_doc_idx = docs_per_rank * (self.rank + 1)

        if self.rank == self.world_size - 1:
            self.end_doc_idx = self.total_docs

    def _seek(self, global_doc_idx: int):
        """Seek to a specific global document index.

        Finds which shard contains the document and sets internal offsets.
        """
        # Validate bounds
        if not (self.start_doc_idx <= global_doc_idx <= self.end_doc_idx):
            # If we are exactly at the end, it's allowed (means finished)
            if global_doc_idx == self.end_doc_idx:
                self.global_doc_idx = global_doc_idx
                return

            logger.warning(
                f"Seeking to {global_doc_idx} which is outside rank bounds "
                f"[{self.start_doc_idx}, {self.end_doc_idx}]. "
                f"Clamping or correcting might be needed if world_size changed."
            )
            # We enforce strict bounds for now
            global_doc_idx = max(
                self.start_doc_idx, min(global_doc_idx, self.end_doc_idx)
            )

        self.global_doc_idx = global_doc_idx

        if self.global_doc_idx >= self.end_doc_idx:
            # Done
            self.current_shard_idx = len(self.shards)
            return

        # Find the shard containing global_doc_idx
        cumulative_docs = 0
        found_shard = False

        for i, count in enumerate(self.shard_num_docs):
            if cumulative_docs + count > global_doc_idx:
                self.current_shard_idx = i
                self.shard_doc_offset = global_doc_idx - cumulative_docs
                found_shard = True
                break
            cumulative_docs += count

        if not found_shard:
            # Should be covered by boundary check, but safety fallback
            raise RuntimeError(
                f"Could not find shard for document index {global_doc_idx}"
            )

        # Load the shard and calculate token offset
        self._load_current_shard(init_doc_offset=self.shard_doc_offset)

    def _load_current_shard(self, init_doc_offset: int = 0):
        """Memory-map the current shard files."""
        if self.current_shard_idx >= len(self.shards):
            self.current_shard_tokens = None
            self.current_shard_doc_lens = None
            return

        token_path, lens_path = self.shards[self.current_shard_idx]

        # Load full shard into memory
        self.current_shard_tokens = np.load(token_path, mmap_mode="r")
        self.current_shard_doc_lens = np.load(lens_path, mmap_mode="r")

        self.shard_doc_offset = init_doc_offset

        # Calculate token offset for the starting document
        if init_doc_offset > 0:
            self.shard_token_offset = np.sum(
                self.current_shard_doc_lens[:init_doc_offset]
            )
        else:
            self.shard_token_offset = 0

    def reset(self, initial_state: dict[str, Any] | None = None):
        """Restore dataset state or start from the rank's assigned beginning."""
        super().reset(initial_state)

        target_doc_idx = None

        if initial_state:
            assert self.rank == initial_state.get("rank", self.rank)
            assert self.world_size == initial_state.get("world_size", self.world_size)
            target_doc_idx = initial_state.get("global_doc_idx")

        # Load metadata and calculate boundaries for (potentially new) rank
        self._load_index()

        if target_doc_idx is None:
            target_doc_idx = self.start_doc_idx

        # Seek to the correct position
        self._seek(target_doc_idx)

    def next(self):
        """Yield the next tokenized document."""
        # Check global limit
        if self.global_doc_idx >= self.end_doc_idx:
            raise StopIteration

        # Check if we need to move to next shard
        if self.current_shard_doc_lens is None or self.shard_doc_offset >= len(
            self.current_shard_doc_lens
        ):
            self.current_shard_idx += 1
            # Reset offsets for new shard
            self._load_current_shard(init_doc_offset=0)

            if self.current_shard_tokens is None:
                raise StopIteration

        # Get current document length
        doc_len = self.current_shard_doc_lens[self.shard_doc_offset]

        # Extract tokens
        start = self.shard_token_offset
        end = start + doc_len

        # Handle potential edge case where lens file doesn't match token file
        if end > len(self.current_shard_tokens):
            logger.error(
                f"Shard {self.current_shard_idx} mismatch: expected {end} tokens, got {len(self.current_shard_tokens)}"
            )
            raise RuntimeError("Data corruption: lens file does not match token file.")

        tokens = self.current_shard_tokens[start:end]

        # Prepare output
        item = {
            "input_ids": np.array(tokens, dtype=self.dtype),  # Ensure copy/array
            "document_id": self.global_doc_idx,
        }

        # Advance pointers
        self.shard_token_offset = end
        self.shard_doc_offset += 1
        self.global_doc_idx += 1

        return item

    def get_state(self):
        """Return the current global document index for checkpointing."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "global_doc_idx": self.global_doc_idx,
        }
