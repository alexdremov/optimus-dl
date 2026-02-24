import json
import logging
from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import MISSING

from optimus_dl.core.registry import (
    RegistryConfig,
    RegistryConfigStrict,
)

from . import register_dataset
from .base import BaseDataset
from .strategies import build_dataset_sampling_strategy
from .strategies.document import DocumentStrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class TokenizedDatasetConfig(RegistryConfigStrict):
    """Configuration for pre-tokenized sharded datasets.

    Attributes:
        data_dir: Path to the directory containing shards and index file.
        index_file: Name of the JSON index file (defaults to index.json).
        limit: Optional maximum number of documents to read.
        strategy: Sampling strategy configuration.
    """

    data_dir: str = MISSING
    index_file: str = "index.json"
    limit: int | None = None  # Optional limit on number of documents
    strategy: RegistryConfig = field(
        default_factory=lambda: DocumentStrategyConfig(_name="document")
    )


@register_dataset("tokenized_dataset", TokenizedDatasetConfig)
class TokenizedDataset(BaseDataset):
    """Dataset that streams full tokenized documents from numpy shards.

    This dataset expects data prepared by `scripts/prepare_data.py`, consisting
    of multiple `.npy` shards and a global `index.json`. It provides:

    - **Memory Mapping**: Efficiently reads shards using `mmap_mode="r"`.
    - **Pluggable Strategies**: Supports different sampling strategies (sequential, random chunking, etc.).
    - **Precise Seeking**: Can jump to any document index globally for resuming.

    Yields:
        Dictionary: {"input_ids": np.array([...]), "document_id": int | np.array}
    """

    def __init__(
        self,
        cfg: TokenizedDatasetConfig,
        rank: int,
        world_size: int,
        seed: int,
        **kwargs,
    ):
        super().__init__(cfg)
        self.data_dir = Path(cfg.data_dir)
        self.index_file = cfg.index_file
        self.rank = rank
        self.world_size = world_size
        self.limit = cfg.limit

        # Internal State
        self.shards = []
        self.shard_num_docs = []
        self.total_docs = 0
        self.doc_lengths: np.ndarray | None = None
        self.doc_to_shard_map: np.ndarray | None = None

        # Strategy
        self.strategy = build_dataset_sampling_strategy(
            cfg.strategy,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )

        # Current Shard State
        self.current_shard_idx = -1
        self.current_shard_tokens: np.ndarray | None = None
        self.current_shard_doc_lens: np.ndarray | None = None
        self.shard_doc_start_idx = 0  # Global doc index where current shard starts

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
        all_lengths = []
        self.total_docs = 0

        # Pre-load lengths and build shard map
        shard_indices = []

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

            # Load lengths (cast to int64 to prevent overflow)
            shard_lens = np.load(lens_file, mmap_mode="r").astype(np.int64)
            all_lengths.append(shard_lens)

            # Create mapping: [shard_idx] * num_docs
            shard_indices.append(
                np.full(num_docs, len(self.shards) - 1, dtype=np.int32)
            )

            self.total_docs += num_docs

        # Concatenate all lengths
        if all_lengths:
            self.doc_lengths = np.concatenate(all_lengths)
            self.doc_to_shard_map = np.concatenate(shard_indices)
        else:
            self.doc_lengths = np.array([], dtype=np.int64)
            self.doc_to_shard_map = np.array([], dtype=np.int32)

        # Apply limit
        if self.limit is not None:
            self.total_docs = min(self.total_docs, self.limit)
            self.doc_lengths = self.doc_lengths[: self.limit]
            self.doc_to_shard_map = self.doc_to_shard_map[: self.limit]

        # Initialize strategy
        self.strategy.initialize(self.doc_lengths)

    def _load_shard_for_doc(self, doc_idx: int):
        """Ensure the shard containing doc_idx is loaded."""
        shard_idx = self.doc_to_shard_map[doc_idx]

        if shard_idx != self.current_shard_idx:
            # Load new shard
            token_path, lens_path = self.shards[shard_idx]
            self.current_shard_tokens = np.load(token_path, mmap_mode="r")
            self.current_shard_doc_lens = np.load(lens_path, mmap_mode="r")
            self.current_shard_idx = shard_idx

            # Calculate where this shard starts in global doc indices
            count = 0
            for i in range(shard_idx):
                count += self.shard_num_docs[i]
            self.shard_doc_start_idx = count

    def _fetch_segment(self, doc_idx: int, start: int, end: int) -> np.ndarray:
        """Fetch a specific segment of tokens from a document."""
        self._load_shard_for_doc(doc_idx)

        # Local document index within the shard
        local_doc_idx = doc_idx - self.shard_doc_start_idx

        # Cache cumulative offsets for the current shard to enable O(1) lookups
        if (
            not hasattr(self, "_current_shard_offsets")
            or self._current_shard_offsets_shard_idx != self.current_shard_idx
        ):
            self._current_shard_offsets = np.concatenate(
                ([0], np.cumsum(self.current_shard_doc_lens))
            ).astype(np.int64)
            self._current_shard_offsets_shard_idx = self.current_shard_idx

        doc_start_token = self._current_shard_offsets[local_doc_idx]

        # Extract
        abs_start = int(doc_start_token) + start
        abs_end = int(doc_start_token) + end

        if abs_end > len(self.current_shard_tokens):
            logger.error(
                f"Shard {self.current_shard_idx} mismatch: expected end {abs_end} > len {len(self.current_shard_tokens)}"
            )
            raise RuntimeError("Data corruption: lens file does not match token file.")

        return self.current_shard_tokens[abs_start:abs_end]

    def reset(self, initial_state: dict[str, Any] | None = None):
        """Restore dataset state."""
        super().reset(initial_state)

        # Reload index and lengths
        self._load_index()

        # Pass state to strategy
        strategy_state = initial_state.get("strategy_state") if initial_state else None
        self.strategy.reset(strategy_state)

    def next(self):
        """Yield the next sample."""
        try:
            segments = self.strategy.next_sample()
        except StopIteration:
            raise

        if not segments:
            raise StopIteration

        # Collect data
        data_parts = []
        seq_lens = []
        doc_ids_parts = []

        is_multi_doc = len(segments) > 1

        for doc_idx, (start, end) in segments:
            part = self._fetch_segment(doc_idx, start, end)
            data_parts.append(part)
            length = len(part)
            seq_lens.append(length)

            if is_multi_doc:
                doc_ids_parts.append(np.full(length, doc_idx, dtype=np.int64))

        if len(data_parts) == 1:
            input_ids = data_parts[0]
            document_id = segments[0][0]
        else:
            input_ids = np.concatenate(data_parts)
            document_id = np.concatenate(doc_ids_parts)

        # Ensure correct dtype
        if input_ids.dtype != self.dtype:
            input_ids = input_ids.astype(self.dtype)

        item = {
            "input_ids": input_ids,
            "document_id": document_id,
            "seq_lens": np.array(seq_lens, dtype=np.int32),
        }

        return item

    def get_state(self):
        """Return state for checkpointing."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "strategy_state": self.strategy.get_state(),
        }
