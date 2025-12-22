"""
Handles writing tokenized documents into sized-shards on disk
and creating the final index file.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import OutputConfig, ProcessingConfig

logger = logging.getLogger(__name__)


class Sharder:
    """
    Manages the creation of numpy shards from tokenized documents.

    This class accumulates documents, writes them to shard files when a size
    threshold is reached, and generates a final index file. It also supports
    state management for checkpointing.
    """

    def __init__(self, output_config: OutputConfig, proc_config: ProcessingConfig):
        self.output_dir = Path(output_config.dir)
        self.output_name = output_config.name
        self.max_shard_bytes = proc_config.shard_size_mb * 1024 * 1024

        # Determine dtype for tokens
        self.dtype = np.uint16 if proc_config.dtype == "uint16" else np.uint32

        # Internal state for the current shard
        self.current_shard_tokens: list[int] = []
        self.current_shard_doc_lens: list[int] = []
        self.current_shard_size_bytes = 0

        # Overall state
        self.shard_idx = 0
        self.file_metadata: list[dict[str, Any]] = []
        self.total_tokens = 0

    def get_state(self) -> dict[str, Any]:
        """Returns the sharder's current state for checkpointing."""
        return {
            "shard_idx": self.shard_idx,
            "file_metadata": self.file_metadata,
            "total_tokens": self.total_tokens,
            "current_shard_tokens": self.current_shard_tokens,
            "current_shard_doc_lens": self.current_shard_doc_lens,
        }

    def load_state(self, state: dict[str, Any]):
        """Restores the sharder's state from a checkpoint."""
        self.shard_idx = state.get("shard_idx", 0)
        self.file_metadata = state.get("file_metadata", [])
        self.total_tokens = state.get("total_tokens", 0)
        self.current_shard_tokens = state.get("current_shard_tokens", [])
        self.current_shard_doc_lens = state.get("current_shard_doc_lens", [])

        itemsize = np.dtype(self.dtype).itemsize
        self.current_shard_size_bytes = len(self.current_shard_tokens) * itemsize

    def add(self, doc_tokens: list[int]) -> bool:
        """
        Adds a document to the current shard. Flushes the shard if it exceeds
        the size limit.

        Args:
            doc_tokens: A list of integers representing the tokenized document.

        Returns:
            True if a shard was flushed, False otherwise.
        """
        doc_len = len(doc_tokens)
        doc_bytes = doc_len * np.dtype(self.dtype).itemsize

        if (
            self.current_shard_size_bytes + doc_bytes > self.max_shard_bytes
            and self.current_shard_tokens
        ):
            self.flush()
            # After flushing, the new doc becomes the first in the new shard
            self.current_shard_tokens.extend(doc_tokens)
            self.current_shard_doc_lens.append(doc_len)
            self.current_shard_size_bytes += doc_bytes
            return True

        # Default case: add to current shard
        self.current_shard_tokens.extend(doc_tokens)
        self.current_shard_doc_lens.append(doc_len)
        self.current_shard_size_bytes += doc_bytes
        return False

    def flush(self):
        """
        Writes the accumulated tokens and document lengths to numpy shard files.
        """
        if not self.current_shard_tokens:
            return

        shard_name = f"{self.output_name}_{self.shard_idx:010d}.npy"
        shard_path = self.output_dir / shard_name
        token_arr = np.array(self.current_shard_tokens, dtype=self.dtype)
        np.save(shard_path, token_arr)

        lens_name = f"{self.output_name}_{self.shard_idx:010d}_lens.npy"
        lens_path = self.output_dir / lens_name
        lens_arr = np.array(self.current_shard_doc_lens, dtype=np.uint32)
        np.save(lens_path, lens_arr)

        num_tokens_in_shard = len(token_arr)
        num_docs_in_shard = len(lens_arr)

        metadata = {
            "file": shard_name,
            "lens_file": lens_name,
            "num_tokens": num_tokens_in_shard,
            "num_docs": num_docs_in_shard,
            "shard_idx": self.shard_idx,
        }
        self.file_metadata.append(metadata)
        self.total_tokens += num_tokens_in_shard

        logger.debug(
            f"Saved shard {shard_name} ({num_tokens_in_shard} tokens, {num_docs_in_shard} docs)"
        )

        # Reset current shard state
        self.shard_idx += 1
        self.current_shard_tokens = []
        self.current_shard_doc_lens = []
        self.current_shard_size_bytes = 0

    def finalize(self, final_config: dict[str, Any]):
        """
        Flushes any remaining data to a final shard and writes the main index file.
        """
        self.flush()  # Flush any remaining tokens

        index_data = {
            "files": self.file_metadata,
            "total_tokens": self.total_tokens,
            "config": final_config,
        }

        index_path = self.output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Done! Total tokens: {self.total_tokens:,}")
        logger.info(f"Metadata saved to {index_path}")
