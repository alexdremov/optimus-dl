from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import numpy as np
from omegaconf import MISSING
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    MapperConfig,
    register_transform,
)


@dataclass
class FlatTokensBatcherConfig(RegistryConfigStrict):
    """Configuration for token aggregation and batching.

    Attributes:
        batch_size: Number of sequences per batch.
        seq_len: Sequence length for each sample.
        worker_cfg: Configuration for map workers (not used directly by batcher).
        field: The dictionary key containing the tokens (defaults to input_ids).
        add_one_for_shift: If True, yields seq_len + 1 tokens per sample.
        mask_documents: If True, tracks document boundaries and emits document_ids/position_ids.
        flatten: If True, yields a single flat sequence of shape (1, B*T) instead of (B, T).
    """

    batch_size: int = MISSING
    seq_len: int = MISSING
    worker_cfg: MapperConfig = field(
        default_factory=MapperConfig,
    )
    field: str = "input_ids"
    add_one_for_shift: bool = True
    mask_documents: bool = False
    flatten: bool = False


class FlatTokensBatcherNode(BaseNode):
    """Internal node for performing token aggregation and batching.

    Accumulates tokens from variable-length document sources into a buffer
    until it has enough to form a complete batch of the target size.
    """

    def __init__(self, node: BaseNode, cfg: FlatTokensBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.buffer = []
        self.position_ids_buffer = []
        self.document_ids_buffer = []
        self._current_doc_id = 0

    @property
    def target_size(self):
        """Calculate total number of tokens needed for one batch."""
        return self.cfg.batch_size * (
            self.cfg.seq_len + (1 if self.cfg.add_one_for_shift else 0)
        )

    def reset(self, initial_state: dict | None = None):
        """Restore batcher buffer and source node state."""
        super().reset(initial_state)
        self.buffer = []
        self.position_ids_buffer = []
        self.document_ids_buffer = []
        self._current_doc_id = 0
        if initial_state:
            self.buffer = initial_state["buffer"]
            self.position_ids_buffer = initial_state.get("position_ids_buffer", [])
            self.document_ids_buffer = initial_state.get("document_ids_buffer", [])
            self._current_doc_id = initial_state.get("_current_doc_id", 0)
            self.cfg = initial_state["cfg"]
            self.node.reset(initial_state["source_state"])
        else:
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        """Collect current buffer and source state for checkpointing."""
        return {
            "buffer": self.buffer,
            "position_ids_buffer": self.position_ids_buffer,
            "document_ids_buffer": self.document_ids_buffer,
            "_current_doc_id": self._current_doc_id,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
        }

    def next(self) -> Any:
        """Yield the next complete batch of tokens, filling from source as needed."""
        while len(self.buffer) < self.target_size:
            item = next(self.node)
            tokens = item[self.cfg.field]
            self.buffer.extend(tokens)
            if self.cfg.mask_documents:
                self.position_ids_buffer.extend(range(len(tokens)))
                self.document_ids_buffer.extend([self._current_doc_id] * len(tokens))
                self._current_doc_id += 1

        return_buff = self.buffer[: self.target_size]
        self.buffer = self.buffer[self.target_size :]

        if self.cfg.flatten:
            reshape_args = (1, -1)
        else:
            reshape_args = (self.cfg.batch_size, -1)

        output = {
            "input_ids": np.array(return_buff, dtype=np.int64).reshape(*reshape_args)
        }

        if self.cfg.mask_documents:
            pos_buff = self.position_ids_buffer[: self.target_size]
            doc_buff = self.document_ids_buffer[: self.target_size]
            self.position_ids_buffer = self.position_ids_buffer[self.target_size :]
            self.document_ids_buffer = self.document_ids_buffer[self.target_size :]

            # Re-base document IDs to avoid huge numbers and overflow
            doc_ids = np.array(doc_buff, dtype=np.int64)
            _, doc_ids = np.unique(doc_ids, return_inverse=True)

            output["position_ids"] = np.array(pos_buff, dtype=np.int64).reshape(
                *reshape_args
            )
            output["document_ids"] = doc_ids.reshape(*reshape_args)

            if self.cfg.flatten:
                # Compute cumulative sequence lengths for the flat batch
                # Find indices where document ID changes
                flat_doc_ids = doc_ids.reshape(-1)
                diff = np.diff(flat_doc_ids, prepend=-1)
                change_indices = np.where(diff != 0)[0]
                cu_seqlens = np.append(change_indices, len(flat_doc_ids)).astype(
                    np.int32
                )
                output["cu_seqlens"] = cu_seqlens
                output["max_seqlen"] = int(np.max(np.diff(cu_seqlens)))

        return output


@register_transform("flat_batcher", FlatTokensBatcherConfig)
class FlatTokensBatcher(BaseTransform):
    """Transform that aggregates token IDs and yields fixed-size batches.

    Unlike standard batchers that batch whole examples, this batcher pools all
    tokens from incoming documents and yields packed sequences, minimizing
    padding.

    Args:
        cfg: Batching configuration.
    """

    def __init__(self, cfg: FlatTokensBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        """Apply the batching transformation to a source node."""
        return FlatTokensBatcherNode(source, self.cfg)
