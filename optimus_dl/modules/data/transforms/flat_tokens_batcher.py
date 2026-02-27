from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import numpy as np
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
        batch_size: Number of sequences per batch. Required if max_tokens is None.
        seq_len: Sequence length for each sample. Required if max_tokens is None.
        max_tokens: Total number of tokens per batch. If provided, overrides batch_size * seq_len.
        worker_cfg: Configuration for map workers (not used directly by batcher).
        field: The dictionary key containing the tokens (defaults to input_ids).
        mask_documents: If True, tracks document boundaries and emits document_ids/position_ids.
        flatten: If True, yields a single flat sequence of shape (1, sum_T) instead of (B, T).
    """

    batch_size: int | None = None
    seq_len: int | None = None
    max_tokens: int | None = None
    worker_cfg: MapperConfig = field(
        default_factory=MapperConfig,
    )
    field: str = "input_ids"
    mask_documents: bool = False
    flatten: bool = False


class FlatTokensBatcherNode(BaseNode):
    """Internal node for performing token aggregation and batching.

    Accumulates pre-shifted segments from variable-length document sources
    into buffers until it has enough to form a complete batch of the target size.
    This ensures that document transitions are excluded from the sequence.
    """

    def __init__(self, node: BaseNode, cfg: FlatTokensBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.input_buffer = []
        self.label_buffer = []
        self.position_ids_buffer = []
        self.document_ids_buffer = []
        self._current_doc_id = 0

        # Configuration Validation
        if self.cfg.flatten:
            if not self.cfg.mask_documents:
                raise ValueError(
                    "FlatTokensBatcher: 'mask_documents' must be True when 'flatten' is True. "
                    "Flat batches require document tracking to generate 'cu_seqlens' and 'position_ids' for sequence isolation."
                )
            if self.cfg.max_tokens is None and (
                self.cfg.batch_size is None or self.cfg.seq_len is None
            ):
                raise ValueError(
                    "FlatTokensBatcher (flatten=True) requires either 'max_tokens' or both 'batch_size' and 'seq_len'."
                )
        else:
            if self.cfg.batch_size is None or self.cfg.seq_len is None:
                raise ValueError(
                    "FlatTokensBatcher (flatten=False) requires 'batch_size' and 'seq_len' to define the batch layout."
                )

    @property
    def target_size(self):
        """Calculate total number of tokens needed for one batch of inputs."""
        if self.cfg.max_tokens is not None:
            return self.cfg.max_tokens

        return self.cfg.batch_size * self.cfg.seq_len

    def reset(self, initial_state: dict | None = None):
        """Restore batcher buffer and source node state."""
        super().reset(initial_state)
        self.input_buffer = []
        self.label_buffer = []
        self.position_ids_buffer = []
        self.document_ids_buffer = []
        self._current_doc_id = 0
        if initial_state:
            self.input_buffer = initial_state.get("input_buffer", [])
            self.label_buffer = initial_state.get("label_buffer", [])
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
            "input_buffer": self.input_buffer,
            "label_buffer": self.label_buffer,
            "position_ids_buffer": self.position_ids_buffer,
            "document_ids_buffer": self.document_ids_buffer,
            "_current_doc_id": self._current_doc_id,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
        }

    def next(self) -> Any:
        """Yield the next complete batch of tokens, filling from source as needed."""
        # Fill buffers with pre-shifted segments
        while len(self.input_buffer) < self.target_size:
            item = next(self.node)
            tokens = item[self.cfg.field]
            if len(tokens) <= 1:
                continue

            self.input_buffer.extend(tokens[:-1])
            self.label_buffer.extend(tokens[1:])

            if self.cfg.mask_documents:
                self.position_ids_buffer.extend(range(len(tokens) - 1))
                self.document_ids_buffer.extend(
                    [self._current_doc_id] * (len(tokens) - 1)
                )
                self._current_doc_id += 1

        # Extract segments
        input_tokens = np.array(self.input_buffer[: self.target_size], dtype=np.int64)
        target_tokens = np.array(self.label_buffer[: self.target_size], dtype=np.int64)

        self.input_buffer = self.input_buffer[self.target_size :]
        self.label_buffer = self.label_buffer[self.target_size :]

        if self.cfg.flatten:
            reshape_args = (1, -1)
        else:
            reshape_args = (self.cfg.batch_size, -1)

        output = {
            "input_ids": input_tokens.reshape(*reshape_args),
            "labels": target_tokens.reshape(*reshape_args),
        }

        if self.cfg.mask_documents:
            pos_in = np.array(
                self.position_ids_buffer[: self.target_size], dtype=np.int64
            )
            doc_in = np.array(
                self.document_ids_buffer[: self.target_size], dtype=np.int64
            )
            self.position_ids_buffer = self.position_ids_buffer[self.target_size :]
            self.document_ids_buffer = self.document_ids_buffer[self.target_size :]

            # Re-base document IDs
            _, doc_ids = np.unique(doc_in, return_inverse=True)

            output["position_ids"] = pos_in.reshape(*reshape_args)
            output["document_ids"] = doc_ids.reshape(*reshape_args)

            if self.cfg.flatten:
                # Compute cu_seqlens for the flat batch
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
