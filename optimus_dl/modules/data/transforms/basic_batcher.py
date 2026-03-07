from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import MISSING
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)


@dataclass
class BasicBatcherConfig(RegistryConfigStrict):
    """Configuration for basic token batching with dynamic padding.

    Attributes:
        batch_size: Maximum number of sequences per batch. If None, only max_tokens limit is used.
        max_tokens: Maximum total number of tokens per batch. If None, only batch_size limit is used.
        pad_token_id: The token ID used to pad sequences to the maximum length.
        field: The dictionary key containing the tokens (defaults to input_ids).
        flatten: If True, yields a single flat sequence of shape (1, sum(lengths)) instead of (B, max_len).
    """

    batch_size: int | None = None
    max_tokens: int | None = None
    pad_token_id: int = MISSING
    field: str = "input_ids"
    flatten: bool = False


class BasicBatcherNode(BaseNode):
    """Internal node for performing dynamic padding and batching.

    Accumulates sequences from the source node up to the specified batch size
    or token limit, finds the maximum sequence length within that batch,
    and pads all sequences to match that length (or flattens them).
    """

    def __init__(self, node: BaseNode, cfg: BasicBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self._peeked_item = None

        if self.cfg.batch_size is None and self.cfg.max_tokens is None:
            raise ValueError(
                "Either batch_size or max_tokens must be specified in BasicBatcherConfig."
            )

    def reset(self, initial_state: dict | None = None):
        """Restore source node state."""
        super().reset(initial_state)
        self._peeked_item = None
        if initial_state:
            self.cfg = initial_state["cfg"]
            self.node.reset(initial_state["source_state"])
            self._peeked_item = initial_state.get("_peeked_item")
        else:
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        """Collect source state for checkpointing."""
        return {
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
            "_peeked_item": self._peeked_item,
        }

    def next(self) -> dict[str, Any]:
        """Yield the next complete batch of padded tokens and their original lengths."""
        batch_items = []
        lengths = []
        current_tokens = 0
        current_count = 0

        # Collect items from the source node
        while True:
            # Check if we reached batch_size limit
            if self.cfg.batch_size is not None and current_count >= self.cfg.batch_size:
                break

            # Get next item (either peeked or from source)
            try:
                if self._peeked_item is not None:
                    item = self._peeked_item
                    self._peeked_item = None
                else:
                    item = next(self.node)
            except StopIteration:
                if not batch_items:
                    raise
                break

            seq = item[self.cfg.field]
            seq_len = len(seq)

            # Check if we reached max_tokens limit
            if self.cfg.max_tokens is not None:
                # If adding this sequence exceeds max_tokens, but we already have items,
                # save it for next batch and stop.
                if current_tokens + seq_len > self.cfg.max_tokens:
                    if batch_items:
                        self._peeked_item = item
                        break
                    # else: Single item exceeds max_tokens, yield it anyway (standard behavior)

            batch_items.append(item)
            lengths.append(seq_len)
            current_tokens += seq_len
            current_count += 1

        # Extract sequence data
        seqs = [item[self.cfg.field] for item in batch_items]
        batch_size = len(seqs)

        if self.cfg.flatten:
            # Pack all sequences into one flat 1D sequence with causal shifting
            if isinstance(seqs[0], torch.Tensor):
                device = seqs[0].device
                input_ids = torch.cat([s[:-1] for s in seqs])
                labels = torch.cat([s[1:] for s in seqs])

                # Align metadata with input_ids
                shifted_lengths = [length - 1 for length in lengths]
                position_ids = torch.cat(
                    [torch.arange(length, device=device) for length in shifted_lengths]
                )
                document_ids = torch.cat(
                    [
                        torch.full((length,), i, device=device, dtype=torch.long)
                        for i, length in enumerate(shifted_lengths)
                    ]
                )

                return {
                    self.cfg.field: input_ids[None, :],
                    "labels": labels[None, :],
                    "position_ids": position_ids[None, :],
                    "document_ids": document_ids[None, :],
                    "seq_lens": torch.tensor([len(input_ids)], device=device),
                    "cu_seqlens": torch.cumsum(
                        torch.tensor([0] + shifted_lengths, device=device), dim=0
                    ).to(torch.int32),
                    "max_seqlen": int(max(shifted_lengths)),
                }
            else:
                input_ids = np.concatenate([s[:-1] for s in seqs])
                labels = np.concatenate([s[1:] for s in seqs])

                shifted_lengths = [length - 1 for length in lengths]
                position_ids = np.concatenate(
                    [np.arange(length) for length in shifted_lengths]
                )
                document_ids = np.concatenate(
                    [np.full(length, i) for i, length in enumerate(shifted_lengths)]
                )

                return {
                    self.cfg.field: input_ids[None, :].astype(np.int64),
                    "labels": labels[None, :].astype(np.int64),
                    "position_ids": position_ids[None, :].astype(np.int64),
                    "document_ids": document_ids[None, :].astype(np.int64),
                    "seq_lens": np.array([len(input_ids)], dtype=np.int64),
                    "cu_seqlens": np.cumsum([0] + shifted_lengths).astype(np.int32),
                    "max_seqlen": int(max(shifted_lengths)),
                }

        # Determine the maximum sequence length in this specific batch
        max_len = max(lengths)

        # Optimization: Pre-allocate and fill instead of list-and-stack
        if isinstance(seqs[0], torch.Tensor):
            device = seqs[0].device
            dtype = seqs[0].dtype

            batched_seqs = torch.full(
                (batch_size, max_len), self.cfg.pad_token_id, device=device, dtype=dtype
            )
            for i, (seq, seq_len) in enumerate(zip(seqs, lengths, strict=True)):
                batched_seqs[i, :seq_len] = seq

            batched_lens = torch.tensor(lengths, dtype=torch.long, device=device)

            # Vectorized metadata creation
            # position_ids: [0, 1, 2, ..., max_len-1] expanded to (B, max_len)
            batched_pos = (
                torch.arange(max_len, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .clone()
            )
            # Mask out positions beyond seq_len (pad with 0 for safety with RoPE)
            pos_mask = torch.arange(max_len, device=device).unsqueeze(
                0
            ) >= batched_lens.unsqueeze(1)
            batched_pos[pos_mask] = 0

            # document_ids: [0, 0, ...], [1, 1, ...], [B-1, B-1, ...]
            batched_docs = (
                torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_len)
            )

        else:
            # Fallback for NumPy/Lists
            batched_seqs = np.full(
                (batch_size, max_len), self.cfg.pad_token_id, dtype=np.int64
            )
            for i, (seq, seq_len) in enumerate(zip(seqs, lengths, strict=True)):
                batched_seqs[i, :seq_len] = seq

            batched_lens = np.array(lengths, dtype=np.int64)

            # Vectorized metadata creation (NumPy)
            batched_pos = np.tile(np.arange(max_len), (batch_size, 1))
            pos_mask = np.arange(max_len)[None, :] >= batched_lens[:, None]
            batched_pos[pos_mask] = 0

            batched_docs = np.arange(batch_size)[:, None] * np.ones(
                (1, max_len), dtype=np.int64
            )

        return {
            self.cfg.field: batched_seqs,
            "seq_lens": batched_lens,
            "position_ids": batched_pos,
            "document_ids": batched_docs,
        }


@register_transform("basic_batcher", BasicBatcherConfig)
class BasicBatcher(BaseTransform):
    """Transform that aggregates sequences and dynamically pads them.

    Unlike the flat batcher which packs tokens to minimize padding, this
    batcher keeps documents separate and pads shorter documents with a
    designated `pad_token_id` to match the longest document in the batch.
    It yields a dictionary containing the batched items and a `seq_lens`
    tensor recording the actual unpadded lengths of the batch.

    Args:
        cfg: Batching configuration.
    """

    def __init__(self, cfg: BasicBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        """Apply the batching transformation to a source node."""
        return BasicBatcherNode(source, self.cfg)
