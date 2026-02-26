from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
        batch_size: Number of sequences per batch.
        pad_token_id: The token ID used to pad sequences to the maximum length.
        field: The dictionary key containing the tokens (defaults to input_ids).
        flatten: If True, yields a single flat sequence of shape (1, sum(lengths)) instead of (B, max_len).
    """

    batch_size: int = MISSING
    pad_token_id: int = MISSING
    field: str = "input_ids"
    flatten: bool = False


class BasicBatcherNode(BaseNode):
    """Internal node for performing dynamic padding and batching.

    Accumulates sequences from the source node up to the specified batch size,
    finds the maximum sequence length within that batch, and pads all sequences
    to match that length. Also records the original unpadded sequence lengths.
    """

    def __init__(self, node: BaseNode, cfg: BasicBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node

    def reset(self, initial_state: dict | None = None):
        """Restore source node state."""
        super().reset(initial_state)
        if initial_state:
            self.cfg = initial_state["cfg"]
            self.node.reset(initial_state["source_state"])
        else:
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        """Collect source state for checkpointing."""
        return {
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
        }

    def next(self) -> dict[str, Any]:
        """Yield the next complete batch of padded tokens and their original lengths."""
        batch_items = []

        # Collect batch_size items from the source node
        try:
            for _ in range(self.cfg.batch_size):
                batch_items.append(next(self.node))
        except StopIteration:
            # Yield whatever we have collected so far as a final, smaller batch
            if not batch_items:
                raise

        # Extract sequence data
        seqs = [item[self.cfg.field] for item in batch_items]

        # Record original lengths before padding
        lengths = [len(seq) for seq in seqs]

        if self.cfg.flatten:
            # Pack all sequences into one flat 1D sequence
            input_ids = np.concatenate(seqs)
            position_ids = np.concatenate([np.arange(length) for length in lengths])
            document_ids = np.concatenate(
                [np.full(length, i) for i, length in enumerate(lengths)]
            )

            return {
                self.cfg.field: input_ids[None, :].astype(np.int64),
                "position_ids": position_ids[None, :].astype(np.int64),
                "document_ids": document_ids[None, :].astype(np.int64),
                "seq_lens": np.array([len(input_ids)], dtype=np.int64),
                "cu_seqlens": np.cumsum([0] + lengths).astype(np.int32),
                "max_seqlen": int(max(lengths)),
            }

        # Determine the maximum sequence length in this specific batch
        max_len = max(lengths)

        # Pad sequences and generate position_ids/document_ids
        padded_seqs = []
        position_ids = []
        document_ids = []

        for i, (seq, seq_len) in enumerate(zip(seqs, lengths, strict=True)):
            pad_len = max_len - seq_len

            # Pad tokens
            if isinstance(seq, torch.Tensor):
                padded_seq = F.pad(seq, (0, pad_len), value=self.cfg.pad_token_id)
            elif isinstance(seq, np.ndarray):
                padded_seq = np.pad(
                    seq, (0, pad_len), constant_values=self.cfg.pad_token_id
                )
            else:
                padded_seq = list(seq) + [self.cfg.pad_token_id] * pad_len
            padded_seqs.append(padded_seq)

            # Generate position IDs (0..seq_len-1 then padded with 0 or -1, using 0 for safety with RoPE)
            pos_ids = np.zeros(max_len, dtype=np.int64)
            pos_ids[:seq_len] = np.arange(seq_len)
            position_ids.append(pos_ids)

            # Generate document IDs (each row is its own document)
            doc_ids = np.full(max_len, i, dtype=np.int64)
            document_ids.append(doc_ids)

        # Build final batched outputs depending on the input type
        if isinstance(padded_seqs[0], torch.Tensor):
            batched_seqs = torch.stack(padded_seqs)
            batched_lens = torch.tensor(lengths, dtype=torch.long)
            batched_pos = torch.from_numpy(np.stack(position_ids))
            batched_docs = torch.from_numpy(np.stack(document_ids))
        else:
            batched_seqs = np.array(padded_seqs, dtype=np.int64)
            batched_lens = np.array(lengths, dtype=np.int64)
            batched_pos = np.stack(position_ids)
            batched_docs = np.stack(document_ids)

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
