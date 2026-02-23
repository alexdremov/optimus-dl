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
    """

    batch_size: int = MISSING
    pad_token_id: int = 0
    field: str = "input_ids"


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
                batch_items.append(self.node.next())
        except StopIteration:
            # Yield whatever we have collected so far as a final, smaller batch
            if not batch_items:
                raise

        # Extract sequence data
        seqs = [item[self.cfg.field] for item in batch_items]

        # Record original lengths before padding
        lengths = [len(seq) for seq in seqs]

        # Determine the maximum sequence length in this specific batch
        max_len = max(lengths)

        # Pad sequences
        padded_seqs = []
        for seq, seq_len in zip(seqs, lengths, strict=True):
            pad_len = max_len - seq_len
            if pad_len > 0:
                if isinstance(seq, torch.Tensor):
                    # F.pad takes padding sizes starting from the last dimension: (pad_left, pad_right)
                    padded_seq = F.pad(seq, (0, pad_len), value=self.cfg.pad_token_id)
                elif isinstance(seq, np.ndarray):
                    padded_seq = np.pad(
                        seq, (0, pad_len), constant_values=self.cfg.pad_token_id
                    )
                else:
                    padded_seq = list(seq) + [self.cfg.pad_token_id] * pad_len
            else:
                padded_seq = seq

            padded_seqs.append(padded_seq)

        # Build final batched outputs depending on the input type
        if isinstance(padded_seqs[0], torch.Tensor):
            batched_seqs = torch.stack(padded_seqs)
            batched_lens = torch.tensor(lengths, dtype=torch.long)
        else:
            batched_seqs = np.array(padded_seqs, dtype=np.int64)
            batched_lens = np.array(lengths, dtype=np.int64)

        return {self.cfg.field: batched_seqs, "input_lens": batched_lens}


@register_transform("basic_batcher", BasicBatcherConfig)
class BasicBatcher(BaseTransform):
    """Transform that aggregates sequences and dynamically pads them.

    Unlike the flat batcher which packs tokens to minimize padding, this
    batcher keeps documents separate and pads shorter documents with a
    designated `pad_token_id` to match the longest document in the batch.
    It yields a dictionary containing the batched items and an `input_lens`
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
