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
    """

    batch_size: int = MISSING
    seq_len: int = MISSING
    worker_cfg: MapperConfig = field(
        default_factory=MapperConfig,
    )
    field: str = "input_ids"
    add_one_for_shift: bool = True


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
        if initial_state:
            self.buffer = initial_state["buffer"]
            self.cfg = initial_state["cfg"]
            self.node.reset(initial_state["source_state"])
        else:
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        """Collect current buffer and source state for checkpointing."""
        return {
            "buffer": self.buffer,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
        }

    def next(self) -> Any:
        """Yield the next complete batch of tokens, filling from source as needed."""
        while len(self.buffer) < self.target_size:
            self.buffer.extend(next(self.node)[self.cfg.field])

        return_buff = self.buffer[: self.target_size]
        self.buffer = self.buffer[self.target_size :]
        return {
            "input_ids": np.array(return_buff, dtype=np.int64).reshape(
                self.cfg.batch_size, -1
            )
        }


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
