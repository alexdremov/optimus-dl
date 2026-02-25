import logging
from dataclasses import dataclass

import numpy as np
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ShuffleTransformConfig(RegistryConfigStrict):
    """Configuration for data shuffling.

    Attributes:
        buffer_size: Number of items to hold in the shuffling buffer. Larger
            buffers provide better shuffling but use more memory.
    """

    buffer_size: int = 1024


class ShuffleTransformNode(BaseNode):
    """Internal node for performing buffer-based shuffling.

    Fills an internal buffer from the source node and yields items selected
    randomly from that buffer.
    """

    def __init__(
        self,
        node: BaseNode,
        cfg: ShuffleTransformConfig,
        rank: int,
        seed: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.buffer = []
        self.terminated = False
        self.rank = rank

        self.rng = np.random.default_rng(seed + rank * 41)

    def reset(self, initial_state: dict | None = None):
        """Restore the shuffle buffer and RNG state."""
        super().reset(initial_state)
        self.buffer = []
        self.terminated = False
        if initial_state:
            self.buffer = initial_state["buffer"]
            self.cfg = initial_state["cfg"]
            self.rng.bit_generator.state = initial_state["rng_state"]
            self.terminated = initial_state["terminated"]

            assert self.rank == initial_state["rank"]

            self.node.reset(initial_state["source_state"])
        else:
            self.node.reset()

    def get_state(self):
        """Collect current buffer, terminated flag, and RNG state for checkpointing."""
        return {
            "buffer": self.buffer,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
            "rng_state": self.rng.bit_generator.state,
            "terminated": self.terminated,
            "rank": self.rank,
        }

    def next(self):
        """Yield a randomly selected item from the shuffle buffer."""
        while len(self.buffer) < self.cfg.buffer_size and not self.terminated:
            try:
                self.buffer.append(next(self.node))
            except StopIteration:
                self.terminated = True
                break
        if len(self.buffer) == 0:
            raise StopIteration
        return self.buffer.pop(self.rng.integers(0, len(self.buffer)))


@register_transform("shuffle", ShuffleTransformConfig)
class ShuffleTransform(BaseTransform):
    """Transform that shuffles data items using an internal buffer.

    Ensures that items are yielded in a randomized order within a sliding window
    of `buffer_size`. Seed is automatically adjusted per rank to ensure variety
    in distributed training.

    Args:
        cfg: Shuffling configuration.
        rank: Distributed rank.
    """

    def __init__(self, cfg: ShuffleTransformConfig, rank: int, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.rank = rank
        self.seed = seed

    def build(self, source: BaseNode) -> BaseNode:
        """Apply the shuffling transformation to a source node."""
        return ShuffleTransformNode(source, self.cfg, rank=self.rank, seed=self.seed)
