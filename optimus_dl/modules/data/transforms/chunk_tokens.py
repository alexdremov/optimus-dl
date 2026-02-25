import logging
from dataclasses import dataclass

from omegaconf import MISSING
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkTransformConfig(RegistryConfigStrict):
    """Configuration for chunking token sequences.

    Attributes:
        max_seq_len: Maximum length of each produced chunk.
        add_one_for_shift: If True, adds 1 to max_seq_len (useful for causal LM training).
    """

    max_seq_len: int = MISSING
    add_one_for_shift: bool = True


class ChunkTransformNode(BaseNode):
    """Internal node for performing sequence chunking.

    Maintains a buffer of tokens from the source node and yields segments of
    length `max_seq_len`.
    """

    def __init__(self, node: BaseNode, cfg: ChunkTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.buffer = []

    def reset(self, initial_state: dict | None = None):
        """Restore the buffer and source node state."""
        super().reset(initial_state)
        self.buffer = []
        if initial_state:
            self.buffer = initial_state["buffer"]
            self.cfg = initial_state["cfg"]

            self.node.reset(initial_state["source_state"])
        else:
            self.node.reset()

    def get_state(self):
        """Collect current buffer and source state for checkpointing."""
        return {
            "buffer": self.buffer,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
        }

    def next(self):
        """Yield the next chunk of tokens, refilling the buffer if empty."""
        if len(self.buffer) == 0:
            self.buffer = next(self.node)["input_ids"]

        taken = min(
            self.cfg.max_seq_len + (1 if self.cfg.add_one_for_shift else 0),
            len(self.buffer),
        )
        return_buff = self.buffer[:taken]
        self.buffer = self.buffer[taken:]
        return {"input_ids": return_buff}


@register_transform("chunk_tokens", ChunkTransformConfig)
class ChunkTransform(BaseTransform):
    """Transform that splits variable-length documents into fixed-size chunks.

    Useful when datasets yield full documents that are longer than the desired
    training sequence length.

    Args:
        cfg: Chunking configuration.
    """

    def __init__(self, cfg: ChunkTransformConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        """Apply the chunking transformation to a source node."""
        return ChunkTransformNode(source, self.cfg)
