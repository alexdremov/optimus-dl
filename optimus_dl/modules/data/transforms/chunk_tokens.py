import logging
from dataclasses import dataclass

from omegaconf import MISSING
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkTransformConfig(RegistryConfig):
    max_seq_len: int = MISSING
    add_one_for_shift: bool = True


class ChunkTransformNode(BaseNode):
    def __init__(self, node: BaseNode, cfg: ChunkTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.buffer = []

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        self.buffer = []
        if initial_state:
            self.buffer = initial_state["buffer"]
            self.cfg = initial_state["cfg"]

            self.node.reset(initial_state["source_state"])
        else:
            self.node.reset()

    def get_state(self):
        return {
            "buffer": self.buffer,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
        }

    def next(self):
        if not self.buffer:
            self.buffer = self.node.next()["input_ids"]

        taken = min(
            self.cfg.max_seq_len + (1 if self.cfg.add_one_for_shift else 0),
            len(self.buffer),
        )
        return_buff = self.buffer[:taken]
        self.buffer = self.buffer[taken:]
        return {"input_ids": return_buff}


@register_transform("chunk_tokens", ChunkTransformConfig)
class ChunkTransform(BaseTransform):
    def __init__(self, cfg: ChunkTransformConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        return ChunkTransformNode(source, self.cfg)
