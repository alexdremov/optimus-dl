from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import MISSING
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    MapperConfig,
    register_transform,
)


@dataclass
class FlatTokensBatcherConfig(RegistryConfig):
    batch_size: int = MISSING
    seq_len: int = MISSING
    worker_cfg: MapperConfig = field(
        default_factory=MapperConfig,
    )
    field: str = "input_ids"
    add_one_for_shift: bool = True


class FlatTokensBatcherNode(BaseNode):
    def __init__(self, node: BaseNode, cfg: FlatTokensBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.buffer = []

    @property
    def target_size(self):
        return self.cfg.batch_size * (
            self.cfg.seq_len + (1 if self.cfg.add_one_for_shift else 0)
        )

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        self.buffer = []
        if initial_state:
            self.buffer = initial_state["buffer"]
            self.cfg = initial_state["cfg"]
            self.node.reset(initial_state["source_state"])
        else:
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
        }

    def next(self) -> Any:
        while len(self.buffer) < self.target_size:
            self.buffer.extend(self.node.next()[self.cfg.field])

        return_buff = self.buffer[: self.target_size]
        self.buffer = self.buffer[self.target_size :]
        return {
            "input_ids": np.array(return_buff, dtype=np.int64).reshape(
                self.cfg.batch_size, -1
            )
        }


@register_transform("flat_batcher", FlatTokensBatcherConfig)
class FlatTokensBatcher(BaseTransform):
    def __init__(self, cfg: FlatTokensBatcherConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        return FlatTokensBatcherNode(source, self.cfg)
