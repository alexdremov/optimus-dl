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
    buffer_size: int = 1024
    seed: int = 42


class ShuffleTransformNode(BaseNode):
    def __init__(
        self, node: BaseNode, cfg: ShuffleTransformConfig, rank: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node
        self.buffer = []
        self.terminated = False
        self.rank = rank

        self.rng = np.random.default_rng(cfg.seed + rank * 41)

    def reset(self, initial_state: dict | None = None):
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
        return {
            "buffer": self.buffer,
            "cfg": self.cfg,
            "source_state": self.node.state_dict(),
            "rng_state": self.rng.bit_generator.state,
            "terminated": self.terminated,
            "rank": self.rank,
        }

    def next(self):
        while len(self.buffer) < self.cfg.buffer_size and not self.terminated:
            try:
                self.buffer.append(self.node.next())
            except StopIteration:
                self.terminated = True
                break
        if len(self.buffer) == 0:
            raise StopIteration
        return self.buffer.pop(self.rng.integers(0, len(self.buffer)))


@register_transform("shuffle", ShuffleTransformConfig)
class ShuffleTransform(BaseTransform):
    def __init__(self, cfg: ShuffleTransformConfig, rank: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.rank = rank

    def build(self, source: BaseNode) -> BaseNode:
        return ShuffleTransformNode(source, self.cfg, rank=self.rank)
