from dataclasses import dataclass
from typing import (
    Any,
)

from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)


@dataclass
class SkipInterleavedTransformConfig(RegistryConfigStrict):
    """Configuration for skip_interleaved.

    Attributes:
        skip_count: Number of items to skip before producing a new one. First item is always produced.
    """

    skip_count: int = 1


class SkipInterleavedTransformNonde(BaseNode):
    def __init__(
        self, node: BaseNode, cfg: SkipInterleavedTransformConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.node = node

        self.counter = 0

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        if initial_state:
            self.counter = initial_state.get("counter", 0)
            self.node.reset(initial_state.get("source_state"))
        else:
            self.counter = 0
            self.node.reset()

    def get_state(self) -> dict[str, Any]:
        return {
            "source_state": self.node.state_dict(),
            "counter": self.counter,
        }

    def next(self):
        if self.counter > 0:
            for _ in range(self.cfg.skip_count):
                next(self.node)
                self.counter += 1

        # Fetch the actual item we want to produce
        value = next(self.node)
        self.counter += 1

        return value


@register_transform("skip_interleaved", SkipInterleavedTransformConfig)
class SkipInterleavedTransform(BaseTransform):
    """Transform that pre-fetches data items in a background thread.

    This helps hide data loading and transformation latency by keeping a buffer
    of items ready for the training loop.

    Args:
        cfg: Prefetching configuration.
    """

    def __init__(self, cfg: SkipInterleavedTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        """Wrap the source node with a Prefetcher."""
        return SkipInterleavedTransformNonde(source, self.cfg)
