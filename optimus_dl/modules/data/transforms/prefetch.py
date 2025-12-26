from dataclasses import dataclass

import torchdata.nodes
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)


@dataclass
class PrefetchTransformConfig(RegistryConfigStrict):
    """Configuration for prefetching.

    Attributes:
        prefetch_factor: Number of items to fetch ahead of request.
    """

    prefetch_factor: int = 8


@register_transform("prefetch", PrefetchTransformConfig)
class PrefetchTransform(BaseTransform):
    """Transform that pre-fetches data items in a background thread.

    This helps hide data loading and transformation latency by keeping a buffer
    of items ready for the training loop.

    Args:
        cfg: Prefetching configuration.
    """

    def __init__(self, cfg: PrefetchTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        """Wrap the source node with a Prefetcher."""
        return torchdata.nodes.Prefetcher(
            source, prefetch_factor=self.cfg.prefetch_factor
        )
