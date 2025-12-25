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
    prefetch_factor: int = 8


@register_transform("prefetch", PrefetchTransformConfig)
class PrefetchTransform(BaseTransform):
    def __init__(self, cfg: PrefetchTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def build(self, source: BaseNode) -> BaseNode:
        return torchdata.nodes.Prefetcher(
            source, prefetch_factor=self.cfg.prefetch_factor
        )
