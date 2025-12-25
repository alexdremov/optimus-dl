from dataclasses import dataclass

from omegaconf import MISSING
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfig, RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    build_transform,
    register_transform,
)


@dataclass
class CompositeTransformConfig(RegistryConfigStrict):
    transforms: list[RegistryConfig] = MISSING


@register_transform("compose", CompositeTransformConfig)
class CompositeTransform(BaseTransform):
    def __init__(self, cfg: CompositeTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        transforms = []
        for transform in cfg.transforms:
            transforms.append(build_transform(transform, *args, **kwargs))
        self.transforms = transforms

    def build(self, source: BaseNode) -> BaseNode:
        for transform in self.transforms:
            source = transform.build(source)
        return source
