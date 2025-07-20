import logging
from dataclasses import dataclass, field

import torch
import torchdata.nodes
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.device import get_best_device
from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ToDeviceTransformConfig(RegistryConfig):
    properties: list[str] | None = field(default_factory=lambda: None)


@register_transform("to_device", ToDeviceTransformConfig)
class ToDeviceTransform(BaseTransform):
    def __init__(self, cfg: ToDeviceTransformConfig, device, **kwargs):
        super().__init__(**kwargs)
        self.properties = cfg.properties
        self.device = device

        assert isinstance(device, torch.device)

    def _map(self, sample: dict):
        if self.properties is None:
            properties = sample.keys()
        else:
            properties = self.properties

        for property in properties:
            if self.device.type != "cuda":
                value = torch.as_tensor(sample[property], device=self.device)
            else:
                value = torch.as_tensor(sample[property])
                value = value.to(self.device, non_blocking=True)
            sample[property] = value
        return sample

    def build(self, source: BaseNode) -> BaseNode:
        if self.device.type == "cuda":
            source = torchdata.nodes.PinMemory(
                source=source,
                pin_memory_device="cuda",
            )
            source = torchdata.nodes.Prefetcher(
                source=source,
                prefetch_factor=2,
            )
        return torchdata.nodes.Mapper(
            source=source,
            map_fn=self._map,
        )
