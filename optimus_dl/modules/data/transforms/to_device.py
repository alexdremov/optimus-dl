import logging
from dataclasses import dataclass, field

import torch
import torchdata.nodes
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ToDeviceTransformConfig(RegistryConfigStrict):
    """Configuration for device transfers.

    Attributes:
        properties: List of dictionary keys to move to the device. If None,
            moves all values in the dictionary.
    """

    properties: list[str] | None = field(default_factory=lambda: None)


@register_transform("to_device", ToDeviceTransformConfig)
class ToDeviceTransform(BaseTransform):
    """Transform that moves data tensors to the target compute device.

    This transform ensures that input data is on the correct device (e.g., CUDA)
    before it enters the model. It includes performance optimizations for GPUs:

    - **Memory Pinning**: Automatically uses `PinMemory` to speed up CPU-to-GPU transfers.
    - **Asynchronous Transfers**: Uses `non_blocking=True` for CUDA devices.
    - **Prefetching**: Adds an additional prefetch layer after pinning.

    Args:
        cfg: Device transfer configuration.
        device: The target PyTorch device.
    """

    def __init__(self, cfg: ToDeviceTransformConfig, device, **kwargs):
        super().__init__(**kwargs)
        self.properties = cfg.properties
        self.device = device

        assert isinstance(device, torch.device)

    def _map(self, sample: dict):
        """Map function to move specific dictionary entries to the device."""
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
        """Wrap the source node with pinning, prefetching, and the device map."""
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
