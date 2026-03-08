import logging
from dataclasses import (
    dataclass,
    field,
)

import numpy as np
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
    pin_memory: bool = True
    pin_prefetch_factor: int = 2
    pin_snapshot_frequency: int = 128


@register_transform("to_device", ToDeviceTransformConfig)
class ToDeviceTransform(BaseTransform):
    """Transform that moves data tensors to the target compute device.

    This transform ensures that input data is on the correct device (e.g., CUDA)
    before it enters the model. It includes performance optimizations for GPUs:

    - **Memory Pinning**: Automatically uses `PinMemory` to speed up CPU-to-GPU transfers.
    - **Asynchronous Transfers**: Uses `non_blocking=True` for CUDA devices.
    - **Prefetching**: Adds an additional prefetch layer to overlap device transfers with computation.

    Args:
        cfg: Device transfer configuration.
        device: The target PyTorch device.
    """

    def __init__(self, cfg: ToDeviceTransformConfig, device, **kwargs):
        super().__init__(**kwargs)
        self.properties = cfg.properties
        self.device = device
        self.pin_memory = cfg.pin_memory
        self.pin_prefetch_factor = cfg.pin_prefetch_factor
        self.pin_snapshot_frequency = cfg.pin_snapshot_frequency

        assert isinstance(device, torch.device)

    def _map(self, sample: dict):
        """Map function to move specific dictionary entries to the device."""
        if self.properties is None:
            properties = sample.keys()
        else:
            properties = self.properties

        for property in properties:
            val = sample[property]
            # Skip non-numeric values (like strings) if we are in "all values" mode
            if self.properties is None:
                if not (
                    torch.is_tensor(val)
                    or isinstance(val, (np.ndarray, int, float, list))
                ):
                    continue
                # If it's a list, check if it's numeric (at least the first element)
                if isinstance(val, list) and len(val) > 0:
                    if not isinstance(val[0], (int, float, np.number)):
                        continue

            if self.device.type != "cuda":
                value = torch.as_tensor(val, device=self.device)
            else:
                # For CUDA, we expect memory to be pinned for maximum async performance
                value = torch.as_tensor(val)
                value = value.to(self.device, non_blocking=True)
            sample[property] = value
        return sample

    def build(self, source: BaseNode) -> BaseNode:
        """Wrap the source node with pinning, prefetching, and the device map."""
        if self.device.type == "cuda" and self.pin_memory:
            # 1. Pin CPU memory
            source = torchdata.nodes.PinMemory(
                source=source,
                pin_memory_device="cuda",
                snapshot_frequency=self.pin_snapshot_frequency,
            )
            # 2. Map to device (starts async transfer)
            source = torchdata.nodes.Mapper(
                source=source,
                map_fn=self._map,
            )
            # 3. Prefetch the async transfers so they overlap with training
            if self.pin_prefetch_factor > 0:
                source = torchdata.nodes.Prefetcher(
                    source=source,
                    prefetch_factor=self.pin_prefetch_factor,
                    snapshot_frequency=self.pin_snapshot_frequency,
                )
            return source

        # For non-CUDA or no-pinning, just apply the map
        return torchdata.nodes.Mapper(
            source=source,
            map_fn=self._map,
        )
