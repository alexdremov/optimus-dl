"""Device and distributed setup utilities.

This module provides functions for automatically detecting and setting up the best
available compute device (CUDA, MPS, XPU, or CPU) and initializing distributed
training collectives.
"""

from typing import Any, NamedTuple

import torch

from optimus_dl.modules.distributed.config import DistributedConfig


class DeviceSetup(NamedTuple):
    """Container for device and collective setup results.

    Attributes:
        device: The PyTorch device to use for computation.
        collective: The distributed collective object for multi-GPU/multi-node training.
    """

    device: torch.device
    collective: Any


def get_best_device() -> torch.device:
    """Detect and return the best available compute device.

    Checks for available devices in order of preference:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs)
    3. XPU (Intel GPUs)
    4. CPU (fallback)

    Returns:
        The best available torch.device. Always returns a valid device,
        defaulting to CPU if no accelerators are available.

    Example:
        ```python
        device = get_best_device()
        print(device)  # cuda, mps, xpu, or cpu
        ```
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def setup_device_and_collective(
    use_gpu: bool, config: DistributedConfig
) -> DeviceSetup:
    """Setup compute device and distributed training collective.

    This function initializes the training environment by:
    1. Selecting the appropriate compute device (GPU or CPU)
    2. Setting up distributed communication if multiple devices are available
    3. Returning both the device and collective for use in training

    Args:
        use_gpu: If True, attempts to use GPU if available. If False, uses CPU.
        config: Distributed configuration specifying how to set up multi-GPU mesh.

    Returns:
        DeviceSetup namedtuple containing:

        - device: The PyTorch device to use for computation
        - collective: Distributed collective object for multi-GPU coordination

    Example:
        ```python
        from optimus_dl.modules.distributed.config import DistributedConfig
        config = DistributedConfig()
        setup = setup_device_and_collective(use_gpu=True, config=config)
        model = model.to(setup.device)
        # Use setup.collective for distributed operations
        ```
    """
    from optimus_dl.modules.distributed import build_best_collective

    device = torch.device("cpu")
    if use_gpu:
        device = get_best_device()
    collective = build_best_collective(config=config, device=device)
    device = collective.default_device
    return DeviceSetup(device=device, collective=collective)
