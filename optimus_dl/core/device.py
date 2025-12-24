from typing import Any, NamedTuple

import torch


class DeviceSetup(NamedTuple):
    device: torch.device
    collective: Any


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def setup_device_and_collective(use_gpu: bool, tp_size: int) -> DeviceSetup:
    """Setup device and distributed collective.

    Args:
        use_gpu: Whether to use GPU if available

    Returns:
        DeviceSetup namedtuple containing device and collective
    """
    from optimus_dl.modules.distributed import build_best_collective

    device = torch.device("cpu")
    if use_gpu:
        device = get_best_device()
    collective = build_best_collective(device=device, tp_size=tp_size)
    device = collective.default_device
    return DeviceSetup(device=device, collective=collective)
