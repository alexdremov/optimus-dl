from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfig


@dataclass
class DistributedConfig(RegistryConfig):
    use_gpu: bool = True
    tp_size: int = 1
