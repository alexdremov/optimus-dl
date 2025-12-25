from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfigStrict


@dataclass
class DistributedConfig(RegistryConfigStrict):
    tp_size: int = 1
    sharding_world_size: int | None = None
