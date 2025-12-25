from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfig


@dataclass
class DistributedConfig(RegistryConfig):
    use_gpu: bool = True
    tp_size: int = 1
    # World size for sharding dimension (typically one node size, e.g., 8 for 8 GPUs per node)
    # Only relevant when use_hybrid_sharding=True
    sharding_world_size: int | None = None
