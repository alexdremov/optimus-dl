from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfigStrict


@dataclass
class DistributedConfig(RegistryConfigStrict):
    """Configuration for distributed training topologies.

    Attributes:
        tp_size: Degree of Tensor Parallelism (number of GPUs to shard each layer across).
        sharding_world_size: Size of FSDP sharding groups. If None, defaults to
            the number of GPUs per node (intra-node sharding).
    """

    tp_size: int = 1
    sharding_world_size: int | None = None
