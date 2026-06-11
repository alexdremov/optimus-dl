from dataclasses import (
    dataclass,
    field,
)

from optimus_dl.core.registry import RegistryConfigStrict


@dataclass
class GroupPartitionConfig(RegistryConfigStrict):
    """Configuration for a single mesh group partition.

    Attributes:
        name: Name of the group (e.g., 'actor', 'reference').
        ranks: List of global ranks belonging to this group.
        tp_size: Tensor parallelism degree for this group.
        sharding_world_size: FSDP sharding degree for this group.
    """

    name: str = ""
    ranks: list[int] = field(default_factory=list)
    tp_size: int = 1
    sharding_world_size: int | None = 1


@dataclass
class DistributedConfig(RegistryConfigStrict):
    """Configuration for distributed training topologies.

    Attributes:
        tp_size: Degree of Tensor Parallelism (number of GPUs to shard each layer across).
        sharding_world_size: Size of FSDP sharding groups. If None, defaults to
            the number of GPUs per node (intra-node sharding).
        partitions: Optional list of group partitions for cluster decoupling (GRPO).
    """

    tp_size: int = 1
    sharding_world_size: int | None = 1
    partitions: list[GroupPartitionConfig] | None = field(default=None)
