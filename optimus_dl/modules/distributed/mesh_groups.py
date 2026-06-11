"""Coordinator for partitioning the distributed cluster into disjoint mesh groups."""

import logging
from dataclasses import (
    dataclass,
    field,
)

import torch.distributed as dist

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.distributed.config import (
    GroupPartitionConfig,
)
from optimus_dl.modules.distributed.mesh import MeshCollective

logger = logging.getLogger(__name__)


@dataclass
class MeshGroupConfig(RegistryConfigStrict):
    """Configuration for cluster partitioning.

    Attributes:
        partitions: List of group partitions.
    """

    partitions: list[GroupPartitionConfig] = field(default_factory=list)


class MeshGroupCoordinator:
    """Orchestrates the partitioning of the global cluster into disjoint groups.

    Allows different models (e.g., Policy and Reference) to run on separate
    sets of GPUs with independent parallelization strategies.
    """

    def __init__(
        self,
        config: MeshGroupConfig,
        global_rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        device_type: str,
    ):
        self.config = config
        self.global_rank = global_rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.device_type = device_type

        self._validate_partitions()

    def _validate_partitions(self):
        """Ensure partitions are disjoint and cover necessary ranks."""
        all_ranks = []
        for p in self.config.partitions:
            all_ranks.extend(p.ranks)

        if len(all_ranks) != len(set(all_ranks)):
            raise ValueError("Mesh group partitions must be disjoint.")

        # It's okay if not all ranks are assigned, but the current rank MUST be.

    def get_my_group_config(self) -> GroupPartitionConfig | None:
        """Find the partition config that the current rank belongs to."""
        for p in self.config.partitions:
            if self.global_rank in p.ranks:
                return p
        return None

    def build_collective(self) -> MeshCollective | None:
        """Build the MeshCollective for the current rank's partition.

        Returns:
            A MeshCollective restricted to the partition's process group,
            or None if the rank is not part of any partition.
        """
        group_cfg = self.get_my_group_config()
        if group_cfg is None:
            logger.warning(f"Rank {self.global_rank} is not part of any mesh group.")
            return None

        # Ensure default process group is initialized (WORLD)
        if not dist.is_initialized():
            backend = "nccl" if self.device_type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)

        # Create the sub-process group for this partition
        new_group = dist.new_group(ranks=group_cfg.ranks)

        # We need to calculate local info relative to the NEW group for MeshCollective
        # BUT MeshCollective currently expects global rank/world_size and then
        # it builds its own meshes.

        # Actually, MeshCollective.init_device_mesh can take a sub-mesh.
        # However, the current MeshCollective implementation is quite rigid.

        # Let's refine MeshCollective to support being "group-aware".
        # For now, we can instantiate a MeshCollective with the new process group.

        # We need to adjust world_size/rank for the sub-group
        group_rank = dist.get_rank(group=new_group)
        group_world_size = dist.get_world_size(group=new_group)

        # We also need to know local_world_size and local_rank WITHIN the group.
        # This is tricky because a group might span partial nodes.
        # For SOTA GRPO, we assume groups are node-aligned for simplicity or
        # we calculate it.

        # For now, let's assume the user provides a clean partitioning.
        # We'll use the global local_rank for device assignment.

        collective = MeshCollective(
            rank=group_rank,
            world_size=group_world_size,
            local_world_size=group_world_size,  # Simplified: assume group is its own 'world'
            local_rank=self.local_rank,
            device_type=self.device_type,
            process_group=new_group,
            tp_size=group_cfg.tp_size,
            sharding_world_size=group_cfg.sharding_world_size,
            mesh_ranks=group_cfg.ranks,
        )

        return collective
