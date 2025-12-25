import logging
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp, init_process_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from typing_extensions import override

from optimus_dl.modules.distributed.base import Collective

logger = logging.getLogger(__name__)


class Meshes(NamedTuple):
    parallel_mesh: DeviceMesh
    physical_mesh: DeviceMesh


class MeshCollective(Collective):
    def __init__(
        self,
        rank,
        world_size,
        local_world_size,
        local_rank,
        device_type,
        mesh: Meshes | None = None,
        process_group=None,
        tp_size: int = 1,
        sharding_world_size: int | None = None,
    ) -> None:
        super().__init__(rank, world_size)
        assert world_size % local_world_size == 0

        self._device_type = device_type
        self._local_rank = local_rank
        self._local_world_size = local_world_size
        self._tp_size = tp_size

        if mesh is None:
            if world_size % tp_size != 0:
                raise ValueError(
                    f"World size ({world_size}) must be divisible by tp_size ({tp_size})"
                )

            # Determine mesh structure
            # Goal: (dp_replicate, dp_shard, tp)
            # tp_size: innermost dim
            # dp_shard: intra-node data parallelism
            # dp_replicate: inter-node data parallelism

            if local_world_size % tp_size != 0:
                raise ValueError(
                    f"Local world size ({local_world_size}) must be divisible by tp_size ({tp_size})."
                )

            sharding_world_size = sharding_world_size or (local_world_size // tp_size)
            if world_size % (sharding_world_size * tp_size) != 0:
                raise ValueError(
                    f"World size ({world_size}) must be divisible by ({sharding_world_size * tp_size = })"
                )

            # Standard Case: TP is within node.
            # Calculate dimension sizes
            replicate_size = world_size // (sharding_world_size * tp_size)
            shard_size = sharding_world_size

            mesh_dims = (replicate_size, shard_size, tp_size)
            mesh_names = ("dp_replicate", "dp_shard", "tp")

            assert (
                replicate_size * shard_size * tp_size == world_size
            ), "Invalid mesh dimensions"

            mesh_device_type = "cpu"
            if device_type == "cuda":
                mesh_device_type = "cuda"
            if device_type == "mps":
                logger.warning("MPS distributed training uses cpu collective")
                mesh_device_type = "cpu"

            if not dist.is_initialized():
                backend = "nccl" if mesh_device_type == "cuda" else "gloo"
                logger.info(f"Initializing default PG with {backend = }")
                device_id = None
                if mesh_device_type == "cuda":
                    torch.cuda.set_device(local_rank)
                    device_id = torch.device(f"cuda:{local_rank}")
                init_process_group(
                    backend=backend,
                    rank=rank,
                    world_size=world_size,
                    device_id=device_id,
                )

            logger.info(
                f"Initializing mesh with {mesh_device_type=}, shape={mesh_dims}, names={mesh_names}"
            )
            parallel_mesh = init_device_mesh(
                device_type=mesh_device_type,
                mesh_shape=mesh_dims,
                mesh_dim_names=mesh_names,
            )
            physical_mesh = init_device_mesh(
                device_type=mesh_device_type,
                mesh_shape=(world_size // local_world_size, local_world_size),
                mesh_dim_names=("nodes", "local_ranks"),
            )
            mesh = Meshes(parallel_mesh, physical_mesh)

        self._mesh: Meshes = mesh
        self._process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )

    def __repr__(self) -> str:
        group_size = (
            dist.get_world_size(group=self._process_group)
            if self._process_group
            else "unknown"
        )
        group_rank = (
            dist.get_rank(group=self._process_group)
            if self._process_group
            else "unknown"
        )
        is_local = self._process_group != dist.group.WORLD
        group_type = "local" if is_local else "global"

        # Get list of ranks in this group
        ranks = dist.get_process_group_ranks(self._process_group)

        return f"MeshCollective(rank={self.rank}/{self.world_size}, {group_type}_group={group_rank}/{group_size}, tp_size={self._tp_size}, mesh={self._mesh}, ranks={ranks})"

    @property
    def tp_mesh(self):
        """Returns the sub-mesh for Tensor Parallelism if it exists."""
        assert self._mesh.parallel_mesh.mesh_dim_names is not None
        if "tp" in self._mesh.parallel_mesh.mesh_dim_names:
            return self._mesh.parallel_mesh["tp"]
        return None

    @property
    def dp_mesh(self):
        """Returns the sub-mesh for Data Parallelism if it exists."""
        assert self._mesh.physical_mesh.mesh_dim_names is not None
        return self._mesh.parallel_mesh["dp_replicate", "dp_shard"]

    @property
    @override
    def local(self) -> "MeshCollective":
        assert self._mesh.physical_mesh.mesh_dim_names is not None
        if len(self._mesh.physical_mesh.mesh_dim_names) == 1:
            return self
        process_group = self._mesh.physical_mesh.get_group("local_ranks")
        return MeshCollective(
            rank=self._local_rank,
            world_size=self._local_world_size,
            local_world_size=self._local_world_size,
            local_rank=self._local_rank,
            device_type=self._device_type,
            mesh=self._mesh,
            tp_size=self._tp_size,
            process_group=process_group,
        )

    @property
    @override
    def tp_world(self) -> "MeshCollective":
        assert self._mesh.parallel_mesh.mesh_dim_names is not None

        process_group = self._mesh.parallel_mesh.get_group("tp")
        return MeshCollective(
            rank=self.tp_rank,
            world_size=self.tp_world_size,
            local_world_size=self.tp_world_size,
            local_rank=self.tp_rank,
            device_type=self._device_type,
            mesh=self._mesh,
            tp_size=self._tp_size,
            process_group=process_group,
        )

    @property
    @override
    def local_rank(self):
        return self._local_rank

    @property
    @override
    def dp_rank(self):
        return self.rank // self._tp_size

    @property
    @override
    def dp_world_size(self):
        return self.world_size // self._tp_size

    @property
    @override
    def tp_rank(self):
        return self.rank % self._tp_size

    @property
    @override
    def tp_world_size(self):
        return self._tp_size

    @property
    @override
    def default_device(self) -> torch.device:
        """Get the default device for this collective."""
        if self._device_type == "cuda":
            return torch.device(f"cuda:{self._local_rank}")
        elif self._device_type == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @override
    def close(self) -> None:
        pass

    @override
    def barrier(self) -> None:
        dist.barrier(group=self._process_group)

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOp.RedOpType):
        dist.all_reduce(
            tensor,
            op,
            group=self._process_group,
        )

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor):
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=self._process_group,
        )

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        dist.all_gather(output_tensors, input_tensor, group=self._process_group)

    def all_gather_objects(
        self,
        object: object,
    ) -> list[object]:
        object_list = [None] * dist.get_world_size(group=self._process_group)
        dist.all_gather_object(
            object_list=object_list, obj=object, group=self._process_group
        )
        return object_list

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        dist.broadcast(tensor, source_rank, group=self._process_group)

    @override
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        dist.broadcast_object_list(objects, source_rank, group=self._process_group)

    @property
    @override
    def process_group(self) -> ProcessGroup | None:
        return self._process_group
