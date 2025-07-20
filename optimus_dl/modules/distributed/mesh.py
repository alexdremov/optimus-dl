import logging

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp, init_process_group
from torch.distributed.device_mesh import init_device_mesh
from typing_extensions import override

from optimus_dl.modules.distributed.base import Collective

logger = logging.getLogger(__name__)


class MeshCollective(Collective):
    def __init__(
        self,
        rank,
        world_size,
        local_world_size,
        local_rank,
        device_type,
        mesh=None,
        process_group=None,
    ) -> None:
        super().__init__(rank, world_size)
        assert world_size % local_world_size == 0

        self._device_type = device_type
        self._local_rank = local_rank
        self._local_world_size = local_world_size

        mesh_shape = (
            world_size // local_world_size,
            local_world_size,
        )
        mesh_dim_names = ("dp_replicate", "dp_shard")

        if mesh is None:
            logger.info(f"Initialized mesh with {mesh_shape = }")
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
            logger.info(f"Initializing mesh with {mesh_device_type = }")
            mesh = init_device_mesh(
                device_type=mesh_device_type,
                mesh_shape=mesh_shape,
                mesh_dim_names=mesh_dim_names,
            )
        self._mesh = mesh

        # Default to global group (world group)
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
        try:
            ranks = dist.get_process_group_ranks(self._process_group)
        except:
            ranks = ["unknown"]

        return f"MeshCollective(rank={self.rank}/{self.world_size}, {group_type}_group={group_rank}/{group_size}, local_rank={self._local_rank}/{self._local_world_size}, ranks={ranks})"

    @property
    @override
    def local(self) -> "MeshCollective":
        if len(self._mesh.mesh_dim_names) == 1:
            return self
        return MeshCollective(
            rank=self._local_rank,
            world_size=self._local_world_size,
            local_world_size=self._local_world_size,
            local_rank=self._local_rank,
            device_type=self._device_type,
            mesh=self._mesh,
            process_group=self._mesh.get_group(1),
        )

    @property
    @override
    def local_rank(self):
        return self._local_rank

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
    def all_reduce(self, tensor: Tensor, op: ReduceOp.RedOpType) -> None:
        dist.all_reduce(
            tensor,
            op,
            group=self._process_group,
        )

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
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
    def global_process_group(self) -> ProcessGroup | None:
        return self._process_group
