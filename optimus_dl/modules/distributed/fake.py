import torch
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp
from torch.types import Device
from typing_extensions import override

from optimus_dl.modules.distributed.base import Collective


class FakeCollective(Collective):
    def __init__(self, rank, world_size, device_type: str = "cpu") -> None:
        super().__init__(rank, world_size)
        assert world_size == 1, "Fake collective is fake"
        self._device_type = device_type

    @property
    @override
    def local(self) -> "FakeCollective":
        return self

    @property
    @override
    def local_rank(self):
        return self.rank

    @property
    @override
    def default_device(self) -> torch.device:
        """Get the default device for this collective."""
        if self._device_type == "cuda":
            return torch.device("cuda:0")  # Single device for fake collective
        elif self._device_type == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @override
    def close(self) -> None:
        pass

    @override
    def barrier(self) -> None:
        pass

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOp.RedOpType) -> None:
        pass

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        if not output_tensor.is_contiguous():
            raise ValueError("`output_tensor` must be contiguous.")

        if output_tensor.ndim != input_tensor.ndim + 1:
            raise ValueError(
                "`output_tensor` must have a shape that is compatible with all-gather."
            )

        if output_tensor.size(0) != self.world_size:
            raise ValueError(
                f"The size of the first dimension of `output_tensor` must match the number of processes in the gang ({self._size}), but is {output_tensor.size(0)} instead."
            )

        for i in range(self.world_size):
            output_tensor[i].copy_(input_tensor)

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        if len(output_tensors) != self.world_size:
            raise ValueError(
                f"The length of `output_tensors` must match the number of processes in the gang ({self._size}), but is {len(output_tensors)} instead."
            )

        for i in range(self.world_size):
            output_tensors[i].copy_(input_tensor)

    @override
    def all_gather_objects(
        self,
        object: object,
    ):
        return [object]

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        if source_rank != self.rank:
            raise ValueError(
                f"`source_rank` must be {self.rank}, but is {source_rank} instead."
            )

    @override
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        if source_rank != self.rank:
            raise ValueError(
                f"`source_rank` must be {self.rank}, but is {source_rank} instead."
            )

    @property
    @override
    def global_process_group(self) -> ProcessGroup | None:
        return None
