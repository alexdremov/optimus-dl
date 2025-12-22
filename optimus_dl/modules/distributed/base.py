from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp


class Collective(ABC):
    rank: int
    world_size: int

    def __init__(self, rank, world_size) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        assert rank < world_size

    @property
    @abstractmethod
    def local(self) -> "Collective": ...

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def is_local_master(self) -> bool:
        return self.local_rank == 0

    @property
    @abstractmethod
    def local_rank(self) -> int: ...

    @property
    @abstractmethod
    def default_device(self) -> torch.device:
        """Get the default device for this collective."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def barrier(self) -> None: ...

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOp.RedOpType) -> None: ...

    @abstractmethod
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None: ...

    @abstractmethod
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None: ...

    @abstractmethod
    def all_gather_objects(
        self,
        object: object,
    ) -> list[object]: ...

    @abstractmethod
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None: ...

    @abstractmethod
    def broadcast_objects(
        self, objects: list[object], source_rank: int = 0
    ) -> None: ...

    @property
    @abstractmethod
    def global_process_group(self) -> ProcessGroup | None: ...
