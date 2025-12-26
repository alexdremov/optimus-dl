from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.distributed import ProcessGroup, ReduceOp


class Collective(ABC):
    """Abstract base class for distributed communication.

    This class defines the interface for all collective operations and distributed
    topology information. It allows the framework to switch between real
    distributed training (MeshCollective) and single-device/CPU execution
    (FakeCollective) without changing the training logic.

    Attributes:
        rank: Global rank of the current process.
        world_size: Total number of processes in the global gang.
    """

    rank: int
    world_size: int

    def __init__(self, rank, world_size) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        assert rank < world_size

    @property
    @abstractmethod
    def local(self) -> "Collective":
        """Get a collective limited to the current node (local ranks)."""
        ...

    @property
    @abstractmethod
    def tp_world(self) -> "Collective":
        """Get a collective for the current Tensor Parallelism group."""
        ...

    @property
    def is_master(self) -> bool:
        """True if the current process is the master (rank 0)."""
        return self.rank == 0

    @property
    def is_local_master(self) -> bool:
        """True if the current process is the master of its node (local rank 0)."""
        return self.local_rank == 0

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """Rank within the current node."""
        ...

    @property
    @abstractmethod
    def dp_rank(self) -> int:
        """Rank within the Data Parallelism group."""
        ...

    @property
    @abstractmethod
    def dp_world_size(self) -> int:
        """Size of the Data Parallelism group."""
        ...

    @property
    @abstractmethod
    def tp_rank(self) -> int:
        """Rank within the Tensor Parallelism group."""
        ...

    @property
    @abstractmethod
    def tp_world_size(self) -> int:
        """Size of the Tensor Parallelism group."""
        ...

    @property
    @abstractmethod
    def default_device(self) -> torch.device:
        """Get the default PyTorch device for this process."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up distributed resources."""
        ...

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all processes in the collective."""
        ...

    @abstractmethod
    def all_reduce(self, tensor: Tensor, op: ReduceOp.RedOpType) -> None:
        """Perform an all-reduce operation."""
        ...

    @abstractmethod
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        """Perform an all-gather operation into a single tensor."""
        ...

    @abstractmethod
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        """Perform an all-gather operation into a list of tensors."""
        ...

    @abstractmethod
    def all_gather_objects(
        self,
        object: object,
    ) -> list[object]:
        """Perform an all-gather operation for picklable Python objects."""
        ...

    @abstractmethod
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        """Broadcast a tensor from the source rank to all other ranks."""
        ...

    @abstractmethod
    def broadcast_objects(self, objects: list[object], source_rank: int = 0) -> None:
        """Broadcast picklable Python objects from the source rank."""
        ...

    @property
    @abstractmethod
    def process_group(self) -> ProcessGroup | None:
        """The underlying PyTorch ProcessGroup, if available."""
        ...
