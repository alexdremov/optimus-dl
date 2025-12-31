from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
)

import numpy as np

from optimus_dl.core.registry import RegistryConfigStrict


class BaseStrategyConfig(RegistryConfigStrict):
    pass


class BaseStrategy(ABC):
    """Base class for dataset sampling strategies."""

    def __init__(self, cfg: BaseStrategyConfig, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.doc_lengths: np.ndarray | None = None

    def initialize(self, doc_lengths: np.ndarray):
        """Initialize the strategy with document lengths."""
        self.doc_lengths = doc_lengths

    @abstractmethod
    def next_sample(self) -> list[tuple[int, tuple[int, int]]]:
        """Yield the next sample.

        Returns:
            A list of segments required to construct the sample.
            Each segment is a tuple: (doc_id, (start_offset, end_offset)).
            - doc_id: Global document index.
            - start_offset: Start token index within the document (inclusive).
            - end_offset: End token index within the document (exclusive).

        Raises:
            StopIteration: When the strategy is exhausted.
        """
        pass

    @abstractmethod
    def reset(self, initial_state: dict[str, Any] | None = None):
        """Reset state to initial or checkpointed state.

        Args:
            initial_state: State dictionary to restore from (optional).
        """
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            Dictionary containing the current state.
        """
        pass
