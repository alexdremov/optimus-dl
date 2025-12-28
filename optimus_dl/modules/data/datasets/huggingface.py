import logging
from dataclasses import dataclass

import datasets
import datasets.distributed
from omegaconf import (
    OmegaConf,
    MISSING,
)
from datasets import load_dataset

from optimus_dl.core.registry import RegistryConfigStrict

from . import register_dataset
from .base import BaseDataset

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceDatasetConfig(RegistryConfigStrict):
    """Configuration for Hugging Face datasets.

    Attributes:
        dataset_load_kwargs: Dictionary of arguments passed to `datasets.load_dataset`.
            e.g., {"path": "wikitext", "name": "wikitext-2-raw-v1", "split": "train"}.
    """

    dataset_load_kwargs: dict = MISSING


@register_dataset("huggingface_dataset", HuggingFaceDatasetConfig)
class HuggingFaceDataset(BaseDataset):
    """Dataset wrapper for Hugging Face Hub datasets.

    This class integrates with the Hugging Face `datasets` library, supporting:

    - **Streaming**: Automatically enables streaming for efficient loading of
      large datasets without downloading everything.
    - **Distributed Sharding**: Uses `split_dataset_by_node` to ensure each rank
      sees a unique portion of the data.
    - **Checkpointing**: Tracks current position to allow resuming from the middle
      of a stream.

    Args:
        cfg: Hugging Face dataset configuration.
        rank: Distributed rank.
        world_size: Total number of ranks.
    """

    def __init__(self, cfg, rank: int, world_size: int, **kwargs):
        super().__init__(cfg)
        self.rank = rank
        self.world_size = world_size
        self.position = 0

    def get_state(self):
        """Return the current position and configuration for checkpointing."""
        return {
            "cfg": self.cfg,
            "dataset_state": (
                self.dataset.state_dict()
                if hasattr(self.dataset, "state_dict")
                else None
            ),
            "world_size": self.world_size,
            "rank": self.rank,
            "position": self.position,
        }

    def reset(self, initial_state: dict | None = None):
        """Initialize or restore the dataset stream.

        Configures streaming, performs distributed sharding, and skips to the
        saved position if restoring from a checkpoint.
        """
        super().reset(initial_state)
        if initial_state is not None:
            self.cfg = initial_state.get("cfg", self.cfg)
            self.cfg = OmegaConf.merge(
                OmegaConf.structured(HuggingFaceDatasetConfig), self.cfg
            )
            self.position = initial_state["position"]

            assert self.rank == initial_state.get("rank", self.rank)
            assert self.world_size == initial_state.get("world_size", self.world_size)

        if (
            "streaming" in self.cfg.dataset_load_kwargs
            and not self.cfg.dataset_load_kwargs["streaming"]
        ):
            logger.info("streaming=False is not recommended")
        else:
            self.cfg.dataset_load_kwargs["streaming"] = True

        if not self.cfg.dataset_load_kwargs.get("streaming"):
            self.cfg.dataset_load_kwargs.setdefault("num_proc", 4)
        self.dataset = load_dataset(**self.cfg.dataset_load_kwargs)

        if self.world_size > 1:
            logger.info(
                f"Sharding dataset... (num_shards={self.world_size}, index={self.rank})"
            )
            self.dataset = datasets.distributed.split_dataset_by_node(
                dataset=self.dataset,
                rank=self.rank,
                world_size=self.world_size,
            )

        if (
            initial_state is not None
            and "dataset_state" in initial_state
            and initial_state["dataset_state"] is not None
        ):
            self.dataset.load_state_dict(initial_state["dataset_state"])

        if not isinstance(self.dataset, datasets.IterableDataset):
            self.dataset = self.dataset.skip(self.position)
        self.iter = iter(self.dataset)

    def next(self):
        """Yield the next example from the Hugging Face dataset."""
        self.position += 1
        return next(self.iter)
