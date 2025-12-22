import copy
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfig

from . import build_dataset, register_dataset
from .base import BaseDataset

logger = logging.getLogger(__name__)


class StopCriteria(StrEnum):
    FIRST_DATASET_EXHAUSTED = "FIRST_DATASET_EXHAUSTED"
    ALL_DATASETS_EXHAUSTED = "ALL_DATASETS_EXHAUSTED"
    CYCLE_FOREVER = "CYCLE_FOREVER"


@dataclass
class DatasetConfig:
    dataset: RegistryConfig = field(
        default=MISSING, metadata={"help": "Dataset config to load"}
    )
    weight: float = field(
        default=1.0, metadata={"help": "Weight of the dataset for sampling"}
    )
    cycle: bool = field(
        default=True,
        metadata={"help": "Whether to cycle through the dataset after it is exhausted"},
    )


@dataclass
class CompositeDatasetConfig(RegistryConfig):
    datasets: dict[str, DatasetConfig] = field(
        default_factory=dict, metadata={"help": "Datasets to load: name -> config"}
    )
    strict_load: bool = field(
        default=True,
        metadata={
            "help": "Whether to raise an error if state dict does not contain all required keys"
        },
    )
    seed: int | None = field(
        default=None, metadata={"help": "Random seed for sampling from datasets"}
    )
    stop_criteria: StopCriteria = field(
        default=StopCriteria.CYCLE_FOREVER,
        metadata={"help": "Stop criteria for the composite dataset"},
    )


def _get_rank_seed(seed: int, epoch: int, rank: int) -> int:
    """
    Generate a unique seed for each rank based on the base seed, rank, and epoch.
    """
    rng = torch.Generator()
    # Mix seed, epoch, and rank deterministically
    rng.manual_seed(seed + epoch * 10000 + rank)
    return rng.initial_seed()


class _WeightedSampler:
    """
    Helper class for weighted sampling.
    """

    def __init__(
        self,
        weights: dict[str, float],
        seed: int,
        rank: int,
        world_size: int,
        epoch: int,
        random_tensor_batch_size: int = 1000,
        initial_state: dict[str, Any] | None = None,
    ):
        _names, _weights = [], []
        for name, weight in weights.items():
            _names.append(name)
            _weights.append(weight)

        self.names = _names
        self.original_weights = torch.tensor(_weights, dtype=torch.float64)
        self.weights = self.original_weights.clone()
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}

        self.random_tensor_batch_size = random_tensor_batch_size

        self._g = torch.Generator()

        self.epoch = epoch
        final_seed = _get_rank_seed(seed, epoch, rank)
        self._g.manual_seed(final_seed)

        # Snapshot of the generator state at the beginning of the batch
        self._g_snapshot = self._g.get_state()

        if initial_state is not None:
            self._g.set_state(initial_state["g_state"])
            self._g_snapshot = self._g.get_state()
            self._offset = initial_state["offset"]
            # Load weights state if available, otherwise assume all active (or original)
            if "weights" in initial_state:
                self.weights = initial_state["weights"]
        else:
            self._offset = 0

        self._batch_of_indices = self._get_batch_of_indices()

    def set_active(self, name: str, active: bool):
        idx = self.name_to_idx[name]
        if active:
            self.weights[idx] = self.original_weights[idx]
        else:
            self.weights[idx] = 0.0

        # Invalidate current batch to force re-sampling with new weights
        self._batch_of_indices = []
        self._offset = 0

    def _get_batch_of_indices(self) -> list[int]:
        if torch.sum(self.weights) == 0:
            return []

        # Update snapshot to the state before generating the batch
        self._g_snapshot = self._g.get_state()
        return torch.multinomial(
            self.weights,
            num_samples=self.random_tensor_batch_size,
            replacement=True,
            generator=self._g,
        ).tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if self._offset >= len(self._batch_of_indices):
            self._batch_of_indices = self._get_batch_of_indices()
            self._offset = 0

            if not self._batch_of_indices:
                # Should be handled by caller checking stop criteria,
                # but if we end up here with 0 weights, stop.
                raise StopIteration

        item = self._batch_of_indices[self._offset]
        self._offset += 1
        return self.names[item]

    def state_dict(self):
        return {
            "g_state": self._g_snapshot,
            "offset": self._offset,
            "weights": self.weights,
        }


@register_dataset("composite", CompositeDatasetConfig)
class CompositeDataset(BaseDataset):
    DATASET_NODE_STATES_KEY = "dataset_node_states"
    DATASETS_EXHAUSTED_KEY = "datasets_exhausted"
    EPOCH_KEY = "epoch"
    NUM_YIELDED_KEY = "num_yielded"
    WEIGHTED_SAMPLER_STATE_KEY = "weighted_sampler_state"

    def __init__(
        self, cfg: CompositeDatasetConfig, rank: int, world_size: int, **kwargs
    ):
        super().__init__(cfg)
        self.rank = rank
        self.world_size = world_size

        self.datasets = {}
        self.weights = {}
        self.cycle_flags = {}

        for name, ds_cfg in cfg.datasets.items():
            logger.info(f"Initializing sub-dataset {name} with weight {ds_cfg.weight}")
            # Sub-datasets are likely BaseNodes themselves
            self.datasets[name] = build_dataset(
                ds_cfg.dataset, rank=rank, world_size=world_size, **kwargs
            )
            self.weights[name] = ds_cfg.weight
            self.cycle_flags[name] = ds_cfg.cycle

        self.stop_criteria = cfg.stop_criteria
        self.seed = cfg.seed if cfg.seed is not None else 0
        self.strict_load = cfg.strict_load

        self._validate()

        self._epoch = 0
        self._num_yielded = 0
        self._weighted_sampler = None
        self._datasets_exhausted = {}

    def _validate(self):
        for weight in self.weights.values():
            if weight < 0:
                raise ValueError("Weights must be non-negative")

    def reset(self, initial_state: dict[str, Any] | None = None):
        super().reset(initial_state)

        config_datasets = self.datasets.keys()

        if initial_state is not None:
            # Handle strict_load
            state_datasets = initial_state.get(self.DATASET_NODE_STATES_KEY, {}).keys()

            if self.strict_load:
                if set(state_datasets) != set(config_datasets):
                    raise ValueError(
                        f"Strict load enabled. Mismatch in datasets.\n"
                        f"Config: {list(config_datasets)}\nState: {list(state_datasets)}"
                    )

            self._num_yielded = initial_state.get(self.NUM_YIELDED_KEY, 0)
            self._epoch = initial_state.get(self.EPOCH_KEY, 0)
            self._datasets_exhausted = initial_state.get(
                self.DATASETS_EXHAUSTED_KEY, {k: False for k in config_datasets}
            )

            # If config matches state datasets, we can load sampler state
            if set(state_datasets) == set(config_datasets):
                sampler_state = initial_state.get(self.WEIGHTED_SAMPLER_STATE_KEY)
                self._weighted_sampler = self._get_new_weighted_sampler(sampler_state)
            else:
                # Mismatch and strict_load=False: Reset sampler
                logger.warning(
                    "Dataset configuration changed (or strict_load=False), resetting weighted sampler state."
                )
                self._weighted_sampler = self._get_new_weighted_sampler(None)

            # Load sub-datasets
            for name, dataset in self.datasets.items():
                if name in initial_state.get(self.DATASET_NODE_STATES_KEY, {}):
                    dataset.reset(initial_state[self.DATASET_NODE_STATES_KEY][name])
                else:
                    if self.strict_load:
                        # Should have been caught above, but safety check
                        raise ValueError(f"Missing state for dataset {name}")
                    logger.info(f"Resetting dataset {name} (not found in state).")
                    dataset.reset()
        else:
            # Fresh start
            self._num_yielded = 0
            self._epoch = 0

            self._weighted_sampler = self._get_new_weighted_sampler()
            self._datasets_exhausted = {key: False for key in self.datasets}
            for dataset in self.datasets.values():
                dataset.reset()

    def _get_new_weighted_sampler(self, initial_state=None):
        return _WeightedSampler(
            weights=self.weights,
            seed=self.seed,
            rank=self.rank,
            world_size=self.world_size,
            epoch=self._epoch,
            initial_state=initial_state,
        )

    def _check_for_stop_iteration(self) -> None:
        if self.stop_criteria == StopCriteria.CYCLE_FOREVER:
            return

        if all(self._datasets_exhausted.values()):
            raise StopIteration()

        if self.stop_criteria == StopCriteria.FIRST_DATASET_EXHAUSTED and any(
            self._datasets_exhausted.values()
        ):
            raise StopIteration()

    def next(self) -> Any:
        while True:
            self._check_for_stop_iteration()

            # Get next dataset to sample from
            try:
                ds_name = next(self._weighted_sampler)
            except StopIteration as err:
                # If sampler is empty (all weights 0), we should have caught it in check_for_stop_iteration
                # unless there's a sync issue. Treat as end of data.
                raise RuntimeError(
                    "Exhausted all datasets and cannot cycle throug"
                ) from err
            try:
                assert not self._datasets_exhausted[ds_name]
                item = next(self.datasets[ds_name])
                self._num_yielded += 1
                return item

            except StopIteration:
                self._datasets_exhausted[ds_name] = True

                if self.cycle_flags[ds_name]:
                    # Reset this dataset
                    logger.debug(f"Cycling dataset {ds_name}")
                    self.datasets[ds_name].reset()
                    self._datasets_exhausted[ds_name] = False
                    try:
                        item = next(self.datasets[ds_name])
                        self._num_yielded += 1
                        return item
                    except StopIteration as err:
                        raise RuntimeError(
                            "Cannot yield at least one item from dataset after resetting and trying to cycle"
                        ) from err
                else:
                    # Not cycling: Disable this dataset in sampler to avoid polling it again
                    self._weighted_sampler.set_active(ds_name, False)

                self._check_for_stop_iteration()

    def get_state(self) -> dict[str, Any]:
        return {
            self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
            self.DATASET_NODE_STATES_KEY: {
                k: v.state_dict() for k, v in self.datasets.items()
            },
            self.EPOCH_KEY: self._epoch,
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.WEIGHTED_SAMPLER_STATE_KEY: (
                self._weighted_sampler.state_dict() if self._weighted_sampler else None
            ),
        }
