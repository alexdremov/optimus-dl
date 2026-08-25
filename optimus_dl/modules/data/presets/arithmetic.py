"""Synthetic arithmetic environment dataset.

Generates an infinite stream of arithmetic problems of the form
``"34+57" -> "91"``. Each sample is a dict with a ``question`` field
(the left-hand side expression without the trailing equals sign) and an
``answer`` field (the exact result as a string).

Problems are generated deterministically from integer indices, which makes
the stream trivially checkpointable and perfectly reproducible across runs
and distributed ranks (each rank consumes a disjoint stride of the index
space).
"""

import logging
import random
from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data import register_dataset
from optimus_dl.modules.data.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

SUPPORTED_OPERATIONS = ("+", "-", "*")


@dataclass
class ArithmeticDatasetConfig(RegistryConfigStrict):
    """Configuration for the synthetic arithmetic dataset.

    Attributes:
        operations: Operators sampled per problem. Supported values are
            "+", "-" and "*". A single operator is used per problem,
            evaluated left-to-right (subtraction may yield negative answers).
        num_operands: Number of operands per problem (>= 2).
        min_digits: Minimum number of digits per operand (>= 1).
        max_digits: Maximum number of digits per operand (>= min_digits).
        max_result: If set, reject problems whose absolute result exceeds
            this value (useful for curricula, e.g. keeping all answers
            single-digit early in training).
        seed: Extra seed offset mixed into the per-rank data seed for
            reproducibility. Change it to draw a different problem stream.
    """

    operations: list[str] = MISSING
    num_operands: int = 2
    min_digits: int = 1
    max_digits: int = 2
    max_result: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.operations is MISSING or len(self.operations) == 0:
            self.operations = ["+"]
        unknown = set(self.operations) - set(SUPPORTED_OPERATIONS)
        assert not unknown, f"Unsupported operations: {sorted(unknown)}"
        assert self.num_operands >= 2, "num_operands must be at least 2"
        assert self.min_digits >= 1, "min_digits must be at least 1"
        assert self.max_digits >= self.min_digits, "max_digits must be >= min_digits"
        assert self.max_result is None or self.max_result >= 1


class ArithmeticDataset(BaseDataset):
    """Infinite deterministic stream of arithmetic problems.

    The problem at global index ``i`` is derived by seeding a private RNG
    with ``"{seed}:{i}"``, so samples are independent, reproducible, and do
    not require storing any RNG state. Rank ``r`` of ``world_size`` yields
    indices ``i = position * world_size + r``, giving each rank a unique,
    interleaved slice of the stream.

    Args:
        cfg: Arithmetic dataset configuration.
        rank: Distributed rank (injected by the data builder).
        world_size: Total number of ranks (injected by the data builder).
        seed: Per-rank data seed (injected by the data builder).
    """

    def __init__(
        self,
        cfg: ArithmeticDatasetConfig,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        **kwargs: Any,
    ):
        super().__init__(cfg)
        self.cfg: ArithmeticDatasetConfig = cfg
        self.rank = rank
        self.world_size = world_size
        self.seed = cfg.seed + int(seed)
        self.position = 0

    def _sample_operand(self, rng: random.Random) -> int:
        digits = rng.randint(self.cfg.min_digits, self.cfg.max_digits)
        low = 10 ** (digits - 1)
        high = 10**digits - 1
        return rng.randint(low, high)

    def _problem_at(self, index: int) -> dict[str, str]:
        rng = random.Random(f"{self.seed}:{index}")
        op = rng.choice(list(self.cfg.operations))

        # Rejection-sample until the result satisfies max_result. The loop is
        # deterministic for a given index, preserving reproducibility.
        for _ in range(1000):
            operands = [self._sample_operand(rng) for _ in range(self.cfg.num_operands)]
            value = operands[0]
            for operand in operands[1:]:
                if op == "+":
                    value += operand
                elif op == "-":
                    value -= operand
                else:
                    value *= operand
            if self.cfg.max_result is None or abs(value) <= self.cfg.max_result:
                break
        else:
            raise ValueError(
                f"Could not satisfy max_result={self.cfg.max_result} for "
                f"operations={self.cfg.operations}; the constraint is likely "
                "infeasible for the configured operand ranges."
            )

        question = f"{operands[0]}"
        for operand in operands[1:]:
            question += f"{op}{operand}"
        return {"question": question, "answer": str(value)}

    def next(self) -> dict[str, str]:
        """Yield the next problem from this rank's slice of the stream."""
        sample = self._problem_at(self.position * self.world_size + self.rank)
        self.position += 1
        return sample

    def reset(self, initial_state: dict | None = None) -> None:
        """Restore the stream position or start from the beginning."""
        super().reset(initial_state)
        self.position = initial_state.get("position", 0) if initial_state else 0
        assert (
            initial_state is None or initial_state.get("rank", self.rank) == self.rank
        ), "Rank mismatch during state restore"
        assert (
            initial_state is None
            or initial_state.get("world_size", self.world_size) == self.world_size
        ), "World size mismatch during state restore"

    def get_state(self) -> dict:
        """Return the current stream position for checkpointing."""
        return {
            "position": self.position,
            "rank": self.rank,
            "world_size": self.world_size,
        }


@register_dataset("preset_arithmetic", ArithmeticDatasetConfig)
def make_arithmetic(
    cfg: ArithmeticDatasetConfig,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    **_: Any,
) -> ArithmeticDataset:
    return ArithmeticDataset(cfg, rank=rank, world_size=world_size, seed=seed)
