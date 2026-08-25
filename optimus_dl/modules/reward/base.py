"""Base classes and shared helpers for reward functions."""

import re
from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

import torch


def extract_last_number(text: str) -> float | None:
    """Extract the last number occurring in ``text`` as a float."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


class BaseReward(ABC):
    """Base class for all reward functions."""

    @abstractmethod
    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate a batch of completions.

        Args:
            prompts: List of prompt strings.
            completions: List of generated completion strings.

        Returns:
            Tensor of rewards of shape (batch_size,).
        """
        pass

    def supports_token_level(self) -> bool:
        """Whether this reward provides per-token (dense) rewards.

        Token-level rewards enable dense GRPO: instead of broadcasting one
        scalar advantage over every completion token, the reward assigns
        credit to individual generated tokens (see ``token_rewards``).
        """
        return False

    def token_rewards(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str] | None = None,
        completion_ids: list[list[int]] | None = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Per-token rewards aligned with each completion's tokens.

        Only called when ``supports_token_level()`` is True. Implementations
        must return one 1-D tensor per sample whose length equals
        ``len(completion_ids[i])``; entry ``t`` credits ``completion_ids[i][t]``.

        Note:
            A token-level reward function should return **zeros** from
            ``__call__`` — its sequence-level contribution flows exclusively
            through ``token_rewards`` (the recipe adds both).
        """
        raise NotImplementedError
