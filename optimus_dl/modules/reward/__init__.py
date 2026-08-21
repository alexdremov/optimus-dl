"""Reward functions for RLHF and GRPO."""

import logging
import math
import re
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    Any,
)

import torch

from optimus_dl.core.registry import (
    RegistryConfigStrict,
    make_registry,
)

logger = logging.getLogger(__name__)


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


@dataclass
class RuleBasedRewardConfig(RegistryConfigStrict):
    """Configuration for rule-based reward."""

    target_string: str = ""
    reward_value: float = 1.0


class RuleBasedReward(BaseReward):
    """A simple reward function that checks for a target string."""

    def __init__(self, cfg: RuleBasedRewardConfig):
        self.cfg = cfg

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        rewards = []
        for completion in completions:
            if self.cfg.target_string in completion:
                rewards.append(self.cfg.reward_value)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards, dtype=torch.float32)


(
    _REWARD_FUNCTION,
    register_reward_function,
    build_reward_function,
) = make_registry("reward_function")


@dataclass
class GSM8kRewardConfig(RegistryConfigStrict):
    """Configuration for GSM8k reward.

    Attributes:
        reward_value: Value for exact matches.
        penalty_value: Base value for incorrect answers (if not dense).
        dense_reward: Whether to use distance-based rewards.
        dense_scale: Scale factor for dense reward (sigma).
            Reward = exp(-abs(pred - true) / dense_scale)
    """

    reward_value: float = 1.0
    penalty_value: float = 0.0
    dense_reward: bool = False
    dense_scale: float = 10.0


class GSM8kReward(BaseReward):
    """Reward function for GSM8k that extracts numerical answers."""

    def __init__(self, cfg: GSM8kRewardConfig):
        self.cfg = cfg

    def extract_answer(self, text: str) -> float | None:
        """Extract the last number from the text as a float."""
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str] = None,  # True answers from the dataset
        **kwargs: Any,
    ) -> torch.Tensor:
        rewards = []
        for i, completion in enumerate(completions):
            if answers is None or i >= len(answers):
                rewards.append(0.0)
                continue

            pred = self.extract_answer(completion)
            true_ans = self.extract_answer(answers[i])

            reward = self.cfg.penalty_value

            if pred is not None and true_ans is not None:
                if abs(pred - true_ans) < 1e-6:
                    reward = self.cfg.reward_value
                elif self.cfg.dense_reward:
                    # Exponential decay based on distance
                    # Reward will be between penalty_value and reward_value
                    dist = abs(pred - true_ans)
                    dense_val = math.exp(-dist / self.cfg.dense_scale)
                    # Scale to [penalty, reward] range
                    reward = (
                        self.cfg.penalty_value
                        + (self.cfg.reward_value - self.cfg.penalty_value) * dense_val
                    )

            logger.debug(
                f"GSM8k Reward: Extracted '{pred}', True '{true_ans}' -> Reward {reward:.4f}"
            )
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)


register_reward_function("gsm8k", GSM8kRewardConfig)(GSM8kReward)
register_reward_function("rule_based", RuleBasedRewardConfig)(RuleBasedReward)
