"""Simple substring-match reward."""

from dataclasses import dataclass
from typing import Any

import torch

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.reward import register_reward_function
from optimus_dl.modules.reward.base import BaseReward


@dataclass
class RuleBasedRewardConfig(RegistryConfigStrict):
    """Configuration for rule-based reward.

    Attributes:
        target_string: Substring checked against every completion.
        reward_value: Reward granted when the target string is present.
    """

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


register_reward_function("rule_based", RuleBasedRewardConfig)(RuleBasedReward)
