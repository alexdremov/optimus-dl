"""Reward functions for RLHF and GRPO.

The registry lives here; concrete reward implementations live in
``optimus_dl.modules.reward.implementations`` and are imported automatically
(see ``bootstrap_module``), so adding a new reward only requires dropping a
module with a ``@register_reward_function`` declaration into that package.

Public classes are re-exported below for convenience.
"""

from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.core.registry import make_registry

from .base import (  # noqa: E402
    BaseReward,
    extract_last_number,
)

(
    _REWARD_FUNCTION,
    register_reward_function,
    build_reward_function,
) = make_registry("reward_function")

bootstrap_module(__name__)
