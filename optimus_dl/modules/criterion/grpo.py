"""GRPO Criterion implementation."""

import logging
from dataclasses import dataclass
from typing import (
    Any,
)

import torch
import torch.nn.functional as F

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.criterion import (
    BaseCriterion,
    register_criterion,
)
from optimus_dl.modules.metrics import log_averaged
from optimus_dl.modules.model.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class GRPOCriterionConfig(RegistryConfigStrict):
    """Configuration for GRPO criterion.

    Attributes:
        eps: PPO clipping parameter.
        kl_coeff: Coefficient for KL divergence penalty.
    """

    eps: float = 0.2
    kl_coeff: float = 0.01


@register_criterion("grpo", GRPOCriterionConfig)
class GRPOCriterion(BaseCriterion):
    """Group Relative Policy Optimization (GRPO) Criterion.

    Computes the clipped PPO policy loss and KL divergence penalty using
    group-relative advantages.
    """

    def __init__(
        self, cfg: GRPOCriterionConfig, collective: Any | None = None, **kwargs: Any
    ):
        self.cfg = cfg
        self.collective = collective

    def __call__(
        self,
        model: BaseModel,
        batch: dict[str, Any],
        requested_protocols: set[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute GRPO loss.

        Expected batch keys:
            - input_ids: Tokens (prompts + completions).
            - completion_mask: Mask for completion tokens.
            - old_logprobs: Logprobs from the rollout policy.
            - ref_logprobs: Logprobs from the reference model.
            - advantages: Group-relative advantages.
        """
        input_ids = batch["input_ids"]
        completion_mask = batch["completion_mask"]
        old_logprobs = batch["old_logprobs"]
        ref_logprobs = batch["ref_logprobs"]
        advantages = batch["advantages"]

        # Forward pass to get current log-probabilities
        outputs = model(input_ids)
        logits = outputs["logits"]

        # Compute logprobs for the completions
        # Shift logits and targets for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = completion_mask[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logprobs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Apply mask
        per_token_logprobs = per_token_logprobs * shift_mask

        # We need to sum logprobs over the completion length for the policy ratio?
        # Actually, PPO can be done per token or per sequence.
        # Standard GRPO/PPO for LLMs often does it per completion.

        # Policy Ratio
        # Only compute ratio where mask is 1 to avoid exp(0) or exp(-logprob) artifacts
        # Note: logprobs are typically negative, so -masked_logprob is positive.
        log_ratio = (per_token_logprobs - old_logprobs) * shift_mask
        ratio = torch.exp(log_ratio)

        # Clipped PPO Loss
        # advantages is (B*G), expand to (B*G, T-1)
        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = torch.clamp(
            ratio, 1.0 - self.cfg.eps, 1.0 + self.cfg.eps
        ) * advantages.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2)

        # KL Penalty (log pi_theta - log pi_ref)
        # DeepSeek uses: kl = exp(log_ref - log_theta) - (log_ref - log_theta) - 1
        kl = (
            torch.exp((ref_logprobs - per_token_logprobs) * shift_mask)
            - ((ref_logprobs - per_token_logprobs) * shift_mask)
            - 1
        )

        total_loss = (policy_loss + self.cfg.kl_coeff * kl) * shift_mask

        # Normalize by number of active tokens
        num_tokens = shift_mask.sum()
        mean_loss = total_loss.sum() / num_tokens
        mean_policy_loss = (policy_loss * shift_mask).sum() / num_tokens
        mean_kl = (kl * shift_mask).sum() / num_tokens
        mean_ratio = (ratio * shift_mask).sum() / num_tokens

        # Log metrics
        weight = input_ids.size(0)
        log_averaged("loss", lambda: mean_loss.item(), weight=weight, round=4)
        log_averaged(
            "policy_loss", lambda: mean_policy_loss.item(), weight=weight, round=4
        )
        log_averaged("kl_div", lambda: mean_kl.item(), weight=weight, round=4)
        log_averaged("ratio", lambda: mean_ratio.item(), weight=weight, round=4)

        exposed = {
            "policy_loss": mean_policy_loss,
            "kl_div": mean_kl,
            "ratio": mean_ratio,
        }

        return mean_loss, exposed
