"""GRPO Criterion implementation."""

import logging
from dataclasses import dataclass
from typing import Any

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
        eps: PPO clipping parameter (ε in the surrogate objective).
        kl_coeff: Coefficient for the per-token KL divergence penalty.
    """

    eps: float = 0.2
    kl_coeff: float = 0.01


@register_criterion("grpo", GRPOCriterionConfig)
class GRPOCriterion(BaseCriterion):
    """Group Relative Policy Optimization (GRPO) Criterion.

    Loss = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ] + kl_coeff · KL(π_θ ‖ π_ref)

    where:
      r      = π_θ(a|s) / π_old(a|s)  — token-level policy ratio
      A      — group-relative advantage (pre-computed in the rollout phase)
      KL     — DeepSeek-style approximation: exp(log_ref - log_θ) - (log_ref - log_θ) - 1

    All quantities are masked to completion tokens only; prompt tokens are
    excluded from the loss.
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
                   input_ids       : (B*G, T)   — prompt + completion token
        IDs.
                   completion_mask : (B*G, T)   — 1 for completion tokens, 0 for prompt.
                   old_logprobs    : (B*G, T-1) — log-probs from the rollout policy.
                   ref_logprobs    : (B*G, T-1) — log-probs from the frozen reference model.
                   advantages      : (B*G,)     — group-relative normalised advantages.
                       A (B*G, T-1) per-token tensor is also accepted for dense
                       (token-level) reward variants; it must be zero outside
                       completion positions.
                   sampling_temperature : float — temperature used during
                       rollout sampling. The policy logits are divided by it
                       before the log-softmax so that ``ratio == 1`` when the
                       policy equals the rollout policy even for T != 1.
        """
        input_ids = batch["input_ids"]
        completion_mask = batch["completion_mask"]
        old_logprobs = batch["old_logprobs"]
        ref_logprobs = batch["ref_logprobs"]
        advantages = batch["advantages"]
        sampling_temperature = float(batch.get("sampling_temperature", 1.0))

        # --- Current policy log-probs ---
        # Standard next-token-prediction shift: logit at position t predicts token t+1.
        # seq_lens masks right-padding out of attention when provided.
        forward_kwargs: dict[str, Any] = {}
        if batch.get("seq_lens") is not None:
            forward_kwargs["seq_lens"] = batch["seq_lens"]
        outputs = model(input_ids, **forward_kwargs)
        shift_logits = outputs["logits"][:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # shift_mask: (B*G, T-1) — 1 for completion positions, 0 for prompt positions
        shift_mask = completion_mask[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits / sampling_temperature, dim=-1)
        per_token_logprobs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        # per_token_logprobs: (B*G, T-1) — NOT pre-masked; masking is applied below.

        # --- Policy ratio (completion positions only) ---
        # log_ratio is 0 at prompt positions because shift_mask = 0 there.
        log_ratio = (per_token_logprobs - old_logprobs) * shift_mask
        ratio = torch.exp(log_ratio)  # = 1 at prompt positions

        # --- Clipped PPO surrogate loss ---
        # advantages: (B*G,) broadcast over tokens, or (B*G, T-1) per-token
        # advantages for dense-reward variants.
        adv = advantages if advantages.dim() == 2 else advantages.unsqueeze(1)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.eps, 1.0 + self.cfg.eps) * adv
        policy_loss = -torch.min(surr1, surr2)
        # policy_loss: (B*G, T-1) — non-zero values at completion positions only
        # (ratio = 1 and clamp(1, …) = 1 at prompt positions, but the outer
        # shift_mask in total_loss zeroes them out anyway).

        # --- KL divergence penalty (DeepSeek approximation) ---
        # KL(π_θ ‖ π_ref) ≈ exp(log_ref - log_θ) - (log_ref - log_θ) - 1
        # Masking inside keeps KL = 0 at prompt positions without relying solely
        # on the outer shift_mask multiplication.
        delta = (ref_logprobs - per_token_logprobs) * shift_mask
        kl = torch.exp(delta) - delta - 1  # (B*G, T-1), = 0 at prompt positions

        # --- Total loss, normalised by completion token count ---
        total_loss = (policy_loss + self.cfg.kl_coeff * kl) * shift_mask

        num_tokens = shift_mask.sum().clamp(min=1)  # avoid division by zero
        mean_loss = total_loss.sum() / num_tokens
        mean_policy_loss = (policy_loss * shift_mask).sum() / num_tokens
        mean_kl = (kl * shift_mask).sum() / num_tokens
        mean_ratio = (ratio * shift_mask).sum() / num_tokens

        # --- Metrics ---
        weight = input_ids.size(0)
        log_averaged("loss", lambda: mean_loss.item(), weight=weight, round=4)
        log_averaged(
            "policy_loss", lambda: mean_policy_loss.item(), weight=weight, round=4
        )
        log_averaged("kl_div", lambda: mean_kl.item(), weight=weight, round=4)
        log_averaged("ratio", lambda: mean_ratio.item(), weight=weight, round=4)
        log_averaged(
            "num_tokens_per_step",
            lambda: num_tokens.item(),
            weight=weight,
            round=0,
        )

        exposed = {
            "policy_loss": mean_policy_loss,
            "kl_div": mean_kl,
            "ratio": mean_ratio,
        }

        return mean_loss, exposed
