"""Tests for GRPOCriterion and GRPOCriterionConfig.

Design principle: no mocking.  log_averaged() is a silent no-op outside a
meters_group context, so criterion calls work exactly as in production — just
without accumulating metrics into any group.  When we need to verify logged
values, we wrap in ``with meters_group(...):`` and read them back.
"""

import torch
import pytest
import torch.nn as nn

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.criterion.grpo import (
    GRPOCriterion,
    GRPOCriterionConfig,
)
from optimus_dl.modules.distributed.fake import FakeCollective
from optimus_dl.modules.metrics import (
    compute_meters,
    meters_group,
    reset_meters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    prompt_len: int,
    advantages: torch.Tensor | None = None,
    *,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    completion_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    completion_mask[:, prompt_len:] = 1
    old_logprobs = torch.randn(batch_size, seq_len - 1) * 0.1 - 1.0
    ref_logprobs = torch.randn(batch_size, seq_len - 1) * 0.1 - 1.0
    if advantages is None:
        advantages = torch.zeros(batch_size)
    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
    }


class _TinyLM(nn.Module):
    """Minimal language model: embed → linear head."""

    def __init__(self, vocab_size: int = 20, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, **_):
        return {"logits": self.head(self.embed(input_ids))}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestGRPOCriterionConfig:
    def test_default_values(self):
        cfg = GRPOCriterionConfig()
        assert cfg.eps == pytest.approx(0.2)
        assert cfg.kl_coeff == pytest.approx(0.01)

    def test_custom_values(self):
        cfg = GRPOCriterionConfig(eps=0.1, kl_coeff=0.05)
        assert cfg.eps == pytest.approx(0.1)
        assert cfg.kl_coeff == pytest.approx(0.05)

    def test_inherits_registry_config(self):
        assert isinstance(GRPOCriterionConfig(), RegistryConfigStrict)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGRPOCriterionInit:
    def test_stores_config_and_collective(self):
        cfg = GRPOCriterionConfig(eps=0.3, kl_coeff=0.02)
        crit = GRPOCriterion(cfg, collective=FakeCollective(0, 1))
        assert crit.cfg is cfg
        assert crit.collective is not None


# ---------------------------------------------------------------------------
# Output shape, type, and finiteness
# ---------------------------------------------------------------------------


class TestGRPOCriterionOutputShape:
    def test_returns_scalar_finite_loss(self):
        B, T, V = 4, 10, 30
        model = _TinyLM(V)
        crit = GRPOCriterion(GRPOCriterionConfig())
        batch = _make_batch(B, T, V, prompt_len=3)
        loss, _ = crit(model, batch)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_exposed_keys(self):
        B, T, V = 2, 8, 20
        model = _TinyLM(V)
        crit = GRPOCriterion(GRPOCriterionConfig())
        batch = _make_batch(B, T, V, prompt_len=2)
        _, exposed = crit(model, batch)
        assert set(exposed) >= {"policy_loss", "kl_div", "ratio"}


# ---------------------------------------------------------------------------
# Masking — prompt tokens must NOT contribute to the loss
# ---------------------------------------------------------------------------


class TestGRPOCriterionMasking:
    def test_prompt_logit_change_does_not_affect_loss(self):
        """Perturbing only prompt logits leaves the loss unchanged."""
        B, T, V, prompt_len = 1, 8, 20, 3
        torch.manual_seed(0)
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=prompt_len, advantages=torch.ones(B))

        # Patch: build a model wrapper that injects prompt-position noise
        class PerturbedPromptModel(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, input_ids, **kw):
                out = self.base(input_ids, **kw)
                logits = out["logits"].clone()
                logits[:, :prompt_len, :] += 100.0  # only prompt positions
                return {"logits": logits}

        crit = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0))
        loss_orig, _ = crit(model, batch)
        loss_perturbed, _ = crit(PerturbedPromptModel(model), batch)

        assert abs(loss_orig.item() - loss_perturbed.item()) < 1e-4

    def test_completion_logit_change_affects_loss(self):
        """Boosting the sampled completion token's logit changes the ratio and loss."""
        B, T, V, prompt_len = 1, 8, 20, 3
        torch.manual_seed(5)
        tokens = torch.randint(0, V, (B, T))
        model = _TinyLM(V)

        # Anchor old_logprobs to the base model: ratio=1, loss=-1 per completion token
        with torch.no_grad():
            base_logits = model(tokens)["logits"]
        base_lp = torch.log_softmax(base_logits[:, :-1, :], dim=-1)
        old_logprobs = (
            torch.gather(base_lp, -1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1).detach()
        )

        batch = _make_batch(B, T, V, prompt_len=prompt_len, advantages=torch.ones(B))
        batch["input_ids"] = tokens
        batch["old_logprobs"] = old_logprobs

        # Perturb: strongly boost only the sampled token at each completion position
        class BoostedCompletionModel(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, input_ids, **kw):
                out = self.base(input_ids, **kw)
                logits = out["logits"].clone()
                for t in range(prompt_len, T - 1):
                    logits[0, t, tokens[0, t + 1].item()] += 30.0
                return {"logits": logits}

        # eps=100 prevents clipping so ratio changes are fully visible
        crit = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0, eps=100.0))
        loss_orig, _ = crit(model, batch)
        loss_boosted, _ = crit(BoostedCompletionModel(model), batch)

        assert (
            abs(loss_orig.item() - loss_boosted.item()) > 0.01
        ), f"original={loss_orig:.4f}, boosted={loss_boosted:.4f}"

    def test_all_prompt_mask_gives_zero_loss(self):
        """All-zero completion_mask → zero loss (nothing to train on)."""
        B, T, V = 2, 8, 20
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=0, advantages=torch.ones(B))
        batch["completion_mask"] = torch.zeros(B, T, dtype=torch.long)
        crit = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0))
        loss, _ = crit(model, batch)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Policy ratio
# ---------------------------------------------------------------------------


class TestGRPOCriterionRatio:
    def test_ratio_one_when_policy_equals_old(self):
        """Current policy == old policy → ratio == 1 for all completion tokens."""
        B, T, V, prompt_len = 2, 8, 20, 2
        torch.manual_seed(0)
        model = _TinyLM(V)
        model.eval()
        batch = _make_batch(B, T, V, prompt_len=prompt_len, advantages=torch.ones(B))
        input_ids = batch["input_ids"]

        with torch.no_grad():
            logits = model(input_ids)["logits"]
        lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        old_logprobs = torch.gather(lp, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        batch["old_logprobs"] = old_logprobs.detach()

        crit = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0))
        _, exposed = crit(model, batch)
        assert exposed["ratio"].item() == pytest.approx(1.0, abs=0.01)

    def test_clip_at_1_plus_eps_for_large_ratio(self):
        """Very large ratio is clipped to 1+eps; loss equals -(1+eps)*adv."""
        B, T, V, prompt_len = 1, 6, 10, 1
        adv_value = 2.0
        eps = 0.2
        batch = _make_batch(
            B, T, V, prompt_len=prompt_len, advantages=torch.full((B,), adv_value)
        )
        # Force huge positive log-ratio: old ≈ -10, current ≈ -log(V)
        batch["old_logprobs"] = torch.full((B, T - 1), -10.0)
        logits = torch.zeros(B, T, V)  # uniform → log-prob = -log(V)

        crit = GRPOCriterion(GRPOCriterionConfig(eps=eps, kl_coeff=0.0))
        _, exposed = crit(lambda ids, **kw: {"logits": logits}, batch)

        expected = -(1 + eps) * adv_value
        assert exposed["policy_loss"].item() == pytest.approx(expected, rel=0.05)


# ---------------------------------------------------------------------------
# KL penalty
# ---------------------------------------------------------------------------


class TestGRPOCriterionKL:
    def test_kl_near_zero_when_policy_equals_ref(self):
        """π_θ == π_ref → KL ≈ 0 regardless of kl_coeff magnitude."""
        B, T, V, prompt_len = 2, 8, 20, 2
        torch.manual_seed(1)
        model = _TinyLM(V)
        model.eval()
        batch = _make_batch(B, T, V, prompt_len=prompt_len)
        with torch.no_grad():
            logits = model(batch["input_ids"])["logits"]
        lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        per_token = torch.gather(
            lp, -1, batch["input_ids"][:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        batch["old_logprobs"] = per_token.detach()
        batch["ref_logprobs"] = per_token.detach()

        crit = GRPOCriterion(GRPOCriterionConfig(kl_coeff=1.0))
        _, exposed = crit(model, batch)
        assert exposed["kl_div"].item() == pytest.approx(0.0, abs=1e-4)

    def test_kl_coeff_scales_total_loss(self):
        """Higher kl_coeff increases total loss when ref ≠ policy."""
        B, T, V, prompt_len = 2, 8, 20, 2
        batch = _make_batch(B, T, V, prompt_len=prompt_len, advantages=torch.zeros(B))
        # divergent ref
        batch["ref_logprobs"] = torch.zeros(B, T - 1)
        batch["old_logprobs"] = torch.full((B, T - 1), -5.0)
        model = _TinyLM(V)

        loss_low, _ = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.001))(model, batch)
        loss_high, _ = GRPOCriterion(GRPOCriterionConfig(kl_coeff=10.0))(model, batch)
        assert loss_high.item() > loss_low.item()


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestGRPOCriterionNumerical:
    @pytest.mark.parametrize("seed", range(10))
    def test_no_nan_random_inputs(self, seed):
        B, T, V = 4, 12, 50
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=3, advantages=torch.randn(B), seed=seed)
        loss, _ = GRPOCriterion(GRPOCriterionConfig())(model, batch)
        assert not torch.isnan(loss), f"NaN at seed={seed}"

    def test_empty_completion_mask_gives_zero(self):
        B, T, V = 2, 8, 20
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=0)
        batch["completion_mask"] = torch.zeros(B, T, dtype=torch.long)
        loss, _ = GRPOCriterion(GRPOCriterionConfig())(model, batch)
        assert not torch.isnan(loss)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("adv", [1000.0, -1000.0])
    def test_extreme_advantages_finite(self, adv):
        B, T, V = 2, 8, 20
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=2, advantages=torch.full((B,), adv))
        loss, _ = GRPOCriterion(GRPOCriterionConfig())(model, batch)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGRPOCriterionGradients:
    def test_gradients_reach_all_parameters(self):
        B, T, V, prompt_len = 2, 10, 20, 3
        torch.manual_seed(7)
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=prompt_len, advantages=torch.randn(B))
        loss, _ = GRPOCriterion(GRPOCriterionConfig())(model, batch)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    def test_zero_advantage_zeroes_policy_gradient(self):
        """With A=0 the PPO surrogate is zero → policy_loss exposed == 0."""
        B, T, V, prompt_len = 2, 8, 20, 2
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=prompt_len, advantages=torch.zeros(B))
        _, exposed = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0))(model, batch)
        assert exposed["policy_loss"].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Metric logging (verified via meters_group, not mocks)
# ---------------------------------------------------------------------------


class TestGRPOCriterionMetrics:
    def test_expected_metrics_are_logged(self):
        """All required metric names must appear in the meters after a forward pass."""
        B, T, V = 2, 8, 20
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=2, advantages=torch.randn(B))

        with meters_group("grpo_test", log_freq=1):
            GRPOCriterion(GRPOCriterionConfig())(model, batch)
            metrics = compute_meters("grpo_test")

        reset_meters("grpo_test")

        for key in ("loss", "policy_loss", "kl_div", "ratio", "num_tokens_per_step"):
            assert key in metrics, f"metric '{key}' not found in {list(metrics)}"

    def test_num_tokens_per_step_matches_completion_count(self):
        """The logged token count must equal the number of 1s in shift_mask."""
        B, T, V, prompt_len = 2, 8, 20, 3
        model = _TinyLM(V)
        batch = _make_batch(B, T, V, prompt_len=prompt_len)

        with meters_group("grpo_test2", log_freq=1):
            GRPOCriterion(GRPOCriterionConfig())(model, batch)
            metrics = compute_meters("grpo_test2")

        reset_meters("grpo_test2")

        shift_mask = batch["completion_mask"][:, 1:]
        expected = shift_mask.sum().item()
        assert metrics["num_tokens_per_step"] == pytest.approx(expected)
