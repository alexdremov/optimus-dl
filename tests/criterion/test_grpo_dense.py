"""Tests for GRPO debugging findings and dense (token-level) rewards."""

import torch
import pytest
from omegaconf import OmegaConf

from optimus_dl.modules.criterion.grpo import (
    GRPOCriterion,
    GRPOCriterionConfig,
)
from optimus_dl.modules.data.presets.arithmetic import (
    ArithmeticDataset,
    ArithmeticDatasetConfig,
)
from optimus_dl.modules.reward import build_reward_function
from optimus_dl.modules.reward.implementations.arithmetic import (
    ArithmeticReward,
    ArithmeticRewardConfig,
    ArithmeticTokenReward,
    ArithmeticTokenRewardConfig,
)


class TinyCausalModel(torch.nn.Module):
    """Minimal model returning logits; lets the criterion run standalone."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.proj = torch.nn.Linear(dim, vocab, bias=False)
        self.embed = torch.nn.Embedding(vocab, dim)

    def forward(self, input_ids, seq_lens=None, **kwargs):
        x = self.embed(input_ids)
        return {"logits": self.proj(x)}


def make_experience(B=2, G=4, T_prompt=3, T_comp=4, vocab=16, temp=1.0):
    torch.manual_seed(0)
    n = B * G
    T = T_prompt + T_comp
    input_ids = torch.randint(1, vocab, (n, T))
    completion_mask = torch.zeros(n, T, dtype=torch.long)
    completion_mask[:, T_prompt:] = 1
    seq_lens = torch.full((n,), T)
    old_logprobs = torch.randn(n, T - 1) - 2.0
    ref_logprobs = torch.randn(n, T - 1) - 2.0
    advantages = torch.randn(n)
    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "seq_lens": seq_lens,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
        "sampling_temperature": temp,
    }


# --------------------------------------------------------------------------- #
# Criterion: temperature consistency and dense advantages                      #
# --------------------------------------------------------------------------- #
def test_ratio_is_one_when_policy_matches_old_at_temperature():
    """With theta == old policy, ratio must be exactly 1 even for T != 1."""
    model = TinyCausalModel()
    batch = make_experience(temp=0.7)

    # Build old_logprobs from the model itself at T=0.7 (rollout condition).
    with torch.no_grad():
        logits = model(batch["input_ids"])["logits"][:, :-1]
        lp = torch.log_softmax(logits / 0.7, dim=-1)
        batch["old_logprobs"] = torch.gather(
            lp, -1, batch["input_ids"][:, 1:, None]
        ).squeeze(-1)

    criterion = GRPOCriterion(GRPOCriterionConfig(_name="grpo", kl_coeff=0.0))
    loss, exposed = criterion(model, batch)
    assert exposed["ratio"].item() == pytest.approx(1.0, abs=1e-5)


def test_dense_advantages_sum_to_scalar_grpo_advantage():
    """Row-sum of dense advantages == scalar GRPO advantage of that row."""
    B, G, T = 2, 4, 7
    n, Tm1 = B * G, T - 1
    total = torch.tensor([0.9, 0.1, 0.5, 0.2, 0.7, 0.3, 0.6, 0.4])
    grouped = total.view(B, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True) + 1e-8

    # Arbitrary per-token dense values over the 4 completion positions,
    # summing to `total`.
    d = torch.rand(n, 4)
    d *= (total / d.sum(dim=1)).view(-1, 1)

    comp_mask = torch.zeros(n, T, dtype=torch.long)
    comp_mask[:, 3:] = 1  # 4 completion tokens -> 4 shifted positions

    adv_2d = torch.zeros(n, Tm1)
    means = mean.expand(-1, G).reshape(-1)
    stds = std.expand(-1, G).reshape(-1)
    for row in range(n):
        cols = (comp_mask[row] == 1).nonzero(as_tuple=True)[0] - 1
        uniform = 0.0 - means[row] / 4  # no scalar part
        adv_2d[row, cols] = (d[row] + uniform) / stds[row]

    scalar_adv = ((grouped - mean) / std).view(-1)
    sums = adv_2d.sum(dim=1)
    assert torch.allclose(sums, scalar_adv, atol=1e-5)


def test_criterion_accepts_2d_advantages():
    model = TinyCausalModel()
    batch = make_experience()
    batch["advantages"] = torch.zeros(
        batch["input_ids"].size(0), batch["old_logprobs"].size(1)
    )
    criterion = GRPOCriterion(GRPOCriterionConfig(_name="grpo", kl_coeff=0.0))
    loss, _ = criterion(model, batch)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Dense arithmetic reward                                                      #
# --------------------------------------------------------------------------- #
class MockTokenizer:
    """Byte-faithful mock: each token is one character of the string."""

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(i) for i in ids if 0 <= i < 256)


def make_token_reward(**overrides):
    cfg = ArithmeticTokenRewardConfig(_name="arithmetic_token", **overrides)
    return ArithmeticTokenReward(cfg)


def test_token_reward_is_zero_at_sequence_level():
    reward = make_token_reward()
    r = reward(prompts=["1+1="], completions=["2"], answers=["2"])
    assert r.tolist() == [0.0]


def test_token_reward_exact_match_credits_number_span():
    reward = make_token_reward(digit_weight=0.5)
    ids = [ord(c) for c in "42"] + [257]  # '4','2',EOS
    rows = reward.token_rewards(
        prompts=["40+2="],
        completions=["42"],
        answers=["42"],
        completion_ids=[ids],
        tokenizer=MockTokenizer(),
    )
    v = rows[0]
    assert v.shape == (3,)
    assert v.sum().item() == pytest.approx(1.0)  # full reward_value
    assert v[0].item() > 0 and v[1].item() > 0
    assert v[2].item() > 0  # trailing stop token included


def test_token_reward_partial_match_skips_junk_tokens():
    reward = make_token_reward(digit_weight=0.5)
    # '7x' vs answer '87': '7' matches under best alignment -> partial credit
    # only on the digit token; junk 'x' is not a stop marker and gets nothing.
    ids = [ord("7"), ord("x"), 257]
    rows = reward.token_rewards(
        prompts=["80+7="],
        completions=["7x"],
        answers=["87"],
        completion_ids=[ids],
        tokenizer=MockTokenizer(),
    )
    v = rows[0]
    expected = 0.5 * (1 / 2)  # value=0.25, all on the single span token
    assert v[0].item() == pytest.approx(expected)
    assert v[1].item() == 0.0  # junk is neither span nor stop marker
    assert v[2].item() == 0.0  # EOS beyond junk is not chained into the span


def test_token_reward_unparseable_gets_uniform_penalty():
    reward = make_token_reward(penalty_value=-0.1)
    ids = [ord("x"), ord("y")]
    rows = reward.token_rewards(
        prompts=["1+1="],
        completions=["xy"],
        answers=["2"],
        completion_ids=[ids],
        tokenizer=MockTokenizer(),
    )
    assert rows[0].tolist() == pytest.approx([-0.1, -0.1])


def test_token_reward_registry_build():
    fn = build_reward_function(OmegaConf.create({"_name": "arithmetic_token"}))
    assert isinstance(fn, ArithmeticTokenReward)
    assert fn.supports_token_level()


def test_scalar_arithmetic_does_not_claim_token_level():
    reward = ArithmeticReward(ArithmeticRewardConfig(_name="arithmetic"))
    assert not reward.supports_token_level()


# --------------------------------------------------------------------------- #
# Dataset sanity for the few-shot SmolLM2 setup                                #
# --------------------------------------------------------------------------- #
def test_two_digit_stream_respects_bounds():
    ds = ArithmeticDataset(
        ArithmeticDatasetConfig(
            _name="preset_arithmetic",
            min_digits=2,
            max_digits=2,
            max_result=198,
            seed=42,
        )
    )
    for sample in [ds.next() for _ in range(50)]:
        a, b = sample["question"].split("+")
        assert 10 <= int(a) <= 99 and 10 <= int(b) <= 99
        assert int(sample["answer"]) <= 198
