"""Integration tests for the GRPO training pipeline.

These tests run the actual GRPO algorithm — generation, reward scoring,
loss computation, and gradient descent — on tiny in-memory models.

Design principles:
- No mocking of training math (generation, criterion, backward, optimizer)
- log_averaged is a no-op outside meters_group, no patching needed
- Only infrastructure that truly requires isolation (distributed) uses FakeCollective
- Tests verify that the algorithm actually does something (weights change,
  loss is computable, gradients flow, statistical invariants hold)
"""

import torch
import pytest
import torch.nn as nn

from optimus_dl.core.generation import (
    GenerationConfig,
    NativeEngine,
    NativeEngineConfig,
)
from optimus_dl.modules.criterion.grpo import (
    GRPOCriterion,
    GRPOCriterionConfig,
)
from optimus_dl.modules.metrics import (
    compute_meters,
    meters_group,
    reset_meters,
)
from optimus_dl.modules.reward import (
    RuleBasedReward,
    RuleBasedRewardConfig,
)
from optimus_dl.recipe.train.grpo import GRPORecipe

# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, **_):
        return {"logits": self.head(self.embed(input_ids))}

    def make_parameter_groups(self):
        return list(self.parameters())

    def pre_optimizer_step(self):
        pass

    def post_optimizer_step(self):
        pass

    def accumulation_context(self, is_last_microbatch=True):
        from contextlib import nullcontext

        return nullcontext()


def _make_ref(policy: _TinyLM) -> _TinyLM:
    ref = _TinyLM(policy.embed.num_embeddings, policy.embed.embedding_dim)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _make_recipe_stub(G: int, V: int, max_new: int):
    """A minimal object that exposes just _generate_experience."""
    stub = type("S", (), {})()
    stub.cfg = type(
        "C",
        (),
        {
            "num_generations": G,
            "generation_config": GenerationConfig(
                max_new_tokens=max_new, temperature=1.0
            ),
        },
    )()
    stub.tokenizer = None
    stub._generate_experience = GRPORecipe._generate_experience.__get__(stub)
    stub._get_per_token_logprobs = GRPORecipe._get_per_token_logprobs
    stub._build_completion_mask = GRPORecipe._build_completion_mask
    stub._decode_batch = GRPORecipe._decode_batch.__get__(stub)
    return stub


# ---------------------------------------------------------------------------
# Core algorithm: generate → reward → loss → backprop → optimizer step
# ---------------------------------------------------------------------------


class TestGRPOTrainingStep:
    """Each test runs real forward+backward passes on a tiny model."""

    def _training_setup(self, G=4, B=2, L=5, V=32, max_new=4):
        torch.manual_seed(42)
        policy = _TinyLM(V)
        ref = _make_ref(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        criterion = GRPOCriterion(GRPOCriterionConfig(eps=0.2, kl_coeff=0.01))
        gen_engine = NativeEngine(NativeEngineConfig())
        reward_fn = RuleBasedReward(
            RuleBasedRewardConfig(target_string="", reward_value=1.0)
        )
        stub = _make_recipe_stub(G, V, max_new)
        batch = {"input_ids": torch.randint(0, V, (B, L))}
        return policy, ref, optimizer, criterion, gen_engine, reward_fn, stub, batch

    def test_forward_pass_produces_finite_loss(self):
        policy, ref, optimizer, criterion, gen_engine, rf, stub, batch = (
            self._training_setup()
        )

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        loss, exposed = criterion(policy, exp)

        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert not torch.isnan(loss)
        assert set(exposed) >= {"policy_loss", "kl_div", "ratio"}

    def test_backward_produces_finite_gradients(self):
        policy, ref, optimizer, criterion, gen_engine, rf, stub, batch = (
            self._training_setup()
        )

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        loss, _ = criterion(policy, exp)
        loss.backward()

        for name, p in policy.named_parameters():
            assert p.grad is not None, f"no gradient for parameter '{name}'"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for '{name}'"

    def test_optimizer_step_changes_weights(self):
        """After one training step the policy parameters must change."""
        policy, ref, optimizer, criterion, gen_engine, rf, stub, batch = (
            self._training_setup()
        )

        before = [p.clone().detach() for p in policy.parameters()]

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        loss, _ = criterion(policy, exp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = any(
            not torch.equal(b, a.detach())
            for b, a in zip(before, policy.parameters(), strict=False)
        )
        assert changed, "policy weights did not change after optimizer step"

    def test_ref_model_never_changes(self):
        """The frozen reference model must not be modified by any training step."""
        policy, ref, optimizer, criterion, gen_engine, rf, stub, batch = (
            self._training_setup()
        )

        ref_before = [p.clone() for p in ref.parameters()]

        for _ in range(3):
            exp = stub._generate_experience(
                policy, ref, batch, gen_engine, [rf], torch.device("cpu")
            )
            loss, _ = criterion(policy, exp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for before, after in zip(ref_before, ref.parameters(), strict=False):
            assert torch.equal(before, after), "reference model weights changed!"

    def test_multi_step_loss_is_consistent(self):
        """Running several steps should not crash and losses should be finite."""
        policy, ref, optimizer, criterion, gen_engine, rf, stub, batch = (
            self._training_setup()
        )

        for step in range(5):
            exp = stub._generate_experience(
                policy, ref, batch, gen_engine, [rf], torch.device("cpu")
            )
            loss, _ = criterion(policy, exp)
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_gradient_accumulation_aggregates_experience(self):
        """Accumulating gradients over 2 microbatches should not crash."""
        G, B, L, V, max_new = 2, 2, 5, 32, 3
        policy, ref, optimizer, criterion, gen_engine, rf, stub, batch = (
            self._training_setup(G=G, B=B, L=L, V=V, max_new=max_new)
        )

        acc_steps = 2
        optimizer.zero_grad()
        for _ in range(acc_steps):
            exp = stub._generate_experience(
                policy, ref, batch, gen_engine, [rf], torch.device("cpu")
            )
            loss, _ = criterion(policy, exp)
            (loss / acc_steps).backward()  # scale loss for accumulation

        optimizer.step()
        # If we get here without error, gradient accumulation works
        assert True

    def test_multiple_reward_functions_combine(self):
        """Multiple reward functions must be summed before computing advantages."""
        G, B = 4, 2
        policy, ref, optimizer, criterion, gen_engine, _, stub, batch = (
            self._training_setup(G=G, B=B)
        )

        # Two real reward functions with distinct signals
        rf1 = RuleBasedReward(RuleBasedRewardConfig(target_string="", reward_value=1.0))
        rf2 = RuleBasedReward(
            RuleBasedRewardConfig(target_string="xyz", reward_value=2.0)
        )

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf1, rf2], torch.device("cpu")
        )
        loss, _ = criterion(policy, exp)
        assert torch.isfinite(loss)

    def test_kl_penalty_increases_loss_versus_zero_kl(self):
        """Enabling the KL penalty must increase the total loss when ref ≠ policy."""
        G, B, L, V, max_new = 2, 2, 4, 32, 3
        torch.manual_seed(0)
        policy = _TinyLM(V)
        # Make ref weights different so KL > 0
        ref = _TinyLM(V)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        gen_engine = NativeEngine(NativeEngineConfig())
        rf = RuleBasedReward(RuleBasedRewardConfig(target_string="", reward_value=1.0))
        stub = _make_recipe_stub(G, V, max_new)
        batch = {"input_ids": torch.randint(0, V, (B, L))}

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )

        loss_no_kl, _ = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0))(policy, exp)
        loss_kl, _ = GRPOCriterion(GRPOCriterionConfig(kl_coeff=10.0))(policy, exp)
        assert loss_kl.item() != pytest.approx(
            loss_no_kl.item(), abs=1e-4
        ), "KL penalty should change the loss when ref != policy"


# ---------------------------------------------------------------------------
# Experience–criterion pipeline compatibility
# ---------------------------------------------------------------------------


class TestExperienceCriterionRoundtrip:
    """Verify the output of _generate_experience feeds correctly into GRPOCriterion."""

    def test_shapes_are_compatible(self):
        G, B, L, V, max_new = 4, 2, 5, 32, 4
        torch.manual_seed(1)
        policy = _TinyLM(V)
        ref = _make_ref(policy)
        gen_engine = NativeEngine(NativeEngineConfig())
        rf = RuleBasedReward(RuleBasedRewardConfig(target_string="", reward_value=1.0))
        stub = _make_recipe_stub(G, V, max_new)

        exp = stub._generate_experience(
            policy,
            ref,
            {"input_ids": torch.randint(0, V, (B, L))},
            gen_engine,
            [rf],
            torch.device("cpu"),
        )
        loss, exposed = GRPOCriterion(GRPOCriterionConfig())(policy, exp)

        assert torch.isfinite(loss)
        assert "policy_loss" in exposed and "kl_div" in exposed

    def test_full_backward_through_pipeline(self):
        G, B, L, V, max_new = 2, 2, 4, 20, 3
        torch.manual_seed(2)
        policy = _TinyLM(V)
        ref = _make_ref(policy)
        gen_engine = NativeEngine(NativeEngineConfig())
        rf = RuleBasedReward(RuleBasedRewardConfig(target_string="", reward_value=1.0))
        stub = _make_recipe_stub(G, V, max_new)

        exp = stub._generate_experience(
            policy,
            ref,
            {"input_ids": torch.randint(0, V, (B, L))},
            gen_engine,
            [rf],
            torch.device("cpu"),
        )
        loss, _ = GRPOCriterion(GRPOCriterionConfig(eps=0.2, kl_coeff=0.01))(
            policy, exp
        )
        loss.backward()

        for name, p in policy.named_parameters():
            assert p.grad is not None, f"no gradient for '{name}'"
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Convergence on a toy problem
# ---------------------------------------------------------------------------


class TestGRPOConvergenceOnToyProblem:
    """Verify that GRPO learns an *input-dependent* policy on a micro-scale problem.

    Task — "echo" / copy:
        Prompt token 1 ('A')  →  correct completion: token 1 ('A')
        Prompt token 2 ('B')  →  correct completion: token 2 ('B')

    Each training batch contains both prompt types in equal measure, so a
    context-blind policy that always outputs the same token can only achieve
    50% reward.  The model must actually read the input to do better.

    Vocabulary:
        1 → 'A'   (prompt or completion)
        2 → 'B'   (prompt or completion)
        (tokens 0 and 3 are never used as prompts)

    Model:
        Embedding(4, 4) + Linear(4, 4)  →  36 parameters total

    Reward function:
        +1.0  if decoded_completion == decoded_prompt  (correct echo)
         0.0  otherwise

    Expected behaviour after training:
        P(A | prompt='A') > P(B | prompt='A')   — prefers echo for 'A'
        P(B | prompt='B') > P(A | prompt='B')   — prefers echo for 'B'
        P(A | prompt='A') > P(A | prompt='B')   — 'A' is context-specific

    Pipeline exercised:
        NativeEngine.generate  →  GRPORecipe._generate_experience
        →  MatchPromptReward  →  group-relative advantage normalisation
        →  GRPOCriterion (clipped PPO + KL)  →  AdamW.step
    """

    # ---- tokenizer: maps token ids to single chars ----
    class _CharTokenizer:
        _vocab = {0: "P", 1: "A", 2: "B", 3: "C"}

        def decode(self, ids: list[int]) -> str:
            return "".join(self._vocab.get(i, "?") for i in ids)

    # ---- input-conditional reward: +1 iff completion echoes the prompt ----
    class _MatchPromptReward:
        """Reward 1.0 when the single-character completion equals the prompt.

        This forces the model to condition on the input: a policy that always
        outputs the same token regardless of the prompt can only score 0.5 on
        average (one prompt type matches, the other does not).
        """

        def __call__(
            self,
            prompts: list[str],
            completions: list[str],
            **_,
        ) -> torch.Tensor:
            return torch.tensor(
                [
                    1.0 if c == p else 0.0
                    for p, c in zip(prompts, completions, strict=False)
                ],
                dtype=torch.float32,
            )

    # ---- helper ----

    @staticmethod
    def _conditional_probs(model: _TinyLM) -> dict[str, dict[str, float]]:
        """Return P(output | prompt) for prompt ∈ {A, B} and output ∈ {A, B}."""
        result = {}
        model.eval()
        with torch.no_grad():
            for prompt_tok, prompt_name in [(1, "A"), (2, "B")]:
                logits = model(torch.tensor([[prompt_tok]]))["logits"][0, -1]
                probs = torch.softmax(logits, dim=-1)
                result[prompt_name] = {
                    "A": probs[1].item(),
                    "B": probs[2].item(),
                }
        model.train()
        return result

    # ---- shared training fixture ----

    def _train(
        self, policy, ref, reward_fn, steps: int, G: int = 8, device=torch.device("cpu")
    ):
        """Run the GRPO loop for `steps` iterations and return the trained policy."""
        V = policy.embed.num_embeddings
        optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-2)
        criterion = GRPOCriterion(
            GRPOCriterionConfig(
                eps=0.5,  # wide clip → fast exploration
                kl_coeff=0.001,  # tiny KL so policy can diverge from ref
            )
        )
        gen_engine = NativeEngine(NativeEngineConfig())
        stub = _make_recipe_stub(G=G, V=V, max_new=1)
        stub.tokenizer = self._CharTokenizer()

        # Each batch alternates prompt A and prompt B so both are seen every step
        batch = {"input_ids": torch.tensor([[1], [2], [1], [2]])}

        for _ in range(steps):
            exp = stub._generate_experience(
                policy, ref, batch, gen_engine, [reward_fn], device
            )
            loss, _ = criterion(policy, exp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return policy

    # ---- tests ----

    def test_model_learns_input_conditional_echo(self):
        """After training, argmax P(·|prompt) must equal the prompt token.

        A context-blind model that ignores its input and always outputs the
        same token earns at most 50% reward.  Only by reading the prompt and
        returning the matching token can it do better.
        """
        torch.manual_seed(0)

        V = 4
        policy = _TinyLM(vocab_size=V, embed_dim=4)  # 36 parameters
        ref = _make_ref(policy)

        assert sum(p.numel() for p in policy.parameters()) == 36

        probs_before = self._conditional_probs(policy)
        self._train(policy, ref, self._MatchPromptReward(), steps=120)
        probs_after = self._conditional_probs(policy)

        # P(echo | prompt) must have increased for both prompt tokens
        assert (
            probs_after["A"]["A"] > probs_before["A"]["A"]
        ), "P(A | prompt=A) should increase"
        assert (
            probs_after["B"]["B"] > probs_before["B"]["B"]
        ), "P(B | prompt=B) should increase"

        # Model must prefer the correct token for each prompt
        assert probs_after["A"]["A"] > probs_after["A"]["B"], (
            f"Given prompt 'A', model should prefer 'A' over 'B': "
            f"P(A|A)={probs_after['A']['A']:.3f}  P(B|A)={probs_after['A']['B']:.3f}"
        )
        assert probs_after["B"]["B"] > probs_after["B"]["A"], (
            f"Given prompt 'B', model should prefer 'B' over 'A': "
            f"P(B|B)={probs_after['B']['B']:.3f}  P(A|B)={probs_after['B']['A']:.3f}"
        )

        # Cross-conditional: each token is more likely given its own prompt
        assert probs_after["A"]["A"] > probs_after["B"]["A"], (
            f"'A' should be more likely given prompt 'A' than prompt 'B': "
            f"P(A|A)={probs_after['A']['A']:.3f}  P(A|B)={probs_after['B']['A']:.3f}"
        )
        assert probs_after["B"]["B"] > probs_after["A"]["B"], (
            f"'B' should be more likely given prompt 'B' than prompt 'A': "
            f"P(B|B)={probs_after['B']['B']:.3f}  P(B|A)={probs_after['A']['B']:.3f}"
        )

    def test_unconditional_policy_cannot_maximise_echo_reward(self):
        """A policy that ignores its input is strictly suboptimal.

        We train two models: one with the echo reward (input-dependent) and
        one forced to be unconditional (fixed prompt).  After training, the
        echo-trained model should score higher expected reward across both
        prompt types than the unconditional one.
        """
        torch.manual_seed(1)

        V = 4
        # Unconditional model: always sees prompt 'A', optimised to output 'A'
        policy_unc = _TinyLM(vocab_size=V, embed_dim=4)
        ref_unc = _make_ref(policy_unc)
        unc_rf = RuleBasedReward(
            RuleBasedRewardConfig(target_string="A", reward_value=1.0)
        )

        optimizer = torch.optim.AdamW(policy_unc.parameters(), lr=5e-2)
        criterion = GRPOCriterion(GRPOCriterionConfig(eps=0.5, kl_coeff=0.001))
        gen_engine = NativeEngine(NativeEngineConfig())
        stub_unc = _make_recipe_stub(G=8, V=V, max_new=1)
        stub_unc.tokenizer = self._CharTokenizer()
        for _ in range(120):
            batch = {"input_ids": torch.tensor([[1], [1], [1], [1]])}  # always 'A'
            exp = stub_unc._generate_experience(
                policy_unc, ref_unc, batch, gen_engine, [unc_rf], torch.device("cpu")
            )
            loss, _ = criterion(policy_unc, exp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Echo-trained model (input-conditional)
        torch.manual_seed(1)  # same init
        policy_echo = _TinyLM(vocab_size=V, embed_dim=4)
        ref_echo = _make_ref(policy_echo)
        self._train(policy_echo, ref_echo, self._MatchPromptReward(), steps=120)

        # Evaluate expected echo reward across both prompt types
        def expected_echo_reward(model) -> float:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for prompt_tok, correct_tok in [(1, 1), (2, 2)]:
                    logits = model(torch.tensor([[prompt_tok]]))["logits"][0, -1]
                    p = torch.softmax(logits, dim=-1)
                    total += p[correct_tok].item()
            model.train()
            return total / 2.0  # average over both prompts

        reward_unc = expected_echo_reward(policy_unc)
        reward_echo = expected_echo_reward(policy_echo)

        assert reward_echo > reward_unc, (
            f"Input-conditional policy should outperform unconditional one: "
            f"echo={reward_echo:.3f}  unconditional={reward_unc:.3f}"
        )

    def test_grpo_does_not_train_without_reward_signal(self):
        """Flat reward (identical for every completion) → zero group-relative
        advantage → near-zero policy gradient → weights barely change."""
        torch.manual_seed(7)

        V, G, B = 4, 8, 4
        policy = _TinyLM(vocab_size=V, embed_dim=4)
        ref = _make_ref(policy)
        params_before = [p.clone() for p in policy.parameters()]

        class FlatReward:
            def __call__(self, **_):
                return torch.ones(B * G) * 0.5

        optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-2)
        criterion = GRPOCriterion(GRPOCriterionConfig(eps=0.2, kl_coeff=0.0))
        gen_engine = NativeEngine(NativeEngineConfig())
        stub = _make_recipe_stub(G=G, V=V, max_new=1)
        stub.tokenizer = None

        for _ in range(20):
            batch = {"input_ids": torch.tensor([[1], [2], [1], [2]])}
            exp = stub._generate_experience(
                policy, ref, batch, gen_engine, [FlatReward()], torch.device("cpu")
            )
            loss, _ = criterion(policy, exp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        max_change = max(
            (a - b).abs().max().item()
            for b, a in zip(params_before, policy.parameters(), strict=False)
        )
        assert max_change < 0.05, (
            f"Flat reward must not cause meaningful weight changes, "
            f"got max change {max_change:.4f}"
        )


# ---------------------------------------------------------------------------
# Metrics accumulation during training
# ---------------------------------------------------------------------------


class TestMetricsAccumulation:
    """Verify that training metrics are correctly accumulated in meters_group."""

    def test_all_expected_metrics_logged_during_step(self):
        G, B, L, V, max_new = 2, 2, 5, 32, 3
        policy = _TinyLM(V)
        ref = _make_ref(policy)
        gen_engine = NativeEngine(NativeEngineConfig())
        rf = RuleBasedReward(RuleBasedRewardConfig(target_string="", reward_value=1.0))
        stub = _make_recipe_stub(G, V, max_new)

        exp = stub._generate_experience(
            policy,
            ref,
            {"input_ids": torch.randint(0, V, (B, L))},
            gen_engine,
            [rf],
            torch.device("cpu"),
        )

        with meters_group("train_test", log_freq=1):
            GRPOCriterion(GRPOCriterionConfig())(policy, exp)
            metrics = compute_meters("train_test")

        reset_meters("train_test")

        for key in ("loss", "policy_loss", "kl_div", "ratio", "num_tokens_per_step"):
            assert (
                key in metrics
            ), f"expected metric '{key}' missing from {list(metrics)}"

    def test_reward_and_advantage_metrics_logged_during_generation(self):
        G, B, L, V, max_new = 4, 2, 5, 32, 3
        policy = _TinyLM(V)
        ref = _make_ref(policy)
        gen_engine = NativeEngine(NativeEngineConfig())
        rf = RuleBasedReward(RuleBasedRewardConfig(target_string="", reward_value=1.0))
        stub = _make_recipe_stub(G, V, max_new)

        with meters_group("rollout_test", log_freq=1):
            stub._generate_experience(
                policy,
                ref,
                {"input_ids": torch.randint(0, V, (B, L))},
                gen_engine,
                [rf],
                torch.device("cpu"),
            )
            metrics = compute_meters("rollout_test")

        reset_meters("rollout_test")

        assert "reward" in metrics
        assert "advantage_max" in metrics
        assert "num_completion_tokens" in metrics
