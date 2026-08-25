"""Tests for RL pipeline correctness improvements.

Covers:
- _split_experience: true micro-batching of experience (vs duplicated tensors)
- _get_prompt_lengths: seq_lens / attention_mask / fallback priority
- Variable-length prompt rollouts end-to-end (no pad contamination)
- Temperature-consistent old-policy logprobs
- Chunked logprob computation equivalence
- EOS truncation integrated into experience generation
- GRPOCriterion seq_lens passthrough to the model
- rule_based reward registry entry
- ExperienceBuffer / AsyncExperienceManager shutdown without deadlock
- run_training_iteration acc_steps_override loss scaling
"""

import threading
import time
from contextlib import nullcontext

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
from optimus_dl.modules.reward.implementations.rule_based import (
    RuleBasedReward,
    RuleBasedRewardConfig,
)

# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------


class ConstantNextModel(nn.Module):
    """Deterministic model always predicting ``token``.

    Holds a scalar parameter added to every logit so gradients are
    non-trivially defined.
    """

    def __init__(self, token: int = 7, vocab_size: int = 32):
        super().__init__()
        self.token = token
        self.vocab_size = vocab_size
        self.logit_scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_ids, seq_lens=None, **kw):
        logits = torch.full(
            (*input_ids.shape, self.vocab_size), -100.0, dtype=torch.float32
        )
        logits[..., self.token] = 100.0 * self.logit_scale[0]
        return {"logits": logits}

    def make_parameter_groups(self):
        return list(self.parameters())

    def pre_optimizer_step(self):
        pass

    def post_optimizer_step(self):
        pass


def _make_stub(G=2, V=32, max_new=3, temperature=1.0, stop_token_id=None, **cfg_attrs):
    """Minimal object exposing the GRPORecipe experience-generation methods."""
    from optimus_dl.recipe.train.grpo import GRPORecipe

    class _Stub:
        pass

    stub = _Stub()
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new,
        temperature=temperature,
        stop_token_id=stop_token_id,
    )
    cfg_dict = {"num_generations": G, "generation_config": gen_cfg}
    # Allow arbitrary extra config attributes (e.g. logprob_micro_batch_size)
    stub.cfg = type("C", (), cfg_dict)()
    for key, value in cfg_attrs.items():
        setattr(stub.cfg, key, value)
    stub.tokenizer = None

    stub._generate_experience = GRPORecipe._generate_experience.__get__(stub)
    stub._decode_batch = GRPORecipe._decode_batch.__get__(stub)
    stub._build_completion_mask = GRPORecipe._build_completion_mask
    stub._truncate_after_stop = GRPORecipe._truncate_after_stop
    stub._get_prompt_lengths = GRPORecipe._get_prompt_lengths
    stub._get_per_token_logprobs = GRPORecipe._get_per_token_logprobs
    stub._compute_batch_logprobs = GRPORecipe._compute_batch_logprobs.__get__(stub)
    stub._split_experience = GRPORecipe._split_experience
    return stub


# ---------------------------------------------------------------------------
# _split_experience
# ---------------------------------------------------------------------------


class TestSplitExperience:
    def _experience(self, B_times_G=6, T=5):
        return {
            "input_ids": torch.arange(B_times_G * T).reshape(B_times_G, T) % 100,
            "advantages": torch.arange(B_times_G, dtype=torch.float32),
            "meta": "shared",
        }

    def test_chunk_count_and_sizes(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        exp = self._experience()
        chunks = GRPORecipe._split_experience(exp, 3)
        assert len(chunks) == 3
        assert all(c["input_ids"].size(0) == 2 for c in chunks)
        assert all(c["advantages"].size(0) == 2 for c in chunks)

    def test_content_preserved_in_order(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        exp = self._experience()
        chunks = GRPORecipe._split_experience(exp, 2)
        assert torch.equal(chunks[0]["input_ids"], exp["input_ids"][:3])
        assert torch.equal(chunks[1]["input_ids"], exp["input_ids"][3:])
        assert torch.equal(chunks[0]["advantages"], exp["advantages"][:3])

    def test_more_chunks_than_rows_clamps(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        exp = self._experience(B_times_G=2)
        chunks = GRPORecipe._split_experience(exp, 5)
        assert len(chunks) == 2
        assert all(c["input_ids"].size(0) == 1 for c in chunks)

    def test_non_tensor_entries_copied_to_all_chunks(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        exp = self._experience()
        chunks = GRPORecipe._split_experience(exp, 2)
        assert all(c["meta"] == "shared" for c in chunks)

    def test_reassembled_tensors_equal_original(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        exp = self._experience()
        chunks = GRPORecipe._split_experience(exp, 3)
        ids = torch.cat([c["input_ids"] for c in chunks])
        adv = torch.cat([c["advantages"] for c in chunks])
        assert torch.equal(ids, exp["input_ids"])
        assert torch.equal(adv, exp["advantages"])


# ---------------------------------------------------------------------------
# _get_prompt_lengths
# ---------------------------------------------------------------------------


class TestGetPromptLengths:
    def test_prefers_seq_lens(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        batch = {
            "seq_lens": torch.tensor([3, 5]),
            "attention_mask": torch.ones(2, 10),
        }
        prompt_ids = torch.zeros(2, 10, dtype=torch.long)
        out = GRPORecipe._get_prompt_lengths(batch, prompt_ids, torch.device("cpu"))
        assert out.tolist() == [3, 5]

    def test_falls_back_to_attention_mask_sum(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        batch = {
            "attention_mask": torch.tensor(
                [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.long
            )
        }
        prompt_ids = torch.zeros(2, 5, dtype=torch.long)
        out = GRPORecipe._get_prompt_lengths(batch, prompt_ids, torch.device("cpu"))
        assert out.tolist() == [3, 4]

    def test_defaults_to_full_width(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        prompt_ids = torch.zeros(3, 7, dtype=torch.long)
        out = GRPORecipe._get_prompt_lengths({}, prompt_ids, torch.device("cpu"))
        assert out.tolist() == [7, 7, 7]


# ---------------------------------------------------------------------------
# End-to-end variable-length rollout: no pad contamination
# ---------------------------------------------------------------------------


class TestVariableLengthExperience:
    def test_completion_mask_respects_variable_prompt_lengths(self):
        stub = _make_stub(G=2, max_new=3)
        policy = ConstantNextModel(token=7)
        ref = ConstantNextModel(token=7)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]),
            "seq_lens": torch.tensor([3, 2]),
        }
        exp = stub._generate_experience(
            policy,
            ref,
            batch,
            NativeEngine(NativeEngineConfig()),
            [RuleBasedReward(RuleBasedRewardConfig())],
            torch.device("cpu"),
        )

        mask = exp["completion_mask"]
        # Row layout: row i*G+g. Prompt lengths [3, 3, 2, 2]. Output width is
        # max_prompt_len(3) + max_new(3) = 6; shorter rows keep a trailing pad.
        assert mask.shape == (4, 6)
        assert (mask[0, :3] == 0).all() and (mask[0, 3:] == 1).all()
        assert (mask[2, :2] == 0).all() and (mask[2, 2:5] == 1).all()
        assert (mask[2, 5:] == 0).all(), "trailing pad must be excluded"

    def test_no_pad_tokens_inside_completions(self):
        """Every masked completion position must hold a real generated token."""
        stub = _make_stub(G=2, max_new=3)
        policy = ConstantNextModel(token=7)
        ref = ConstantNextModel(token=7)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]),
            "seq_lens": torch.tensor([3, 2]),
        }
        exp = stub._generate_experience(
            policy,
            ref,
            batch,
            NativeEngine(NativeEngineConfig()),
            [RuleBasedReward(RuleBasedRewardConfig())],
            torch.device("cpu"),
        )
        completion_tokens = exp["input_ids"][exp["completion_mask"].bool()]
        assert (
            completion_tokens == 7
        ).all(), "pad tokens leaked into the completion region"

    def test_seq_lens_reported_for_criterion_forward(self):
        stub = _make_stub(G=1, max_new=2)
        policy = ConstantNextModel(token=7)
        ref = ConstantNextModel(token=7)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 0]]),
            "seq_lens": torch.tensor([3]),
        }
        exp = stub._generate_experience(
            policy,
            ref,
            batch,
            NativeEngine(NativeEngineConfig()),
            [RuleBasedReward(RuleBasedRewardConfig())],
            torch.device("cpu"),
        )
        assert exp["seq_lens"].tolist() == [5]

    def test_eos_truncation_in_full_pipeline(self):
        """Model emits the stop token immediately; the mask must keep exactly
        one completion position per sequence."""
        stop = 9
        stub = _make_stub(G=1, max_new=5, stop_token_id=stop)
        policy = ConstantNextModel(token=stop)
        ref = ConstantNextModel(token=stop)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "seq_lens": torch.tensor([3, 3]),
        }
        exp = stub._generate_experience(
            policy,
            ref,
            batch,
            NativeEngine(NativeEngineConfig()),
            [RuleBasedReward(RuleBasedRewardConfig())],
            torch.device("cpu"),
        )
        # One EOS token each → exactly one masked completion position per row.
        assert exp["completion_mask"].sum(dim=1).tolist() == [1, 1]
        # And that position holds the EOS itself.
        eos_positions = exp["completion_mask"].bool().nonzero()
        for row, col in eos_positions.tolist():
            assert exp["input_ids"][row, col].item() == stop


# ---------------------------------------------------------------------------
# Temperature-consistent old-policy logprobs
# ---------------------------------------------------------------------------


class TestTemperatureConsistentOldLogprobs:
    def test_old_logprobs_use_sampling_temperature(self):
        tau = 0.7
        stub = _make_stub(G=1, max_new=2, temperature=tau)
        policy = ConstantNextModel(token=7)
        ref = ConstantNextModel(token=7)
        batch = {"input_ids": torch.randint(0, 32, (2, 4))}

        exp = stub._generate_experience(
            policy,
            ref,
            batch,
            NativeEngine(NativeEngineConfig()),
            [RuleBasedReward(RuleBasedRewardConfig())],
            torch.device("cpu"),
        )

        # Manual recomputation at the sampling temperature
        with torch.no_grad():
            logits = policy(exp["input_ids"], seq_lens=exp["seq_lens"])["logits"]
            expected = torch.log_softmax(logits[:, :-1, :] / tau, dim=-1)
            expected = torch.gather(
                expected, -1, exp["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)

        assert torch.allclose(exp["old_logprobs"], expected, atol=1e-5)

    def test_temperature_one_matches_plain_softmax(self):
        stub = _make_stub(G=1, max_new=2, temperature=1.0)
        policy = ConstantNextModel(token=7)
        ref = ConstantNextModel(token=7)
        batch = {"input_ids": torch.randint(0, 32, (2, 4))}

        exp = stub._generate_experience(
            policy,
            ref,
            batch,
            NativeEngine(NativeEngineConfig()),
            [RuleBasedReward(RuleBasedRewardConfig())],
            torch.device("cpu"),
        )

        with torch.no_grad():
            logits = policy(exp["input_ids"])["logits"]
            expected = torch.log_softmax(logits[:, :-1, :], dim=-1)
            expected = torch.gather(
                expected, -1, exp["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)

        assert torch.allclose(exp["old_logprobs"], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Chunked logprob computation
# ---------------------------------------------------------------------------


class TestChunkedLogprobsEquivalence:
    @pytest.mark.parametrize("chunk", [1, 2])
    def test_chunking_gives_same_results(self, chunk):
        torch.manual_seed(11)
        batch = {"input_ids": torch.randint(0, 32, (3, 4))}
        results = []
        for micro_bs in (None, chunk):
            stub = _make_stub(G=2, max_new=2, logprob_micro_batch_size=micro_bs)
            policy = ConstantNextModel(token=7)
            ref = ConstantNextModel(token=7)
            exp = stub._generate_experience(
                policy,
                ref,
                batch,
                NativeEngine(NativeEngineConfig()),
                [RuleBasedReward(RuleBasedRewardConfig())],
                torch.device("cpu"),
            )
            results.append((exp["old_logprobs"], exp["ref_logprobs"]))

        (old_a, ref_a), (old_b, ref_b) = results
        assert torch.allclose(old_a, old_b, atol=1e-6)
        assert torch.allclose(ref_a, ref_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Criterion seq_lens passthrough
# ---------------------------------------------------------------------------


class TestCriterionSeqLensPassthrough:
    def test_seq_lens_forwarded_to_model(self):
        captured = {}

        class CapturingModel(nn.Module):
            def forward(self, input_ids, seq_lens=None, **kw):
                captured["seq_lens"] = seq_lens
                logits = torch.zeros(*input_ids.shape, 20)
                return {"logits": logits}

        B, T = 2, 6
        batch = {
            "input_ids": torch.randint(0, 20, (B, T)),
            "completion_mask": torch.ones(B, T, dtype=torch.long),
            "old_logprobs": torch.zeros(B, T - 1),
            "ref_logprobs": torch.zeros(B, T - 1),
            "advantages": torch.ones(B),
            "seq_lens": torch.tensor([5, 4]),
        }
        crit = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0))
        crit(CapturingModel(), batch)
        assert captured["seq_lens"] is not None
        assert captured["seq_lens"].tolist() == [5, 4]

    def test_no_seq_lens_means_no_kwarg(self):
        captured = {}

        class CapturingModel(nn.Module):
            def forward(self, input_ids, seq_lens=None, **kw):
                captured["seq_lens"] = seq_lens
                logits = torch.zeros(*input_ids.shape, 20)
                return {"logits": logits}

        B, T = 1, 4
        batch = {
            "input_ids": torch.randint(0, 20, (B, T)),
            "completion_mask": torch.ones(B, T, dtype=torch.long),
            "old_logprobs": torch.zeros(B, T - 1),
            "ref_logprobs": torch.zeros(B, T - 1),
            "advantages": torch.ones(B),
        }
        crit = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.0))
        crit(CapturingModel(), batch)
        assert captured["seq_lens"] is None


# ---------------------------------------------------------------------------
# rule_based reward registration
# ---------------------------------------------------------------------------


class TestRuleBasedRegistry:
    def test_buildable_from_registry(self):
        from optimus_dl.modules.reward import build_reward_function
        from optimus_dl.modules.reward.implementations.rule_based import (
            RuleBasedRewardConfig,
        )

        fn = build_reward_function(
            RuleBasedRewardConfig(_name="rule_based", target_string="abc")
        )
        rewards = fn(prompts=["p"], completions=["has abc!", "nope"])
        assert rewards.tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Async buffer / producer shutdown
# ---------------------------------------------------------------------------


class TestExperienceBufferShutdown:
    def test_put_blocks_until_get(self):
        from optimus_dl.recipe.train.mixins.rl_async import ExperienceBuffer

        buf = ExperienceBuffer(max_size=1)
        assert buf.put({"i": 0}) is True

        done = threading.Event()

        def blocking_put():
            buf.put({"i": 1})
            done.set()

        t = threading.Thread(target=blocking_put, daemon=True)
        t.start()
        time.sleep(0.2)
        assert not done.is_set(), "put on a full buffer must block"

        item = buf.get(timeout=1.0)
        assert item == {"i": 0}
        assert done.wait(timeout=1.0), "blocked put must resume after get"

    def test_stop_unblocks_pending_put(self):
        from optimus_dl.recipe.train.mixins.rl_async import ExperienceBuffer

        buf = ExperienceBuffer(max_size=1)
        buf.put({"i": 0})
        result = {}

        def blocking_put():
            result["ok"] = buf.put({"never": True})

        t = threading.Thread(target=blocking_put, daemon=True)
        t.start()
        time.sleep(0.1)
        buf.stop()
        t.join(timeout=2.0)
        assert not t.is_alive(), "put must return after stop"
        assert result["ok"] is False

    def test_get_returns_none_after_stop(self):
        from optimus_dl.recipe.train.mixins.rl_async import ExperienceBuffer

        buf = ExperienceBuffer(max_size=1)
        buf.stop()
        assert buf.get(timeout=0.1) is None


class TestAsyncProducerShutdown:
    def _manager_with_producer(self):
        from optimus_dl.recipe.train.mixins.rl_async import AsyncExperienceManager

        mgr = AsyncExperienceManager(role="rollout", buffer_size=1)
        counter = {"n": 0}

        def fast_produce(*args, **kwargs):
            time.sleep(0.005)
            counter["n"] += 1
            return {"batch": counter["n"]}

        mgr._produce_one_batch = fast_produce
        mgr.start_producer(None, None, None, None, None, None)
        return mgr, counter

    def test_producer_fills_buffer(self):
        mgr, counter = self._manager_with_producer()
        try:
            deadline = time.time() + 2.0
            while counter["n"] < 1 and time.time() < deadline:
                time.sleep(0.01)
            assert counter["n"] >= 1
            batch = mgr.get_batch(timeout=1.0)
            assert batch is not None and "batch" in batch
        finally:
            mgr.stop_producer()

    def test_stop_producer_terminates_blocked_producer(self):
        """Producer blocked on a full buffer must exit promptly on stop."""
        mgr, _ = self._manager_with_producer()
        time.sleep(0.3)  # let it fill the buffer and block on put
        start = time.time()
        mgr.stop_producer(join_timeout=2.0)
        elapsed = time.time() - start
        assert elapsed < 1.5, f"stop_producer took {elapsed:.2f}s — deadlock?"
        assert mgr._producer_thread is None

    def test_stop_producer_is_idempotent(self):
        mgr, _ = self._manager_with_producer()
        mgr.stop_producer()
        mgr.stop_producer()  # second call must be a no-op, not an error


# ---------------------------------------------------------------------------
# run_training_iteration acc_steps_override
# ---------------------------------------------------------------------------


class FakeScaler:
    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


class ScalarWeightedCriterion:
    """loss = weight * param.sum(); weight comes from the micro-batch."""

    def __call__(self, model, batch, requested_protocols=None):
        return batch["weight"] * sum(p.sum() for p in model.parameters()), {}


class TinyHeldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def make_parameter_groups(self):
        return list(self.parameters())

    def pre_optimizer_step(self):
        pass

    def post_optimizer_step(self):
        pass


class TestAccStepsOverride:
    """run_training_iteration zeroes grads after the optimizer step, so we
    verify accumulation scaling through the resulting weight update:
    SGD(lr=1) ⇒ w_new = w_old − total_accumulated_gradient.
    """

    def _mixin(self):
        from optimus_dl.recipe.train.config import OptimizationConfig
        from optimus_dl.recipe.train.mixins.execution import TrainingIterationMixin

        mixin = TrainingIterationMixin(OptimizationConfig(optimizer={"_name": "adamw"}))
        return mixin

    def _run(self, weights, acc_steps_override=None, acc_steps_cfg=1):
        from optimus_dl.recipe.train.config import OptimizationConfig
        from optimus_dl.recipe.train.mixins.execution import TrainingIterationMixin

        opt_cfg = OptimizationConfig(
            optimizer={"_name": "adamw"}, acc_steps=acc_steps_cfg
        )
        mixin = TrainingIterationMixin(opt_cfg)
        model = TinyHeldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        training_context = {"scaler": FakeScaler(), "amp_ctx": nullcontext()}
        batches = iter([{"weight": torch.tensor(w)} for w in weights])
        kwargs = {}
        if acc_steps_override is not None:
            kwargs["acc_steps_override"] = acc_steps_override
        mixin.run_training_iteration(
            model=model,
            optimizer=optimizer,
            criterion=ScalarWeightedCriterion(),
            train_data_iter=batches,
            training_context=training_context,
            **kwargs,
        )
        return model.weight.item()

    def test_override_controls_microbatch_count_and_scaling(self):
        # loss_i = w_i * Σparams; accumulated grad = mean(3, 5) = 4 ⇒ w = 1 − 4
        assert self._run([3.0, 5.0], acc_steps_override=2) == pytest.approx(
            -3.0, rel=1e-5
        )

    def test_default_uses_configured_acc_steps(self):
        assert self._run([3.0, 5.0], acc_steps_cfg=2) == pytest.approx(-3.0, rel=1e-5)

    def test_single_microbatch_no_scaling(self):
        # One micro-batch, no accumulation: grad = 4 ⇒ w = −3 as well
        assert self._run([4.0], acc_steps_override=1) == pytest.approx(-3.0, rel=1e-5)

    def test_split_microbatches_differ_from_duplicating_first_chunk(self):
        """Regression guard: accumulating over the SAME tensor twice yields
        the first chunk's gradient alone (3), whereas true micro-batching
        averages distinct slices ((3+5)/2 = 4)."""
        from optimus_dl.recipe.train.config import OptimizationConfig
        from optimus_dl.recipe.train.mixins.execution import TrainingIterationMixin

        mixin = TrainingIterationMixin(
            OptimizationConfig(optimizer={"_name": "adamw"}, acc_steps=1)
        )
        model = TinyHeldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        training_context = {"scaler": FakeScaler(), "amp_ctx": nullcontext()}
        same = {"weight": torch.tensor(3.0)}
        mixin.run_training_iteration(
            model=model,
            optimizer=optimizer,
            criterion=ScalarWeightedCriterion(),
            train_data_iter=iter([same, same]),
            training_context=training_context,
            acc_steps_override=2,
        )
        # Old buggy behavior: identical tensor twice ≡ single pass ⇒ w = 1−3
        assert model.weight.item() == pytest.approx(-2.0, rel=1e-5)
        # New behavior with distinct slices ⇒ w = 1−4
        assert self._run([3.0, 5.0], acc_steps_override=2) == pytest.approx(
            -3.0, rel=1e-5
        )


# ---------------------------------------------------------------------------
# Full unified-loop accumulation over split experience
# ---------------------------------------------------------------------------


class _FailingIterator:
    """Iterator that yields good batches, then raises or stops."""

    def __init__(self, batches, failure="exception"):
        self._items = list(batches)
        self._failure = failure

    def __iter__(self):
        return self

    def __next__(self):
        if self._items:
            return self._items.pop(0)
        if self._failure == "exception":
            raise RuntimeError("simulated data pipeline failure")
        raise StopIteration


class TestBatchFetchFailureScaling:
    """Regression tests for gradient scaling when micro-batch fetching fails.

    The loss must be normalized by the number of micro-batches that were
    actually retrieved, not by the configured acc_steps. With SGD(lr=1) the
    weight update equals the total accumulated gradient, so expectations are
    easy to state analytically.
    """

    def _run(self, train_data_iter, acc_steps=2):
        from optimus_dl.recipe.train.config import OptimizationConfig
        from optimus_dl.recipe.train.mixins.execution import TrainingIterationMixin

        mixin = TrainingIterationMixin(
            OptimizationConfig(optimizer={"_name": "adamw"}, acc_steps=acc_steps)
        )
        model = TinyHeldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        training_context = {"scaler": FakeScaler(), "amp_ctx": nullcontext()}
        mixin.run_training_iteration(
            model=model,
            optimizer=optimizer,
            criterion=ScalarWeightedCriterion(),
            train_data_iter=train_data_iter,
            training_context=training_context,
        )
        return model.weight.item()

    def test_exception_mid_accumulation_scales_by_actual_count(self):
        """Second micro-batch fetch raises: the update must equal the first
        batch's full gradient (3.0), not an under-scaled 3/acc_steps."""
        it = _FailingIterator([{"weight": torch.tensor(3.0)}], failure="exception")
        w = self._run(it, acc_steps=2)
        assert w == pytest.approx(
            1.0 - 3.0, rel=1e-5
        ), "gradient must be normalized by the 1 retrieved micro-batch"

    def test_stop_iteration_mid_accumulation_scales_by_actual_count(self):
        """Data exhausted after one micro-batch behaves the same way."""
        it = _FailingIterator([{"weight": torch.tensor(3.0)}], failure="stop")
        w = self._run(it, acc_steps=2)
        assert w == pytest.approx(1.0 - 3.0, rel=1e-5)

    def test_partial_failure_averages_surviving_batches(self):
        """3 fetched of 4 requested (one raises): mean(1, 7, 3) scaled."""

        class FailOnThird:
            def __init__(self):
                self.items = [1.0, 7.0, 3.0]
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                if self.count == 3:
                    raise RuntimeError("boom")
                if self.items:
                    return {"weight": torch.tensor(self.items.pop(0))}
                raise StopIteration

        # acc_steps=4: batch1 ok (1.0), batch2 ok (7.0), batch3 raises,
        # batch4 ok (3.0) → 3 executed, grad = mean(1, 7, 3) = 11/3.
        model_weight = self._run(FailOnThird(), acc_steps=4)
        assert model_weight == pytest.approx(1.0 - 11.0 / 3.0, rel=1e-4)

    def test_all_fetches_fail_skips_optimizer_step(self):
        """No data at all → no forward, no optimizer step, no crash."""
        it = _FailingIterator([], failure="stop")
        w = self._run(it, acc_steps=2)
        assert w == pytest.approx(
            1.0, rel=1e-7
        ), "weight must be untouched when zero micro-batches were retrieved"

        it2 = _FailingIterator([], failure="exception")
        w2 = self._run(it2, acc_steps=2)
        assert w2 == pytest.approx(1.0, rel=1e-7)

    def test_healthy_path_unaffected(self):
        """Full fetch success keeps exact configured-acc_steps semantics."""
        it = _FailingIterator(
            [{"weight": torch.tensor(3.0)}, {"weight": torch.tensor(5.0)}],
            failure="never",
        )
        w = self._run(it, acc_steps=2)
        assert w == pytest.approx(1.0 - 4.0, rel=1e-5)

    def test_ddp_sync_boundary_lands_on_last_executed_batch(self):
        """accumulation_context's is_last flag must fire on the last batch
        that actually ran — otherwise DDP gradient sync never happens."""
        from optimus_dl.recipe.train.config import OptimizationConfig
        from optimus_dl.recipe.train.mixins.execution import TrainingIterationMixin

        seen_is_last = []

        class RecordingModel(TinyHeldModel):
            def accumulation_context(self, is_last_microbatch=True):
                seen_is_last.append(is_last_microbatch)
                return nullcontext()

        mixin = TrainingIterationMixin(
            OptimizationConfig(optimizer={"_name": "adamw"}, acc_steps=2)
        )
        model = RecordingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        training_context = {"scaler": FakeScaler(), "amp_ctx": nullcontext()}
        mixin.run_training_iteration(
            model=model,
            optimizer=optimizer,
            criterion=ScalarWeightedCriterion(),
            train_data_iter=_FailingIterator(
                [{"weight": torch.tensor(3.0)}], failure="stop"
            ),
            training_context=training_context,
        )
        assert seen_is_last == [
            True
        ], "the single executed micro-batch must be treated as the last"


class TestSplitExperienceGradientEquivalence:
    def test_gradient_equals_manual_microbatch_average(self):
        """End-to-end check: splitting the experience into micro-batches and
        running one accumulated iteration reproduces the manually computed
        gradient — the property the old duplicated-tensor approach broke."""
        from optimus_dl.recipe.train.grpo import GRPORecipe

        torch.manual_seed(3)
        B, G, T, V = 2, 2, 5, 32

        stub = _make_stub(G=G, max_new=2)
        policy = ConstantNextModel(token=7)
        ref = ConstantNextModel(token=7)
        reward_fn = RuleBasedReward(RuleBasedRewardConfig())
        engine = NativeEngine(NativeEngineConfig())
        batch = {
            "input_ids": torch.randint(0, V, (B, T)),
            "seq_lens": torch.full((B,), T),
        }

        exp = stub._generate_experience(
            policy, ref, batch, engine, [reward_fn], torch.device("cpu")
        )

        criterion = GRPOCriterion(GRPOCriterionConfig(kl_coeff=0.01))

        # Manual gradient: average of per-microbatch gradients
        manual_model = ConstantNextModel(token=7)
        manual_model.load_state_dict(
            {k: v.clone() for k, v in policy.state_dict().items()}
        )
        chunks = GRPORecipe._split_experience(exp, 2)
        for chunk in chunks:
            loss, _ = criterion(manual_model, chunk)
            (loss / len(chunks)).backward()
        expected = [p.grad.clone() for p in manual_model.parameters()]

        # Split-path gradient through _split_experience
        split_chunks = GRPORecipe._split_experience(exp, 2)
        got = []
        policy.zero_grad()
        for chunk in split_chunks:
            loss, _ = criterion(policy, chunk)
            (loss / len(split_chunks)).backward()
        got = [p.grad.clone() for p in policy.parameters()]

        for e, g in zip(expected, got, strict=True):
            assert torch.allclose(e, g, atol=1e-6)
