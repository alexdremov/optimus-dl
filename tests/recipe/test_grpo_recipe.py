"""Unit and integration tests for GRPORecipe helpers and GRPOConfig.

Design principle: no mocking of training math.
- Real models (tiny embed+linear)
- Real reward functions (RuleBasedReward / GSM8kReward)
- Real generation engine (NativeEngine)
- log_averaged is a no-op outside meters_group — no patching needed
- Only mock I/O / distributed infrastructure (FakeCollective)
"""

from dataclasses import fields as dc_fields

import torch
import pytest
import torch.nn as nn

from optimus_dl.core.generation import (
    NativeEngine,
    NativeEngineConfig,
)
from optimus_dl.modules.reward import (
    GSM8kReward,
    GSM8kRewardConfig,
    RuleBasedReward,
    RuleBasedRewardConfig,
)

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

    def train(self, mode=True):
        return super().train(mode)

    def eval(self):
        return super().eval()


def _make_ref_model(policy: _TinyLM) -> _TinyLM:
    ref = _TinyLM(policy.embed.num_embeddings, policy.embed.embedding_dim)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _make_grpo_recipe_stub(G: int = 4, vocab_size: int = 32):
    """Return a minimal object that has the GRPO recipe methods bound to it."""
    from optimus_dl.core.generation import GenerationConfig
    from optimus_dl.recipe.train.grpo import GRPORecipe

    class _Stub:
        pass

    stub = _Stub()
    stub.cfg = type(
        "cfg",
        (),
        {
            "num_generations": G,
            "generation_config": GenerationConfig(max_new_tokens=4),
        },
    )()
    stub.tokenizer = None

    # Bind instance methods from GRPORecipe
    stub._generate_experience = GRPORecipe._generate_experience.__get__(stub)
    stub._decode_batch = GRPORecipe._decode_batch.__get__(stub)
    stub._get_per_token_logprobs = GRPORecipe._get_per_token_logprobs
    stub._build_completion_mask = GRPORecipe._build_completion_mask
    stub._truncate_after_stop = GRPORecipe._truncate_after_stop
    stub._get_prompt_lengths = GRPORecipe._get_prompt_lengths
    stub._compute_batch_logprobs = GRPORecipe._compute_batch_logprobs.__get__(stub)
    stub._split_experience = GRPORecipe._split_experience

    return stub


# ---------------------------------------------------------------------------
# GRPOConfig field inspection
# ---------------------------------------------------------------------------


class TestGRPOConfig:
    def test_required_fields_exist(self):
        from optimus_dl.recipe.train.grpo import GRPOConfig

        field_names = {f.name for f in dc_fields(GRPOConfig)}
        for name in (
            "num_generations",
            "generation_engine",
            "reward_functions",
            "ref_model_transforms",
            "tokenizer_config",
        ):
            assert name in field_names

    def test_num_generations_default_is_8(self):
        from optimus_dl.recipe.train.grpo import GRPOConfig

        fields = {f.name: f for f in dc_fields(GRPOConfig)}
        assert fields["num_generations"].default == 8

    def test_reward_functions_default_is_empty_list(self):
        from optimus_dl.recipe.train.grpo import GRPOConfig

        fields = {f.name: f for f in dc_fields(GRPOConfig)}
        assert fields["reward_functions"].default_factory() == []

    def test_tokenizer_config_default_is_none(self):
        from optimus_dl.recipe.train.grpo import GRPOConfig

        fields = {f.name: f for f in dc_fields(GRPOConfig)}
        assert fields["tokenizer_config"].default is None


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestGRPORecipeValidateConfig:
    """validate_config() must fail fast on bad configs."""

    def _bare_recipe(self, reward_functions, generation_engine):
        """Create a GRPORecipe with minimal mocked base to test validation only."""
        from unittest.mock import MagicMock

        from optimus_dl.recipe.train.grpo import GRPORecipe

        recipe = object.__new__(GRPORecipe)
        cfg = MagicMock()
        cfg.reward_functions = reward_functions
        cfg.generation_engine = generation_engine
        cfg.model = MagicMock()
        cfg.data = MagicMock()
        cfg.criterion = MagicMock()
        cfg.optimization.iterations = 10
        cfg.optimization.acc_steps = 1
        cfg.common.log_freq = 1
        cfg.common.save_freq = None
        recipe.cfg = cfg
        return recipe

    def test_empty_reward_functions_raises(self):
        recipe = self._bare_recipe(
            reward_functions=[], generation_engine={"_name": "native"}
        )
        with pytest.raises(AssertionError, match="reward_functions"):
            recipe.validate_config()

    def test_none_generation_engine_raises(self):
        from unittest.mock import MagicMock

        recipe = self._bare_recipe(
            reward_functions=[MagicMock()], generation_engine=None
        )
        with pytest.raises(AssertionError, match="generation_engine"):
            recipe.validate_config()


# ---------------------------------------------------------------------------
# _get_per_token_logprobs — static method
# ---------------------------------------------------------------------------


class TestGetPerTokenLogprobs:
    def test_output_shape_is_b_t_minus_1(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T, V = 3, 10, 50
        logits = torch.randn(B, T, V)
        tokens = torch.randint(0, V, (B, T))
        out = GRPORecipe._get_per_token_logprobs(logits, tokens)
        assert out.shape == (B, T - 1)

    def test_output_values_are_log_probabilities(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T, V = 2, 8, 30
        logits = torch.randn(B, T, V)
        tokens = torch.randint(0, V, (B, T))
        out = GRPORecipe._get_per_token_logprobs(logits, tokens)
        assert (out <= 0).all(), "log-probabilities must be ≤ 0"

    def test_confident_prediction_gives_log_prob_near_zero(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T, V = 1, 3, 10
        tokens = torch.tensor([[2, 5, 7]])
        logits = torch.full((B, T, V), -1e9)
        logits[0, 0, 5] = 1e9  # step 0 → token 5
        logits[0, 1, 7] = 1e9  # step 1 → token 7
        out = GRPORecipe._get_per_token_logprobs(logits, tokens)
        assert out[0, 0].item() == pytest.approx(0.0, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(0.0, abs=1e-4)

    def test_gathers_the_correct_token_logprob(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T, V = 1, 2, 5
        tokens = torch.tensor([[3, 1]])  # next token at step 0 is index 1
        logits = torch.zeros(B, T, V)
        logits[0, 0, 1] = 10.0  # strongly predicts token 1
        logits[0, 0, 0] = -10.0
        out = GRPORecipe._get_per_token_logprobs(logits, tokens)
        expected = torch.log_softmax(logits[0, 0], dim=-1)[1]
        assert out[0, 0].item() == pytest.approx(expected.item(), abs=1e-5)


# ---------------------------------------------------------------------------
# _build_completion_mask — static method
# ---------------------------------------------------------------------------


class TestBuildCompletionMask:
    def test_fixed_prompt_length(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T, prompt_len, G = 2, 10, 3, 1
        all_ids = torch.zeros(B * G, T, dtype=torch.long)
        prompt_lengths = torch.full((B * G,), prompt_len, dtype=torch.long)
        total_lengths = torch.full((B * G,), T, dtype=torch.long)
        mask = GRPORecipe._build_completion_mask(all_ids, prompt_lengths, total_lengths)
        assert mask.shape == (B * G, T)
        assert (mask[:, :prompt_len] == 0).all()
        assert (mask[:, prompt_len:] == 1).all()

    def test_variable_prompt_length(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T, G = 2, 10, 1
        all_ids = torch.zeros(B * G, T, dtype=torch.long)
        prompt_lengths = torch.tensor([4, 6], dtype=torch.long)
        total_lengths = torch.full((B * G,), T, dtype=torch.long)
        mask = GRPORecipe._build_completion_mask(all_ids, prompt_lengths, total_lengths)
        assert (mask[0, :4] == 0).all() and (mask[0, 4:] == 1).all()
        assert (mask[1, :6] == 0).all() and (mask[1, 6:] == 1).all()

    def test_total_length_excludes_trailing_padding(self):
        """Positions at or beyond total_lengths must be masked out (padding)."""
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T = 2, 10
        all_ids = torch.zeros(B, T, dtype=torch.long)
        prompt_lengths = torch.tensor([3, 5], dtype=torch.long)
        total_lengths = torch.tensor([8, 7], dtype=torch.long)  # trailing pads
        mask = GRPORecipe._build_completion_mask(all_ids, prompt_lengths, total_lengths)
        assert (mask[0, 8:] == 0).all(), "trailing pad must be excluded"
        assert (mask[0, 3:8] == 1).all()
        assert (mask[1, 7:] == 0).all(), "trailing pad must be excluded"
        assert (mask[1, 5:7] == 1).all()

    def test_g_generations_expand_correctly(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        B, T, prompt_len, G = 2, 8, 3, 3
        all_ids = torch.zeros(B * G, T, dtype=torch.long)
        prompt_lengths = torch.full((B * G,), prompt_len, dtype=torch.long)
        total_lengths = torch.full((B * G,), T, dtype=torch.long)
        mask = GRPORecipe._build_completion_mask(all_ids, prompt_lengths, total_lengths)
        assert mask.shape == (B * G, T)
        for row in range(B * G):
            assert (mask[row, :prompt_len] == 0).all()
            assert (mask[row, prompt_len:] == 1).all()


class TestTruncateAfterStop:
    def test_truncates_after_first_stop_token(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        # row: [P P C C EOS C C] — keep up to & incl EOS, zero after
        all_ids = torch.tensor([[9, 9, 1, 2, 5, 3, 4]])
        completion_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]])
        out = GRPORecipe._truncate_after_stop(all_ids, completion_mask, stop_token_id=5)
        assert out.tolist() == [[0, 0, 1, 1, 1, 0, 0]]

    def test_stop_token_in_prompt_is_ignored(self):
        """A stop token inside the prompt region must not truncate anything."""
        from optimus_dl.recipe.train.grpo import GRPORecipe

        all_ids = torch.tensor([[5, 1, 2, 3]])  # stop id in prompt position 0
        completion_mask = torch.tensor([[0, 1, 1, 1]])
        out = GRPORecipe._truncate_after_stop(all_ids, completion_mask, stop_token_id=5)
        assert out.tolist() == [[0, 1, 1, 1]]

    def test_no_stop_token_keeps_mask(self):
        from optimus_dl.recipe.train.grpo import GRPORecipe

        all_ids = torch.tensor([[1, 2, 3, 4]])
        completion_mask = torch.tensor([[0, 1, 1, 1]])
        out = GRPORecipe._truncate_after_stop(all_ids, completion_mask, stop_token_id=9)
        assert out.tolist() == [[0, 1, 1, 1]]


# ---------------------------------------------------------------------------
# _decode_batch — with and without tokenizer, with and without attention_mask
# ---------------------------------------------------------------------------


class TestDecodeBatch:
    def test_output_length_is_B_times_G(self):
        B, G, T = 3, 4, 8
        stub = _make_grpo_recipe_stub(G=G)
        stub.tokenizer = type("Tok", (), {"decode": lambda self, ids: "text"})()

        prompt_ids = torch.zeros(B, 4, dtype=torch.long)
        prompt_lengths = torch.full((B,), 4, dtype=torch.long)
        all_ids = torch.zeros(B * G, T, dtype=torch.long)
        mask = torch.ones(B * G, T, dtype=torch.long)
        mask[:, :4] = 0
        batch = {"answer": ["a"] * B}

        prompts, completions, answers = stub._decode_batch(
            prompt_ids, prompt_lengths, all_ids, mask, batch, B, G
        )
        assert len(prompts) == len(completions) == len(answers) == B * G

    def test_each_answer_repeated_G_times(self):
        B, G = 2, 3
        stub = _make_grpo_recipe_stub(G=G)
        stub.tokenizer = type("Tok", (), {"decode": lambda self, ids: "x"})()

        T = 6
        prompt_ids = torch.zeros(B, 2, dtype=torch.long)
        prompt_lengths = torch.full((B,), 2, dtype=torch.long)
        all_ids = torch.zeros(B * G, T, dtype=torch.long)
        mask = torch.ones(B * G, T, dtype=torch.long)
        mask[:, :2] = 0
        batch = {"answer": ["A", "B"]}

        _, _, answers = stub._decode_batch(
            prompt_ids, prompt_lengths, all_ids, mask, batch, B, G
        )
        assert answers[:G] == ["A"] * G
        assert answers[G:] == ["B"] * G

    def test_no_tokenizer_returns_empty_strings(self):
        B, G, T = 2, 2, 6
        stub = _make_grpo_recipe_stub(G=G)
        stub.tokenizer = None  # explicitly no tokenizer

        prompt_ids = torch.zeros(B, 3, dtype=torch.long)
        prompt_lengths = torch.full((B,), 3, dtype=torch.long)
        all_ids = torch.zeros(B * G, T, dtype=torch.long)
        mask = torch.ones(B * G, T, dtype=torch.long)
        mask[:, :3] = 0
        batch = {}

        prompts, completions, answers = stub._decode_batch(
            prompt_ids, prompt_lengths, all_ids, mask, batch, B, G
        )
        assert all(p == "" for p in prompts)
        assert all(c == "" for c in completions)

    def test_prompt_lengths_trim_padding_before_decode(self):
        """Decoded prompt must contain only the non-pad tokens."""
        B, G = 1, 1
        stub = _make_grpo_recipe_stub(G=G)

        decoded_prompts = []

        class CaptureTok:
            def decode(self, ids):
                decoded_prompts.append(ids)
                return "decoded"

        stub.tokenizer = CaptureTok()
        prompt_ids = torch.tensor([[10, 20, 30, 0, 0]])  # 3 real + 2 pad
        prompt_lengths = torch.tensor([3], dtype=torch.long)
        all_ids = torch.zeros(B * G, 8, dtype=torch.long)
        mask = torch.ones(B * G, 8, dtype=torch.long)
        mask[:, :3] = 0

        stub._decode_batch(prompt_ids, prompt_lengths, all_ids, mask, {}, B, G)

        assert decoded_prompts[0] == [10, 20, 30]


# ---------------------------------------------------------------------------
# _determine_role
# ---------------------------------------------------------------------------


class TestDetermineRole:
    def _recipe(self, partitions):
        from unittest.mock import MagicMock

        from optimus_dl.recipe.train.grpo import GRPORecipe

        recipe = object.__new__(GRPORecipe)
        recipe.cfg = MagicMock()
        recipe.cfg.common.distributed.partitions = partitions
        return recipe

    def _collective(self, rank):
        from unittest.mock import MagicMock

        c = MagicMock()
        c.rank = rank
        return c

    def test_no_partitions_gives_unified(self):
        assert self._recipe(None)._determine_role(self._collective(0)) == "unified"

    def test_empty_partitions_gives_unified(self):
        assert self._recipe([])._determine_role(self._collective(0)) == "unified"

    def test_actor_partition_gives_trainer(self):
        from unittest.mock import MagicMock

        p = MagicMock()
        p.name = "actor_group"
        p.ranks = [0, 1]
        assert self._recipe([p])._determine_role(self._collective(0)) == "trainer"

    def test_non_actor_partition_gives_rollout(self):
        from unittest.mock import MagicMock

        p = MagicMock()
        p.name = "ref_group"
        p.ranks = [2, 3]
        assert self._recipe([p])._determine_role(self._collective(2)) == "rollout"

    def test_unmatched_rank_gives_idle(self):
        from unittest.mock import MagicMock

        p = MagicMock()
        p.name = "actor_group"
        p.ranks = [0, 1]
        assert self._recipe([p])._determine_role(self._collective(5)) == "idle"


# ---------------------------------------------------------------------------
# _generate_experience — real model + real gen engine + real reward function
# ---------------------------------------------------------------------------


class TestGenerateExperience:
    """Tests for _generate_experience using real NativeEngine and real rewards."""

    def _setup(self, G: int = 4, B: int = 2, L: int = 5, V: int = 32, new_tok: int = 4):
        from optimus_dl.core.generation import GenerationConfig

        policy = _TinyLM(V)
        ref = _make_ref_model(policy)
        gen_engine = NativeEngine(NativeEngineConfig())
        reward_fn = RuleBasedReward(
            RuleBasedRewardConfig(target_string="", reward_value=1.0)
        )

        stub = _make_grpo_recipe_stub(G=G, vocab_size=V)
        stub.cfg.generation_config = GenerationConfig(
            max_new_tokens=new_tok, temperature=1.0
        )
        return (
            stub,
            policy,
            ref,
            gen_engine,
            reward_fn,
            {"input_ids": torch.randint(0, V, (B, L))},
        )

    def test_output_keys_present(self):
        stub, policy, ref, gen_engine, rf, batch = self._setup()
        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        assert {
            "input_ids",
            "completion_mask",
            "seq_lens",
            "old_logprobs",
            "ref_logprobs",
            "advantages",
        } == set(exp)

    def test_output_shapes(self):
        G, B, L, new_tok, V = 3, 2, 4, 5, 32
        stub, policy, ref, gen_engine, rf, batch = self._setup(
            G=G, B=B, L=L, new_tok=new_tok, V=V
        )
        T = L + new_tok
        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        assert exp["input_ids"].shape == (B * G, T)
        assert exp["completion_mask"].shape == (B * G, T)
        assert exp["old_logprobs"].shape == (B * G, T - 1)
        assert exp["ref_logprobs"].shape == (B * G, T - 1)
        assert exp["advantages"].shape == (B * G,)

    def test_completion_mask_is_zero_at_prompt(self):
        G, B, L = 2, 2, 4
        stub, policy, ref, gen_engine, rf, batch = self._setup(G=G, B=B, L=L)
        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        assert (exp["completion_mask"][:, :L] == 0).all()
        assert (exp["completion_mask"][:, L:] == 1).all()

    def test_advantages_zero_mean_per_group(self):
        """Normalised advantages within each group must sum to ~0."""
        G, B = 4, 3
        stub, policy, ref, gen_engine, _, batch = self._setup(G=G, B=B)
        # Use a reward fn that gives distinct scores (position-dependent)
        rewards = torch.arange(B * G, dtype=torch.float32)
        rf = type("RF", (), {"__call__": lambda self, **kw: rewards})()

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        adv = exp["advantages"].view(B, G)
        assert torch.allclose(adv.mean(dim=1), torch.zeros(B), atol=1e-5)

    def test_advantages_unit_std_per_group(self):
        G, B = 4, 2
        stub, policy, ref, gen_engine, _, batch = self._setup(G=G, B=B)
        # Spread-out rewards so group std >> epsilon
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
        rf = type("RF", (), {"__call__": lambda self, **kw: rewards})()

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        adv = exp["advantages"].view(B, G)
        for i, std in enumerate(adv.std(dim=1, correction=1).tolist()):
            assert abs(std - 1.0) < 0.01, f"group {i} std={std:.4f}"

    def test_multiple_reward_fns_summed(self):
        """Two reward functions must be summed before normalisation."""
        G, B = 2, 2
        stub, policy, ref, gen_engine, _, batch = self._setup(G=G, B=B)
        # rf1: all ones; rf2: varying → together they differ across generations
        rf1 = type("R1", (), {"__call__": lambda self, **kw: torch.ones(B * G)})()
        rf2_rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
        rf2 = type("R2", (), {"__call__": lambda self, **kw: rf2_rewards})()

        exp = stub._generate_experience(
            policy, ref, batch, gen_engine, [rf1, rf2], torch.device("cpu")
        )
        # rf1 alone gives all-equal rewards → std 0 → advantages all 0
        # rf1+rf2 gives non-equal rewards → non-zero advantages
        assert exp["advantages"].std().item() > 0

    def test_model_back_to_train_mode_after_generation(self):
        stub, policy, ref, gen_engine, rf, batch = self._setup()
        assert policy.training  # start in train mode
        stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        assert (
            policy.training
        ), "policy must be in train mode after _generate_experience"

    def test_ref_model_remains_frozen(self):
        stub, policy, ref, gen_engine, rf, batch = self._setup()
        ref_params_before = [p.clone() for p in ref.parameters()]
        stub._generate_experience(
            policy, ref, batch, gen_engine, [rf], torch.device("cpu")
        )
        for before, after in zip(ref_params_before, ref.parameters(), strict=False):
            assert torch.equal(before, after), "ref model weights changed!"


# ---------------------------------------------------------------------------
# GSM8k reward function — full unit coverage
# ---------------------------------------------------------------------------


class TestGSM8kReward:
    def _r(self, **kw):
        return GSM8kReward(GSM8kRewardConfig(**kw))

    def test_exact_match_gives_full_reward(self):
        r = self._r(reward_value=1.0, penalty_value=0.0)
        rewards = r(prompts=["q"], completions=["answer is 4"], answers=["4"])
        assert rewards[0].item() == pytest.approx(1.0)

    def test_wrong_answer_gives_penalty(self):
        r = self._r(reward_value=1.0, penalty_value=-1.0)
        rewards = r(prompts=["q"], completions=["answer is 5"], answers=["4"])
        assert rewards[0].item() == pytest.approx(-1.0)

    def test_no_number_gives_penalty(self):
        r = self._r(reward_value=1.0, penalty_value=0.0)
        rewards = r(prompts=["q"], completions=["I do not know"], answers=["4"])
        assert rewards[0].item() == pytest.approx(0.0)

    def test_none_answers_gives_penalty(self):
        r = self._r(reward_value=1.0, penalty_value=0.0)
        rewards = r(prompts=["q"], completions=["4"], answers=None)
        assert rewards[0].item() == pytest.approx(0.0)

    def test_batch_shape_and_dtype(self):
        r = self._r()
        n = 6
        rewards = r(prompts=[""] * n, completions=["4"] * n, answers=["4"] * n)
        assert rewards.shape == (n,)
        assert rewards.dtype == torch.float32

    def test_dense_reward_in_range(self):
        r = self._r(
            reward_value=1.0, penalty_value=0.0, dense_reward=True, dense_scale=10.0
        )
        rewards = r(prompts=["q"], completions=["answer is 5"], answers=["4"])
        assert 0.0 < rewards[0].item() < 1.0

    def test_float_exact_match(self):
        r = self._r(reward_value=1.0, penalty_value=0.0)
        rewards = r(prompts=["q"], completions=["result is 3.14"], answers=["3.14"])
        assert rewards[0].item() == pytest.approx(1.0)

    def test_extract_last_number(self):
        r = self._r()
        assert r.extract_answer("blah 3.5 and 7") == pytest.approx(7.0)
        assert r.extract_answer("nothing here") is None
        assert r.extract_answer("negative -42") == pytest.approx(-42.0)
