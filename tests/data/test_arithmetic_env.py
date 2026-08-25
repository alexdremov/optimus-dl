import re

import torch
import pytest
import torchdata.nodes
from omegaconf import OmegaConf

from optimus_dl.modules.data import build_dataset
from optimus_dl.modules.data.presets.arithmetic import (
    ArithmeticDataset,
    ArithmeticDatasetConfig,
)
from optimus_dl.modules.data.transforms.base import MapperConfig
from optimus_dl.modules.data.transforms.basic_batcher import (
    BasicBatcher,
    BasicBatcherConfig,
)
from optimus_dl.modules.data.transforms.format import (
    FormatTransform,
    FormatTransformConfig,
)
from optimus_dl.modules.data.transforms.tokenize import (
    TokenizeTransform,
    TokenizeTransformConfig,
)
from optimus_dl.modules.reward import build_reward_function
from optimus_dl.modules.reward.implementations.arithmetic import (
    ArithmeticReward,
    ArithmeticRewardConfig,
)
from optimus_dl.modules.tokenizer.implementations.char import CharTokenizerConfig


def make_dataset(**overrides) -> ArithmeticDataset:
    cfg = ArithmeticDatasetConfig(_name="preset_arithmetic", **overrides)
    return ArithmeticDataset(cfg)


def make_reward(**overrides) -> ArithmeticReward:
    cfg = ArithmeticRewardConfig(_name="arithmetic", **overrides)
    return ArithmeticReward(cfg)


def compute_answer(question: str) -> int:
    """Evaluate a left-to-right arithmetic expression like '34+57'."""
    tokens = re.findall(r"\d+|[+\-*]", question)
    value = int(tokens[0])
    i = 1
    while i < len(tokens):
        op, operand = tokens[i], int(tokens[i + 1])
        if op == "+":
            value += operand
        elif op == "-":
            value -= operand
        else:
            value *= operand
        i += 2
    return value


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
def test_samples_are_consistent_and_correct():
    ds = make_dataset(min_digits=1, max_digits=3)
    for sample in [ds.next() for _ in range(100)]:
        assert set(sample.keys()) == {"question", "answer"}
        assert compute_answer(sample["question"]) == int(sample["answer"])


def test_determinism():
    ds_a, ds_b = make_dataset(seed=7), make_dataset(seed=7)
    stream_a = [ds_a.next() for _ in range(20)]
    stream_b = [ds_b.next() for _ in range(20)]
    assert stream_a == stream_b


def test_different_seed_gives_different_stream():
    ds_a, ds_b = make_dataset(seed=1), make_dataset(seed=2)
    assert [ds_a.next() for _ in range(20)] != [ds_b.next() for _ in range(20)]


@pytest.mark.parametrize("digits", [1, 2, 4])
def test_operand_digit_bounds(digits):
    ds = make_dataset(min_digits=digits, max_digits=digits)
    low, high = 10 ** (digits - 1), 10**digits - 1
    for sample in [ds.next() for _ in range(50)]:
        for operand in re.findall(r"\d+", sample["question"]):
            assert low <= int(operand) <= high


def test_rank_streams_are_disjoint():
    cfg = ArithmeticDatasetConfig(_name="preset_arithmetic")
    ds_r0 = ArithmeticDataset(cfg, rank=0, world_size=2)
    ds_r1 = ArithmeticDataset(cfg, rank=1, world_size=2)
    single = ArithmeticDataset(cfg, rank=0, world_size=1)

    stream_single = [single.next()["question"] for _ in range(6)]
    stream_r0 = [ds_r0.next()["question"] for _ in range(3)]
    stream_r1 = [ds_r1.next()["question"] for _ in range(3)]

    interleaved = [x for pair in zip(stream_r0, stream_r1, strict=True) for x in pair]
    assert interleaved == stream_single
    assert len(set(stream_r0) & set(stream_r1)) == 0


def test_infinite_stream():
    ds = make_dataset()
    for _ in range(2000):
        ds.next()


def test_state_save_restore():
    ds = make_dataset()
    for _ in range(5):
        ds.next()

    state = ds.get_state()
    expected = [ds.next()["question"] for _ in range(5)]

    ds_fresh = make_dataset()
    ds_fresh.reset(state)
    restored = [ds_fresh.next()["question"] for _ in range(5)]
    assert restored == expected


def test_reset_without_state_restarts_stream():
    ds = make_dataset()
    first = [ds.next() for _ in range(10)]
    ds.reset(None)
    assert [ds.next() for _ in range(10)] == first


@pytest.mark.parametrize(
    ("operations", "num_operands"),
    [(["+"], 2), (["-"], 2), (["*"], 2), (["+", "-"], 3)],
)
def test_operations(operations, num_operands):
    ds = make_dataset(operations=operations, num_operands=num_operands, max_digits=2)
    for sample in [ds.next() for _ in range(50)]:
        assert compute_answer(sample["question"]) == int(sample["answer"])


def test_invalid_operation_rejected():
    with pytest.raises(AssertionError):
        ArithmeticDatasetConfig(_name="preset_arithmetic", operations=["/"])


def test_max_result_constraint():
    ds = make_dataset(min_digits=1, max_digits=2, max_result=9)
    for sample in [ds.next() for _ in range(100)]:
        assert abs(int(sample["answer"])) <= 9


def test_max_result_infeasible_raises():
    ds = make_dataset(min_digits=1, max_digits=1, max_result=1)
    with pytest.raises(ValueError):
        ds.next()


def test_registry_build():
    cfg = OmegaConf.structured(
        {"_name": "preset_arithmetic", "max_digits": 2, "seed": 42}
    )
    dataset = build_dataset(cfg, rank=0, world_size=1, seed=123)
    assert isinstance(dataset, ArithmeticDataset)
    sample = dataset.next()
    assert compute_answer(sample["question"]) == int(sample["answer"])


# --------------------------------------------------------------------------- #
# Reward                                                                       #
# --------------------------------------------------------------------------- #
def test_exact_match_full_reward():
    reward = make_reward(digit_shaping=False)
    r = reward(prompts=["34+57="], completions=["91"], answers=["91"])
    assert r.shape == (1,)
    assert r.dtype == torch.float32
    assert r[0].item() == pytest.approx(1.0)


def test_digit_shaping_partial_credit():
    """Correct tens digit earns half the shaping budget."""
    reward = make_reward(digit_shaping=True, digit_weight=0.5)
    r = reward(prompts=["34+57="], completions=["90"], answers=["91"])
    assert r[0].item() == pytest.approx(0.25)  # 0.5 weight * 1 matched / 2 digits


def test_digit_shaping_no_credit_for_wrong_digits():
    reward = make_reward(digit_shaping=True)
    r = reward(prompts=["34+57="], completions=["23"], answers=["91"])
    assert r[0].item() == pytest.approx(0.0)


def test_digit_shaping_shared_digit_gets_partial_even_if_reordered():
    """'12' vs '91': digit '1' matches under a shifted alignment."""
    reward = make_reward(digit_shaping=True, digit_weight=0.5)
    r = reward(prompts=["34+57="], completions=["12"], answers=["91"])
    assert r[0].item() == pytest.approx(0.25)


def test_digit_shaping_constant_guess_cannot_exploit():
    """A constant '9' only scores when 9 is genuinely a correct digit."""
    reward = make_reward(digit_shaping=True, digit_weight=0.5)
    wrong_ones = reward(prompts=["7+8="], completions=["9"], answers=["15"])
    assert wrong_ones[0].item() == pytest.approx(0.0)  # '9' vs ones digit '5'
    right_ones = reward(prompts=["1+8="], completions=["9"], answers=["19"])
    assert right_ones[0].item() == pytest.approx(0.25)  # ones digit matches


def test_digit_shaping_shorter_answer_gets_partial_credit():
    reward = make_reward(digit_shaping=True, digit_weight=0.5)
    r = reward(prompts=["7+5="], completions=["2"], answers=["12"])
    assert r[0].item() == pytest.approx(0.25)  # ones digit matches


def test_digit_shaping_best_alignment_credits_prefix():
    """A lone correct tens digit scores even though the answer is longer."""
    reward = make_reward(digit_shaping=True, digit_weight=0.5)
    r = reward(prompts=["12+50="], completions=["6"], answers=["62"])
    assert r[0].item() == pytest.approx(0.25)  # '6' left-aligns with '62'


def test_unparseable_completion_gets_penalty():
    reward = make_reward(digit_shaping=True)
    r = reward(prompts=["34+57="], completions=["<garbage>"], answers=["91"])
    assert r[0].item() == pytest.approx(0.0)


def test_negative_answers_supported():
    reward = make_reward()
    r = reward(prompts=["5-87="], completions=["-82"], answers=["-82"])
    assert r[0].item() == pytest.approx(1.0)


def test_last_number_is_extracted():
    reward = make_reward(digit_shaping=False)
    r = reward(prompts=["1+1="], completions=["The answer is 2"], answers=["2"])
    assert r[0].item() == pytest.approx(1.0)


def test_missing_answers_get_penalty():
    reward = make_reward()
    r = reward(prompts=["1+1="], completions=["2"], answers=None)
    assert r[0].item() == pytest.approx(0.0)


def test_batch_rewards():
    reward = make_reward(digit_shaping=False)
    rewards = reward(
        prompts=["a+b=", "c+d=", "e+f="],
        completions=["7", "wrong text", "9"],
        answers=["7", "8", "10"],
    )
    assert rewards.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_dataset_reward_consistency():
    """A correct completion of a dataset sample earns the full reward."""
    reward = make_reward(digit_shaping=False)
    ds = make_dataset(max_digits=2)
    batch = [ds.next() for _ in range(16)]
    rewards = reward(
        prompts=[f"{s['question']}=" for s in batch],
        completions=[s["answer"] for s in batch],
        answers=[s["answer"] for s in batch],
    )
    assert torch.all(rewards == 1.0)


def test_reward_registry_build():
    fn = build_reward_function(
        OmegaConf.structured({"_name": "arithmetic", "digit_weight": 0.25})
    )
    rewards = fn(prompts=["2+2="], completions=["4"], answers=["4"])
    assert rewards[0].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Pipeline smoke test (config transform chain)                                 #
# --------------------------------------------------------------------------- #
def test_transform_chain_produces_tokenized_prompts_with_answers():
    ds = make_dataset(max_digits=2)
    samples = [ds.next() for _ in range(4)]
    source = torchdata.nodes.IterableWrapper(samples)

    fmt = FormatTransform(
        FormatTransformConfig(template="{question}=", output_field="text")
    )
    tok = TokenizeTransform(
        TokenizeTransformConfig(
            tokenizer_config=CharTokenizerConfig(
                _name="char_tokenize", add_bos=True, add_eos=False
            ),
            worker_cfg=MapperConfig(num_workers=1),
        )
    )
    batcher = BasicBatcher(BasicBatcherConfig(batch_size=4, pad_token_id=0))

    node = fmt.build(source)
    node = tok.build(node)
    node = batcher.build(node)
    batch = list(node)[0]

    assert batch["input_ids"].shape[0] == 4
    assert batch["answer"] == [s["answer"] for s in samples]
    assert batch["seq_lens"].tolist() == [
        1 + len(f"{s['question']}=".encode()) for s in samples
    ]

    # Prompts must not contain EOS (257): it is reserved to stop completions.
    for ids, length in zip(batch["input_ids"], batch["seq_lens"], strict=True):
        assert 257 not in ids[:length].tolist()
