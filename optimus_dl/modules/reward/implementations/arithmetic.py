"""Arithmetic-environment rewards (scalar and dense token-level variants)."""

import re
from dataclasses import dataclass
from typing import Any

import torch

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.reward import register_reward_function
from optimus_dl.modules.reward.base import (
    BaseReward,
    extract_last_number,
)


@dataclass
class ArithmeticRewardConfig(RegistryConfigStrict):
    """Configuration for the arithmetic-environment reward.

    Designed for the synthetic ``preset_arithmetic`` dataset: completions
    must state the exact result of the expression in the prompt.

    Attributes:
        reward_value: Value for exact numeric matches.
        penalty_value: Base value for incorrect/unparseable answers.
        digit_shaping: Whether to grant partial credit for correctly
            predicted digits. Digits are compared under the best alignment
            offset, so a correct digit earns credit regardless of whether
            the prediction is too short or too long (e.g. predicting the
            tens digit alone still scores). Unlike distance-based shaping
            this cannot be exploited by a constant guess, and it teaches
            place-value computation.
        digit_weight: Credit for a fully matched wrong-length answer is
            ``digit_weight``; per-position matches scale down from there.
            Must be < 1.0 so exact matches strictly dominate.
    """

    reward_value: float = 1.0
    penalty_value: float = 0.0
    digit_shaping: bool = True
    digit_weight: float = 0.5


class ArithmeticReward(BaseReward):
    """Reward function for arithmetic tasks (e.g. ``"34+57=" -> "91"``).

    Extracts the last number from each completion and compares it against
    the ground-truth answer. Exact matches receive ``reward_value``; with
    ``digit_shaping`` enabled, partially correct answers receive credit
    proportional to the best-aligned digit match; unparseable completions
    get ``penalty_value``.
    """

    def __init__(self, cfg: ArithmeticRewardConfig):
        self.cfg = cfg

    @staticmethod
    def _best_digit_match_frac(pred_digits: str, true_digits: str) -> float:
        """Fraction of digits matched under the best alignment offset."""
        lp, lt = len(pred_digits), len(true_digits)
        best = 0
        for offset in range(-lp + 1, lt):
            matches = sum(
                1
                for i in range(lp)
                if 0 <= i + offset < lt and pred_digits[i] == true_digits[i + offset]
            )
            best = max(best, matches)
        return best / max(lp, lt)

    def extract_answer(self, text: str) -> float | None:
        """Extract the last number from the text as a float."""
        return extract_last_number(text)

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        rewards = []
        for i, completion in enumerate(completions):
            if answers is None or i >= len(answers):
                rewards.append(self.cfg.penalty_value)
                continue

            pred = self.extract_answer(completion)
            true_ans = extract_last_number(answers[i])

            reward = self.cfg.penalty_value

            if pred is not None and true_ans is not None:
                if abs(pred - true_ans) < 1e-6:
                    reward = self.cfg.reward_value
                elif self.cfg.digit_shaping:
                    pred_digits = str(abs(int(round(pred))))
                    true_digits = str(abs(int(round(true_ans))))
                    frac = self._best_digit_match_frac(pred_digits, true_digits)
                    reward = (
                        self.cfg.penalty_value
                        + (self.cfg.reward_value - self.cfg.penalty_value)
                        * self.cfg.digit_weight
                        * frac
                    )

            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)


@dataclass
class ArithmeticTokenRewardConfig(ArithmeticRewardConfig):
    """Configuration for the dense (token-level) arithmetic reward.

    Same scoring semantics as :class:`ArithmeticReward`, but credit is
    assigned to the individual generated tokens that spell the predicted
    number (plus a trailing stop token, if present) instead of being
    broadcast uniformly over every completion token. Junk tokens receive no
    credit, so gradients directly reinforce the digits that matter.
    """


class ArithmeticTokenReward(ArithmeticReward):
    """Dense variant of :class:`ArithmeticReward`.

    Sequence-level contract: ``__call__`` returns zeros; all reward mass
    flows through ``token_rewards`` so that summing a row's token rewards
    equals what the scalar variant would have scored for that completion.
    """

    @staticmethod
    def _number_span(pieces: list[str], pred_digits: str) -> list[int]:
        """Indices of tokens composing the LAST number in the text.

        ``pieces[t]`` is the decoded text of completion token ``t``. Returns
        the indices of the tokens whose concatenated text contains the last
        number occurrence, plus one immediately following non-digit piece
        (e.g. an ``<|im_end|>`` marker), so clean termination stays
        reinforced.
        """
        full = "".join(pieces)
        matches = list(re.finditer(r"-?\d+(?:\.\d+)?", full))
        if not matches:
            return []
        start, end = matches[-1].span()

        span = []
        cursor = 0
        for t, piece in enumerate(pieces):
            tok_start, tok_end = cursor, cursor + len(piece)
            cursor = tok_end
            if tok_end <= start or tok_start >= end:
                continue
            span.append(t)

        if span and span[-1] + 1 < len(pieces):
            nxt = pieces[span[-1] + 1]
            is_stop_marker = "<|" in nxt or nxt.strip() == "" or not nxt.isprintable()
            if is_stop_marker:
                span.append(span[-1] + 1)
        return span

    def supports_token_level(self) -> bool:
        return True

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Sequence-level contribution of a dense reward is zero: its reward
        # mass flows exclusively through `token_rewards`.
        return torch.zeros(len(completions), dtype=torch.float32)

    def token_rewards(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str] | None = None,
        completion_ids: list[list[int]] | None = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        assert (
            completion_ids is not None and tokenizer is not None
        ), "ArithmeticTokenReward requires completion_ids and tokenizer"
        rows: list[torch.Tensor] = []
        for i, ids in enumerate(completion_ids):
            n = len(ids)
            if n == 0 or answers is None or i >= len(answers):
                rows.append(torch.zeros(max(n, 0), dtype=torch.float32))
                continue

            pieces = [tokenizer.decode([tid]) for tid in ids]
            pred = extract_last_number("".join(pieces))
            true_ans = extract_last_number(answers[i])

            values = torch.full((n,), self.cfg.penalty_value, dtype=torch.float32)

            if pred is None or true_ans is None:
                rows.append(values)
                continue

            if abs(pred - true_ans) < 1e-6:
                value = self.cfg.reward_value
            elif self.cfg.digit_shaping:
                pred_digits = str(abs(int(round(pred))))
                true_digits = str(abs(int(round(true_ans))))
                frac = self._best_digit_match_frac(pred_digits, true_digits)
                value = (
                    self.cfg.penalty_value
                    + (self.cfg.reward_value - self.cfg.penalty_value)
                    * self.cfg.digit_weight
                    * frac
                )
            else:
                value = self.cfg.penalty_value

            if value == self.cfg.penalty_value:
                rows.append(values)
                continue

            span = self._number_span(pieces, str(abs(int(round(pred)))))
            if not span:
                span = [n - 1]
            share = (value - self.cfg.penalty_value) / len(span)
            for t in span:
                values[t] = self.cfg.penalty_value + share
            rows.append(values)
        return rows


register_reward_function("arithmetic", ArithmeticRewardConfig)(ArithmeticReward)
register_reward_function("arithmetic_token", ArithmeticTokenRewardConfig)(
    ArithmeticTokenReward
)
