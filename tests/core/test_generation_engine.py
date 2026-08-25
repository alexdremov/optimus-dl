"""Tests for the NativeEngine generation loop.

Focus areas:
- Padding-aware generation (seq_lens): continuations are appended directly
  after each row's last real token, never after the padded block.
- Stop-token handling: first EOS is written, generation halts afterwards.
- Backward compatibility: without seq_lens, tokens are appended at the end.
"""

import torch
import pytest

from optimus_dl.core.generation import (
    GenerationConfig,
    NativeEngine,
    NativeEngineConfig,
)


class ConstantNextModel(torch.nn.Module):
    """Always predicts ``token`` regardless of input."""

    def __init__(self, token: int, vocab_size: int = 32):
        super().__init__()
        self.token = token
        self.vocab_size = vocab_size
        self.seen_seq_lens = []

    def forward(self, input_ids, seq_lens=None, **kw):
        self.seen_seq_lens.append(
            None if seq_lens is None else seq_lens.clone().tolist()
        )
        logits = torch.full(
            (*input_ids.shape, self.vocab_size), -100.0, dtype=torch.float32
        )
        logits[..., self.token] = 100.0
        return {"logits": logits}


class CopyLastTokenModel(torch.nn.Module):
    """Predicts a token equal to (last real input token + 1) % vocab.

    Verifies that generation actually conditions on the *real* prompt tokens,
    not on padding.
    """

    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, seq_lens=None, **kw):
        B, T = input_ids.shape
        if seq_lens is None:
            last = input_ids[:, -1]
        else:
            last = input_ids[torch.arange(B), seq_lens - 1]
        logits = torch.full((B, self.vocab_size), -100.0, dtype=torch.float32)
        logits[torch.arange(B), (last + 1) % self.vocab_size] = 100.0
        # Expand to (B, T, V): only the last-position read matters, but the
        # engine gathers per-row positions itself.
        return {"logits": logits.unsqueeze(1).expand(B, T, self.vocab_size)}


@pytest.fixture
def engine():
    return NativeEngine(NativeEngineConfig())


class TestPaddingAwareGeneration:
    def test_completions_follow_real_tokens_not_padding(self):
        """Right-padded prompts of different lengths get continuations
        directly after each row's last real token."""
        model = ConstantNextModel(token=7)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # len 3
                [4, 5, 0, 0, 0],  # len 2
            ]
        )
        seq_lens = torch.tensor([3, 2])
        cfg = GenerationConfig(max_new_tokens=3, temperature=0.0)

        out = engine.generate(model, prompts, cfg, seq_lens=seq_lens)

        # Width = max prompt len + max_new_tokens = 3 + 3
        assert out.shape == (2, 6)
        assert out[0].tolist() == [1, 2, 3, 7, 7, 7]
        assert out[1].tolist() == [4, 5, 7, 7, 7, 0]  # trailing pad stays 0

    def test_no_pad_between_prompt_and_completion(self):
        """CopyLastTokenModel proves the continuation is conditioned on the
        real last prompt token — impossible if pads sat in between."""
        model = CopyLastTokenModel(vocab_size=32)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor(
            [
                [10, 11, 12, 0, 0],  # last real token 12 → expect 13
                [20, 0, 0, 0, 0],  # last real token 20 → expect 21
            ]
        )
        seq_lens = torch.tensor([3, 1])
        cfg = GenerationConfig(max_new_tokens=2, temperature=0.0)

        out = engine.generate(model, prompts, cfg, seq_lens=seq_lens)

        # Each step predicts (last real token + 1) % V, so the chain is
        # 12→13→14 and 20→21→22 — proving pads never enter the context.
        assert out[0].tolist()[:5] == [10, 11, 12, 13, 14]
        assert out[1].tolist()[:3] == [20, 21, 22]

    def test_seq_lens_passed_to_model_each_step(self):
        """The engine must forward the running lengths for attention masking."""
        model = ConstantNextModel(token=7)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor([[1, 2, 3, 0, 0]])
        seq_lens = torch.tensor([3])
        cfg = GenerationConfig(max_new_tokens=2, temperature=0.0)

        engine.generate(model, prompts, cfg, seq_lens=seq_lens)

        assert model.seen_seq_lens[0] == [3]
        assert model.seen_seq_lens[1] == [4]

    def test_uniform_lengths_match_plain_append(self):
        """With full-length seq_lens the result equals legacy append behavior."""
        model = ConstantNextModel(token=7)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor([[1, 2, 3]])
        cfg = GenerationConfig(max_new_tokens=2, temperature=0.0)

        out_with = engine.generate(model, prompts, cfg, seq_lens=torch.tensor([3]))
        out_without = engine.generate(model, prompts, cfg)

        assert out_with.tolist() == out_without.tolist() == [[1, 2, 3, 7, 7]]


class TestStopTokenHandling:
    def test_first_eos_is_written_and_generation_stops(self):
        model = ConstantNextModel(token=5)  # 5 is the stop token
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor([[1, 2, 3, 0, 0]])
        seq_lens = torch.tensor([3])
        cfg = GenerationConfig(max_new_tokens=4, temperature=0.0, stop_token_id=5)

        out = engine.generate(model, prompts, cfg, seq_lens=seq_lens)

        # EOS written once at position 3; cursor does not advance afterwards,
        # so the output is trimmed to 4 columns (not 3 + 4).
        assert out.shape == (1, 4)
        assert out[0].tolist() == [1, 2, 3, 5]

    def test_stop_token_per_row(self):
        """Only the row that emits EOS stops; the other keeps generating."""
        model = ConstantNextModel(token=5)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
        seq_lens = torch.tensor([3, 3])
        cfg = GenerationConfig(max_new_tokens=3, temperature=0.0, stop_token_id=5)

        out = engine.generate(model, prompts, cfg, seq_lens=seq_lens)

        # Both rows emit 5 (stop) on the first step → width 3 + 1.
        assert out.shape == (2, 4)
        assert out[0].tolist() == [1, 2, 3, 5]
        assert out[1].tolist() == [4, 5, 6, 5]

    def test_final_stop_token_kept_when_batch_finishes_together(self):
        """Regression: rows finishing on the SAME step must keep their stop token.

        The cached loop used to break before advancing ``cur_lens``, so the
        trim bound excluded the just-written stop token of last-finishing rows.
        """

        class FillThenEosModel(torch.nn.Module):
            """Emits `fill` for `steps` steps, then EOS for every row."""

            def __init__(self, fill, eos, steps, vocab_size=32):
                super().__init__()
                self.fill, self.eos, self.steps = fill, eos, steps
                self.vocab_size = vocab_size
                self.calls = 0

            def forward(self, input_ids, seq_lens=None, **kw):
                token = self.fill if self.calls < self.steps else self.eos
                self.calls += 1
                logits = torch.full(
                    (*input_ids.shape, self.vocab_size),
                    -100.0,
                    dtype=torch.float32,
                )
                logits[..., token] = 100.0
                return {"logits": logits}

        model = FillThenEosModel(fill=7, eos=9, steps=2)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor([[1, 2], [3, 4]])
        seq_lens = torch.tensor([2, 2])
        cfg = GenerationConfig(max_new_tokens=4, temperature=0.0, stop_token_id=9)

        out = engine.generate(model, prompts, cfg, seq_lens=seq_lens)

        # Both rows emit two fill tokens then EOS on step 3; that final stop
        # token must survive the width trim.
        assert out.shape == (2, 5)
        assert out[0].tolist() == [1, 2, 7, 7, 9]
        assert out[1].tolist() == [3, 4, 7, 7, 9]


class TestSamplingModes:
    def test_greedy_is_deterministic(self):
        torch.manual_seed(0)
        model = ConstantNextModel(token=7)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor([[1, 2], [3, 4]])
        cfg = GenerationConfig(max_new_tokens=3, temperature=0.0)

        out1 = engine.generate(model, prompts, cfg)
        out2 = engine.generate(model, prompts, cfg)
        assert out1.tolist() == out2.tolist()

    def test_sampled_output_shape(self):
        model = ConstantNextModel(token=7)
        engine = NativeEngine(NativeEngineConfig())
        prompts = torch.tensor([[1, 2, 3]])
        cfg = GenerationConfig(max_new_tokens=5, temperature=1.0)

        out = engine.generate(model, prompts, cfg)
        assert out.shape == (1, 8)
        assert (out[:, 3:] == 7).all()

    def test_model_back_to_train_mode_after_generate(self):
        """generate() calls eval(); callers rely on restoring train mode
        themselves — verify eval() is actually invoked via a flag model."""

        class TrackingModel(ConstantNextModel):
            def train(self, mode=True):
                self.last_mode = mode
                return super().train(mode)

        tm = TrackingModel(token=7)
        engine = NativeEngine(NativeEngineConfig())
        cfg = GenerationConfig(max_new_tokens=1, temperature=0.0)
        engine.generate(tm, torch.tensor([[1, 2]]), cfg)
        assert tm.last_mode is False, "generate must switch model to eval"
