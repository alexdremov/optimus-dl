"""Tests for KV-cached generation.

The cached path (prefill + single-token decode steps) must produce results
identical to the legacy full-re-forward loop, including for right-padded
variable-length prompts and stop tokens.
"""

import copy

import torch
import pytest

from optimus_dl.core.generation import (
    GenerationConfig,
    NativeEngine,
    NativeEngineConfig,
)
from optimus_dl.modules.model.llama2 import (
    Llama,
    LlamaConfig,
)

VOCAB = 64


def _tiny_llama(**overrides) -> Llama:
    cfg_kwargs = dict(
        vocab_size=VOCAB,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_kv_head=2,
        sequence_length=128,
        intermediate_size=48,
        multiple_of=8,
        tie_word_embeddings=True,
        dropout=0.0,
    )
    cfg_kwargs.update(overrides)
    torch.manual_seed(1234)
    model = Llama(LlamaConfig(**cfg_kwargs))
    model.eval()
    return model


@pytest.fixture
def engine():
    return NativeEngine(NativeEngineConfig())


def _prompts_and_lens():
    prompts = torch.tensor(
        [
            [5, 11, 27, 42, 3],  # len 5
            [8, 19, 0, 0, 0],  # len 2 (right-padded)
            [31, 7, 14, 2, 0],  # len 4
        ]
    )
    seq_lens = torch.tensor([5, 2, 4])
    return prompts, seq_lens


class TestCachedLegacyParity:
    """Cached generation must match the legacy loop token-for-token."""

    @pytest.mark.parametrize("temperature", [0.0, 1.0])
    def test_greedy_and_sampled_parity(self, engine, temperature):
        model = _tiny_llama()
        prompts, seq_lens = _prompts_and_lens()
        cfg = GenerationConfig(max_new_tokens=10, temperature=temperature)

        torch.manual_seed(7)
        out_cached = engine.generate(model, prompts, cfg, seq_lens=seq_lens)

        legacy_model = copy.deepcopy(model)
        legacy_model.get_kv_caches = lambda **kwargs: None  # force fallback
        torch.manual_seed(7)
        out_legacy = engine.generate(legacy_model, prompts, cfg, seq_lens=seq_lens)

        assert out_cached.shape == out_legacy.shape
        assert torch.equal(out_cached, out_legacy)

    def test_stop_token_parity(self, engine):
        """Stop-token handling (write EOS, freeze row) matches the legacy path."""
        model = _tiny_llama()
        prompts, seq_lens = _prompts_and_lens()
        # Pick a stop id from the vocab; both paths must stop identically.
        cfg = GenerationConfig(max_new_tokens=12, temperature=1.0, stop_token_id=13)

        torch.manual_seed(7)
        out_cached = engine.generate(model, prompts, cfg, seq_lens=seq_lens)
        torch.manual_seed(7)
        legacy_model = copy.deepcopy(model)
        legacy_model.get_kv_caches = lambda **kwargs: None
        out_legacy = engine.generate(legacy_model, prompts, cfg, seq_lens=seq_lens)

        assert torch.equal(out_cached, out_legacy)


class TestCacheCorrectness:
    def test_incremental_logits_match_full_forward(self):
        """Decode-step logits from the cache must equal a full forward over
        prompt+token — validates RoPE positions and cache masking."""
        model = _tiny_llama(use_qk_norm=True)
        B, T = 3, 6
        torch.manual_seed(99)
        ids = torch.randint(0, VOCAB, (B, T))

        with torch.no_grad():
            # Full forward over all T tokens
            full = model(ids)["logits"][:, -1, :]

            # Cached: prefill T-1 tokens, then decode the last one
            caches = model.get_kv_caches(batch_size=B, max_len=T + 4)
            assert caches is not None
            model(
                ids[:, :-1],
                position_ids=torch.arange(T - 1).unsqueeze(0).expand(B, -1),
                kv_caches=caches,
            )
            decode_pos = torch.full((B, 1), T - 1, dtype=torch.long)
            incremental = model(ids[:, -1:], position_ids=decode_pos, kv_caches=caches)[
                "logits"
            ][:, -1, :]

        assert torch.allclose(full, incremental, atol=1e-4)

    def test_cache_masks_padding_for_variable_lengths(self):
        """Shorter rows in a padded batch must not attend pad-slot garbage."""
        model = _tiny_llama()
        B, T = 2, 8
        torch.manual_seed(5)
        ids = torch.randint(0, VOCAB, (B, T))
        seq_lens = torch.tensor([T, 3])

        with torch.no_grad():
            # Reference: each row forwarded alone at its true length
            ref_last = []
            for i in range(B):
                li = seq_lens[i].item()
                out = model(ids[i : i + 1, :li])["logits"][0, -1]
                ref_last.append(out)
            ref = torch.stack(ref_last)

            # Cached prefill of the padded batch
            caches = model.get_kv_caches(batch_size=B, max_len=T + 2)
            out = model(
                ids,
                seq_lens=seq_lens,
                position_ids=torch.arange(T).unsqueeze(0).expand(B, -1),
                kv_caches=caches,
            )["logits"]

        for i in range(B):
            li = seq_lens[i].item()
            got = out[i, li - 1]
            assert torch.allclose(
                got, ref[i], atol=1e-4
            ), f"row {i}: cached prefill logits diverge from isolated forward"

    def test_get_kv_caches_shapes(self):
        model = _tiny_llama()
        caches = model.get_kv_caches(batch_size=3, max_len=20)
        assert len(caches) == 2  # n_layer
        for c in caches:
            assert c.k.shape == (3, 2, 20, 8)  # B, n_kv_head, max_len, head_dim
            assert c.v.shape == c.k.shape
            assert c.cursor == 0

    def test_sliding_window_disables_cache(self):
        model = _tiny_llama()
        # Simulate sliding-window attention (not exposed via LlamaConfig).
        model.transformer.h[0].attn.sliding_window = 4
        assert model.get_kv_caches(batch_size=2, max_len=16) is None


class TestEngineIntegration:
    def test_engine_uses_cached_path(self, engine):
        """Sanity check that a Llama actually goes through the fast path:
        the caches handed to the model must be mutated during generate."""
        model = _tiny_llama()
        captured = {}

        original_forward = model.forward

        def spy(input_ids, **kwargs):
            if kwargs.get("kv_caches") is not None:
                captured["decode_calls"] = captured.get("decode_calls", 0) + (
                    1 if input_ids.size(1) == 1 else 0
                )
            return original_forward(input_ids, **kwargs)

        model.forward = spy
        prompts = torch.randint(0, VOCAB, (2, 4))
        cfg = GenerationConfig(max_new_tokens=5, temperature=0.0)
        engine.generate(model, prompts, cfg)

        assert (
            captured.get("decode_calls", 0) == 5
        ), "expected one cached decode step per generated token"

    def test_engine_falls_back_without_cache_support(self, engine):
        """Models without get_kv_caches still generate via the legacy path."""

        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab = VOCAB

            def forward(self, input_ids, **kw):
                logits = torch.zeros(*input_ids.shape, self.vocab)
                logits[..., 9] = 10.0
                return {"logits": logits}

        prompts = torch.tensor([[1, 2, 3]])
        cfg = GenerationConfig(max_new_tokens=3, temperature=0.0)
        out = engine.generate(StubModel(), prompts, cfg)
        assert out.tolist() == [[1, 2, 3, 9, 9, 9]]
