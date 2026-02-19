import torch
import torch.nn as nn

from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.blocks.mlp import SwiGLUMLP
from optimus_dl.modules.model.blocks.rope import precompute_freqs_cis
from optimus_dl.modules.model.olmo3 import (
    Olmo3,
    Olmo3Attention,
    Olmo3Block,
    Olmo3Config,
)


class TestOlmo3RoPE:
    """Tests for Olmo3 RoPE (YaRN) functions"""

    def test_precompute_freqs_cis_yarn_defaults(self):
        """Test precomputation with default (no scaling) config."""
        dim = 64
        seq_len = 100

        # Test without scaling config (should behave like standard RoPE)
        freqs_cis = precompute_freqs_cis(dim, seq_len)
        assert freqs_cis.shape == (seq_len, dim // 2, 2)

        # Check unit magnitude
        magnitudes = freqs_cis.norm(dim=-1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_precompute_freqs_cis_yarn_scaling(self):
        """Test precomputation with YaRN scaling."""
        dim = 64
        seq_len = 100
        scaling_config = {
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "attention_factor": 1.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }

        freqs_cis = precompute_freqs_cis(dim, seq_len, scaling_config=scaling_config)
        assert freqs_cis.shape == (seq_len, dim // 2, 2)

        # With YaRN, magnitudes might not be exactly 1 due to mscale,
        # but in our simplified impl they might be if mscale is 1.0.
        # Let's check dimensions at least.


class TestOlmo3Attention:
    """Tests for Olmo3Attention"""

    def test_init(self):
        config = Olmo3Config(n_embd=256, n_head=4, n_kv_head=4)
        layer_idx = 0
        attn = Olmo3Attention(config, layer_idx)

        assert isinstance(attn.wq, nn.Linear)
        assert isinstance(attn.wk, nn.Linear)
        assert isinstance(attn.wv, nn.Linear)
        assert isinstance(attn.wo, nn.Linear)

    def test_forward_sliding_window(self):
        config = Olmo3Config(
            n_embd=256,
            n_head=4,
            n_kv_head=4,
            sliding_window=10,
            n_layer=1,
            layer_types=["sliding_attention"],
        )
        attn = Olmo3Attention(config, layer_idx=0)

        B, T, C = 2, 20, 256
        x = torch.randn(B, T, C)
        head_dim = C // 4
        freqs_cis = precompute_freqs_cis(head_dim, T)

        out = attn(x, freqs_cis)
        assert out.shape == (B, T, C)

    def test_forward_full_attention(self):
        config = Olmo3Config(
            n_embd=256, n_head=4, n_kv_head=4, layer_types=["full_attention"]
        )
        attn = Olmo3Attention(config, layer_idx=0)

        B, T, C = 2, 20, 256
        x = torch.randn(B, T, C)
        head_dim = C // 4
        freqs_cis = precompute_freqs_cis(head_dim, T)

        out = attn(x, freqs_cis)
        assert out.shape == (B, T, C)


class TestOlmo3Block:
    """Tests for Olmo3Block"""

    def test_init(self):
        config = Olmo3Config(n_embd=256, n_head=4, n_kv_head=4, rmsnorm_eps=1e-6)
        block = Olmo3Block(config, layer_idx=0)

        assert isinstance(block.ln_1, RMSNorm)
        assert isinstance(block.ln_2, RMSNorm)
        assert isinstance(block.attn, Olmo3Attention)
        assert isinstance(block.mlp, SwiGLUMLP)

    def test_forward(self):
        config = Olmo3Config(n_embd=256, n_head=4, n_kv_head=4)
        block = Olmo3Block(config, layer_idx=0)

        B, T, C = 2, 10, 256
        x = torch.randn(B, T, C)
        freqs_cis = precompute_freqs_cis(C // 4, T)

        out = block(x, freqs_cis)
        assert out.shape == (B, T, C)
        assert not torch.allclose(out, x)


class TestOlmo3:
    """Tests for main Olmo3 model"""

    def test_init(self):
        config = Olmo3Config(
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=256,
            layer_types=["sliding_attention", "full_attention"],
        )
        model = Olmo3(config)

        assert len(model.transformer.h) == 2
        assert model.transformer.h[0].attn.layer_type == "sliding_attention"
        assert model.transformer.h[1].attn.layer_type == "full_attention"

    def test_forward(self):
        config = Olmo3Config(
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=256,
            layer_types=["sliding_attention", "full_attention"],
        )
        model = Olmo3(config)

        input_ids = torch.randint(0, 1000, (1, 10))
        out = model(input_ids)

        assert "logits" in out
        assert out["logits"].shape == (1, 10, 1000)

    def test_weight_tying(self):
        config = Olmo3Config(tie_word_embeddings=True, n_kv_head=4)
        model = Olmo3(config)
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_gradient_flow(self):
        config = Olmo3Config(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            layer_types=["sliding_attention", "full_attention"],
        )
        model = Olmo3(config)

        input_ids = torch.randint(0, 100, (1, 10))
        out = model(input_ids)
        loss = out["logits"].sum()
        loss.backward()

        assert model.transformer.wte.weight.grad is not None
        assert model.lm_head.weight.grad is not None
