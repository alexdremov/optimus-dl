import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

import torch
from omegaconf import OmegaConf

from optimus_dl.core.registry import build
from optimus_dl.modules.model.presets.hf_llama import HFLlamaConfig


class TestHFLlamaLoading(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.mock_config = MagicMock()
        self.mock_config.num_hidden_layers = 2
        self.mock_config.num_attention_heads = 4
        self.mock_config.num_key_value_heads = 4
        self.mock_config.hidden_size = 32
        self.mock_config.tie_word_embeddings = False
        self.mock_config.intermediate_size = None
        self.mock_config.rope_theta = 10000.0

        # Calculate expected hidden dim for MLP
        # Llama logic:
        # hidden_dim = 4 * n_embd
        # hidden_dim = int(2 * hidden_dim / 3)
        # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_embd = 32
        multiple_of = 32
        hidden_dim = 4 * n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.mlp_hidden_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.mock_config.vocab_size = 100
        self.mock_config.max_position_embeddings = 64
        self.mock_config.rms_norm_eps = 1e-5

        # Mock model state dict
        self.mock_state_dict = {}
        # Embeddings
        self.mock_state_dict["model.embed_tokens.weight"] = torch.randn(100, 32)
        # Layers
        for i in range(2):
            self.mock_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = (
                torch.randn(32, 32)
            )
            self.mock_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = (
                torch.randn(32, 32)
            )
            self.mock_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = (
                torch.randn(32, 32)
            )
            self.mock_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = (
                torch.randn(32, 32)
            )

            self.mock_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = (
                torch.randn(self.mlp_hidden_dim, 32)
            )
            self.mock_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(
                self.mlp_hidden_dim, 32
            )
            self.mock_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = (
                torch.randn(32, self.mlp_hidden_dim)
            )

            self.mock_state_dict[f"model.layers.{i}.input_layernorm.weight"] = (
                torch.randn(32)
            )
            self.mock_state_dict[
                f"model.layers.{i}.post_attention_layernorm.weight"
            ] = torch.randn(32)
        # Final Norm
        self.mock_state_dict["model.norm.weight"] = torch.randn(32)
        # LM Head
        self.mock_state_dict["lm_head.weight"] = torch.randn(100, 32)

    @patch("optimus_dl.modules.model.presets.hf_llama.AutoConfig")
    @patch("optimus_dl.modules.model.presets.hf_llama.AutoModelForCausalLM")
    def test_make_hf_llama_model(self, mock_automodel, mock_autoconfig):
        mock_autoconfig.from_pretrained.return_value = self.mock_config

        mock_model_instance = MagicMock()
        mock_model_instance.state_dict.return_value = self.mock_state_dict
        mock_automodel.from_pretrained.return_value = mock_model_instance

        cfg = HFLlamaConfig(
            _name="preset_hfllama2",
            hf_model_name="mock-model",
            load_weights=True,
            multiple_of=32,
        )
        cfg = OmegaConf.structured(cfg)

        model = build("model", cfg)

        self.assertIsNotNone(model)
        self.assertEqual(model.config.n_layer, 2)
        self.assertEqual(model.config.n_head, 4)
        self.assertEqual(model.config.n_embd, 32)

        # Check if weights are copied
        # We can check values roughly or just shape

        # Check Q
        q_weight = model.transformer.h[0].attn.wq.weight
        expected_shape_q = (4 * (32 // 4), 32)  # (n_head * head_dim, n_embd) = (32, 32)
        self.assertEqual(q_weight.shape, expected_shape_q)

        # Check K
        k_weight = model.transformer.h[0].attn.wk.weight
        expected_shape_k = (
            4 * (32 // 4),
            32,
        )  # (n_kv_head * head_dim, n_embd) = (32, 32)
        self.assertEqual(k_weight.shape, expected_shape_k)

    @patch("optimus_dl.modules.model.presets.hf_llama.AutoConfig")
    def test_make_hf_llama_model_no_weights(self, mock_autoconfig):
        mock_autoconfig.from_pretrained.return_value = self.mock_config

        cfg = HFLlamaConfig(
            _name="preset_hfllama2", hf_model_name="mock-model", load_weights=False
        )
        cfg = OmegaConf.structured(cfg)

        model = build("model", cfg)
        self.assertIsNotNone(model)
        # Should be initialized but random weights (not checked here)

    @patch("optimus_dl.modules.model.presets.hf_llama.AutoConfig")
    def test_gqa_config(self, mock_autoconfig):
        # Mock GQA config
        self.mock_config.num_attention_heads = 4
        self.mock_config.num_key_value_heads = 2
        mock_autoconfig.from_pretrained.return_value = self.mock_config

        cfg = HFLlamaConfig(
            _name="preset_hfllama2", hf_model_name="mock-gqa-model", load_weights=False
        )
        cfg = OmegaConf.structured(cfg)

        model = build("model", cfg)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.n_kv_head, 2)
        self.assertEqual(model.config.n_head, 4)

    def test_real_llama_config_loading(self):
        """Test loading a real model config (no weights) to ensure mapping logic works with real HF artifacts."""
        try:
            from transformers import AutoConfig

            # Attempt to fetch config - this might fail if no internet or auth,
            # so we'll catch potential connection errors or auth errors
            try:
                # TinyLlama 1.1B uses GQA (32 attn heads, 4 kv heads)
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                AutoConfig.from_pretrained(model_name)
            except Exception as e:
                print(f"Skipping real config test due to access/connection issue: {e}")
                return

            cfg = HFLlamaConfig(
                _name="preset_hfllama2", hf_model_name=model_name, load_weights=False
            )
            cfg = OmegaConf.structured(cfg)

            model = build("model", cfg)
            self.assertIsNotNone(model)

            # TinyLlama specific checks
            self.assertEqual(model.config.n_embd, 2048)
            self.assertEqual(model.config.n_head, 32)
            self.assertEqual(model.config.n_kv_head, 4)  # Verified from HF config
            self.assertEqual(model.config.n_layer, 22)

        except Exception as e:
            self.fail(f"Failed to load real Llama config: {e}")


if __name__ == "__main__":
    unittest.main()
