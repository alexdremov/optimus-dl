import logging

import torch
import pytest
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from optimus_dl.modules.model import build_model


class TestQwen3Integration:
    """Integration tests for Qwen3 model, comparing against Hugging Face implementation."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name",
        [
            "snake7gun/tiny-random-qwen3",
            "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random",
            "Qwen/Qwen3-0.6B",
        ],
    )
    def test_hf_qwen3_logits_match(self, model_name):
        """
        Test that Qwen3 loaded via optimus_dl produces identical logits to
        the Hugging Face Qwen3 implementation.
        """
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float32
        )
        hf_model.eval()
        hf_model.float()

        optimus_model = build_model(
            {
                "_name": "preset_hfqwen3",
                "hf_model_name": model_name,
            }
        )
        optimus_model.eval()
        optimus_model.float()

        print(optimus_model)
        print("=======")
        print(hf_model)

        # --- Add Hooks ---
        hf_outputs = {}
        optimus_outputs = {}

        def get_hook(storage, name):
            def hook(model, input, output):
                storage[name] = output[0] if isinstance(output, tuple) else output

            return hook

        # Hook HF model
        hf_model.model.embed_tokens.register_forward_hook(get_hook(hf_outputs, "wte"))
        for i, layer in enumerate(hf_model.model.layers):
            layer.input_layernorm.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.ln_1")
            )
            # Attention block hooks
            # layer.self_attn.q_proj.register_forward_hook(get_hook(hf_outputs, f"h.{i}.attn.q_proj"))
            # layer.self_attn.k_proj.register_forward_hook(get_hook(hf_outputs, f"h.{i}.attn.k_proj"))
            layer.self_attn.v_proj.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.v_proj")
            )
            # Qwen3 specific QK Norms
            # if hasattr(layer.self_attn, "q_norm"):
            # layer.self_attn.q_norm.register_forward_hook(get_hook(hf_outputs, f"h.{i}.attn.q_norm"))
            # layer.self_attn.k_norm.register_forward_hook(get_hook(hf_outputs, f"h.{i}.attn.k_norm"))
            layer.self_attn.o_proj.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.o_proj")
            )

            layer.post_attention_layernorm.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.ln_2")
            )
            layer.mlp.register_forward_hook(get_hook(hf_outputs, f"h.{i}.mlp"))
        hf_model.model.norm.register_forward_hook(get_hook(hf_outputs, "ln_f"))

        # Hook Optimus model
        optimus_model.transformer.wte.register_forward_hook(
            get_hook(optimus_outputs, "wte")
        )
        for i, block in enumerate(optimus_model.transformer.h):
            block.ln_1.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.ln_1"))
            # block.attn.wq.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.attn.q_proj"))
            # block.attn.wk.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.attn.k_proj"))
            block.attn.wv.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.v_proj")
            )
            # block.attn.q_norm.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.attn.q_norm"))
            # block.attn.k_norm.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.attn.k_norm"))
            block.attn.wo.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.o_proj")
            )
            block.ln_2.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.ln_2"))
            block.mlp.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.mlp"))
        optimus_model.transformer.ln_f.register_forward_hook(
            get_hook(optimus_outputs, "ln_f")
        )

        # 5. Run Inference
        torch.manual_seed(42)
        input_ids = torch.randint(0, hf_config.vocab_size, (1, 10))

        with torch.no_grad():
            hf_out = hf_model(input_ids).logits
            optimus_out = optimus_model(input_ids)["logits"]

        # Compare intermediate outputs
        for name in hf_outputs:
            if name not in optimus_outputs:
                logging.info(f"Skipping check for {name}, not found in optimus outputs")
                continue

            hf_tensor = hf_outputs[name]
            optimus_tensor = optimus_outputs[name]

            max_diff = (hf_tensor.float() - optimus_tensor.float()).abs().max().item()
            logging.info(f"Layer '{name}' max diff: {max_diff}")
            assert torch.allclose(
                hf_tensor.float(),
                optimus_tensor.float(),
                atol=1e-5 if len(hf_outputs) < 10 else 1e-3,
            ), f"Mismatch at layer '{name}'. Max diff: {max_diff}"

        # 6. Verify Final Logits
        max_diff = (optimus_out.float() - hf_out.float()).abs().max().item()
        logging.info(f"Final logits max diff: {max_diff}")

        assert torch.allclose(
            optimus_out.float(), hf_out.float(), atol=1e-4, rtol=1e-4
        ), f"Logits mismatch! Max diff: {max_diff}"
