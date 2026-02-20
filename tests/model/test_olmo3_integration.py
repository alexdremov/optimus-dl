import logging

import torch
import pytest
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from optimus_dl.modules.model import build_model


class TestOlmo3Integration:
    """Integration tests for Olmo3 model, comparing against Hugging Face implementation."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name",
        [
            "dralex/olmo3-0.2b-random-ci",
        ],
    )
    def test_hf_olmo3_logits_match(self, model_name):
        """
        Test that Olmo3 loaded via optimus_dl produces identical logits to
        the Hugging Face Olmo3 implementation.
        """
        if not model_name:
            pytest.skip("No suitable Olmo3 model for integration test.")

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float32
        )
        hf_model.eval()
        hf_model.float()

        optimus_model = build_model(
            {
                "_name": "preset_hfolmo3",
                "hf_model_name": model_name,
            }
        )
        optimus_model.eval()
        optimus_model.float()

        # Compare outputs
        hf_outputs = {}
        optimus_outputs = {}
        hf_masks = {}

        def get_hook(storage, name):
            def hook(model, input, output):
                storage[name] = output[0] if isinstance(output, tuple) else output

            return hook

        def get_mask_hook(name):
            def hook(model, input, kwargs):
                # signature: forward(hidden_states, attention_mask=None, ...)
                # In newer transformers, attention_mask is often in kwargs
                mask = kwargs.get("attention_mask")
                if mask is None and len(input) > 1:
                    mask = input[1]
                hf_masks[name] = mask

            return hook

        # Hook HF model
        hf_model.model.embed_tokens.register_forward_hook(get_hook(hf_outputs, "wte"))
        for i, layer in enumerate(hf_model.model.layers):
            layer.register_forward_pre_hook(
                get_mask_hook(f"h.{i}.mask"), with_kwargs=True
            )
            layer.self_attn.q_proj.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.q_proj")
            )
            layer.self_attn.k_proj.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.k_proj")
            )
            layer.self_attn.v_proj.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.v_proj")
            )
            layer.self_attn.q_norm.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.q_norm")
            )
            layer.self_attn.k_norm.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.k_norm")
            )
            layer.self_attn.o_proj.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.attn.wo")
            )
            layer.post_attention_layernorm.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.post_attn_ln")
            )
            layer.mlp.register_forward_hook(get_hook(hf_outputs, f"h.{i}.mlp"))
            layer.post_feedforward_layernorm.register_forward_hook(
                get_hook(hf_outputs, f"h.{i}.post_mlp_ln")
            )

        # Hook Optimus model
        optimus_model.transformer.wte.register_forward_hook(
            get_hook(optimus_outputs, "wte")
        )
        for i, block in enumerate(optimus_model.transformer.h):
            block.attn.wq.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.q_proj")
            )
            block.attn.wk.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.k_proj")
            )
            block.attn.wv.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.v_proj")
            )
            block.attn.q_norm.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.q_norm")
            )
            block.attn.k_norm.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.k_norm")
            )
            block.attn.wo.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.attn.wo")
            )
            # In our implementation, block.ln_1 is applied after attn
            block.ln_1.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.post_attn_ln")
            )
            block.mlp.register_forward_hook(get_hook(optimus_outputs, f"h.{i}.mlp"))
            # In our implementation, block.ln_2 is applied after mlp
            block.ln_2.register_forward_hook(
                get_hook(optimus_outputs, f"h.{i}.post_mlp_ln")
            )

        # Run Inference
        torch.manual_seed(42)
        # Use smaller sequence length for easier debugging
        seq_len = 164
        input_ids = torch.randint(0, hf_config.vocab_size, (1, seq_len))

        # Debug: Print HF keys
        # print("HF Model Keys:", list(hf_model.state_dict().keys()))

        with torch.no_grad():
            hf_out = hf_model(input_ids).logits
            optimus_out = optimus_model(input_ids)["logits"]

        # Compare masks
        for name, mask in hf_masks.items():
            if mask is not None:
                # Convert mask to boolean if it's additive
                if mask.dtype == torch.float32:
                    mask_bool = mask > -1e4
                else:
                    mask_bool = mask.bool()

                # Create our mask for comparison
                T = seq_len
                q_idx = torch.arange(T).view(-1, 1)
                kv_idx = torch.arange(T).view(1, -1)

                # Extract layer index from name like 'h.1.mask'
                import re

                layer_match = re.search(r"h\.(\d+)\.mask", name)
                layer_idx = int(layer_match.group(1)) if layer_match else 0

                is_sliding = (
                    optimus_model.config.layer_types[layer_idx] == "sliding_attention"
                )
                if is_sliding:
                    our_mask = (q_idx >= kv_idx) & (
                        q_idx - kv_idx < hf_config.sliding_window
                    )
                else:
                    our_mask = q_idx >= kv_idx

                # HF mask is (B, 1, T, T)
                hf_mask_2d = mask_bool[0, 0]
                mask_diff = (hf_mask_2d != our_mask).sum().item()
                logging.info(f"Mask {name} diff count: {mask_diff}")
                if mask_diff > 0:
                    logging.info(f"DIVERGENCE at Mask {name}")

        # Compare intermediate outputs
        head_dim = optimus_model.config.head_dim
        n_head = optimus_model.config.n_head
        n_kv_head = optimus_model.config.n_kv_head

        for name in sorted(hf_outputs.keys()):
            if name not in optimus_outputs:
                continue
            hf_val = hf_outputs[name]
            opt_val = optimus_outputs[name]

            def to_interleaved(w, heads, dim):
                B, T, D = w.shape
                w = w.view(B, T, heads, dim)
                w1 = w[:, :, :, : dim // 2]
                w2 = w[:, :, :, dim // 2 :]
                w_new = torch.stack((w1, w2), dim=-1).view(B, T, D)
                return w_new

            # Special handling for interleaved/half-half permutation if comparing projection outputs
            if "attn.q_proj" in name or "attn.q_norm" in name:
                # permute HF output from half-half to interleaved
                hf_val = to_interleaved(hf_val, n_head, head_dim)

            if "attn.k_proj" in name or "attn.k_norm" in name:
                hf_val = to_interleaved(hf_val, n_kv_head, head_dim)

            max_diff = (hf_val.float() - opt_val.float()).abs().max().item()
            logging.info(f"Layer {name} max diff: {max_diff}")
            # if max_diff > 1e-4:
            #     print(f"DIVERGENCE at {name}")

        # Verify Final Logits
        max_diff = (optimus_out.float() - hf_out.float()).abs().max().item()
        logging.info(f"Final logits max diff: {max_diff}")

        assert torch.allclose(
            optimus_out.float(), hf_out.float(), atol=1e-4, rtol=1e-4
        ), f"Logits mismatch! Max diff: {max_diff}"
