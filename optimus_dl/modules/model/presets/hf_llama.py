"""Preset for loading Hugging Face Llama models."""

import logging
from dataclasses import dataclass

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from optimus_dl.modules.model import register_model
from optimus_dl.modules.model.llama2 import (
    Llama,
    LlamaConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class HFLlamaConfig(LlamaConfig):
    hf_model_name: str = "meta-llama/Llama-2-7b-hf"
    load_weights: bool = (
        True  # If True, will download and load weights. If False, just config is used (random init)
    )


@register_model("preset_hfllama2", HFLlamaConfig)
def make_hf_llama_model(cfg: HFLlamaConfig, **_):
    """Create a Llama model loaded with weights from Hugging Face."""
    logger.info(f"Loading HF model: {cfg.hf_model_name}")

    # Load HF config
    hf_config = AutoConfig.from_pretrained(cfg.hf_model_name)

    # Update local config from HF config
    # Llama 2 uses 4096 context length usually, but it's in config
    optimus_cfg = cfg
    optimus_cfg.n_layer = hf_config.num_hidden_layers
    optimus_cfg.n_head = hf_config.num_attention_heads
    optimus_cfg.n_embd = hf_config.hidden_size
    optimus_cfg.vocab_size = hf_config.vocab_size
    optimus_cfg.sequence_length = getattr(hf_config, "max_position_embeddings", 2048)
    optimus_cfg.block_size = cfg.sequence_length  # Ensure consistency
    optimus_cfg.rmsnorm_eps = hf_config.rms_norm_eps
    optimus_cfg.intermediate_size = getattr(hf_config, "intermediate_size", None)
    optimus_cfg.rope_theta = getattr(hf_config, "rope_theta", 10000.0)

    # Ensure correct tying behavior from HF config
    optimus_cfg.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # Handle GQA (Grouped Query Attention)
    if hasattr(hf_config, "num_key_value_heads"):
        optimus_cfg.n_kv_head = hf_config.num_key_value_heads
    else:
        optimus_cfg.n_kv_head = hf_config.num_attention_heads

    # Initialize local Llama model
    model = Llama(optimus_cfg)

    if not cfg.load_weights:
        return model

    # Load HF model weights
    logger.info("Loading HF model weights...")
    # Use CPU to avoid OOM during conversion if possible, or device="meta" if supported better
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model_name,
        dtype=torch.float32,  # Load as float32 for safety during copy
        low_cpu_mem_usage=True,
    )
    hf_sd = hf_model.state_dict()

    # Map weights
    sd = model.state_dict()

    # Helper to permute weights for RoPE (HF half-half -> Local interleaved)
    def permute_rope(w, n_heads, head_dim):
        # w shape: (n_heads * head_dim, input_dim)
        w = w.view(n_heads, head_dim, -1)
        w1 = w[:, : head_dim // 2, :]
        w2 = w[:, head_dim // 2 :, :]
        # Interleave: (x0, x_half, x1, x_half+1...)
        w_new = torch.stack((w1, w2), dim=2)
        w_new = w_new.reshape(n_heads, head_dim, -1)
        return w_new.reshape(-1, w.shape[-1])

    loaded_keys = set()

    # Helper to copy weights
    def copy_weight(
        src_key, dest_key, transpose=False, permute=False, n_heads=None, head_dim=None
    ):
        nonlocal hf_sd
        if src_key not in hf_sd:
            logger.warning(f"Missing key in HF model: {src_key}")
            return
        w = hf_sd[src_key]
        if transpose:
            w = w.t()

        if permute:
            assert n_heads is not None and head_dim is not None
            w = permute_rope(w, n_heads, head_dim)

        if sd[dest_key].shape != w.shape:
            logger.warning(
                f"Shape mismatch for {dest_key}: {sd[dest_key].shape} vs {w.shape}. Attempting reshape."
            )
            w = w.view(sd[dest_key].shape)

        sd[dest_key].copy_(w)
        loaded_keys.add(dest_key)

    logger.info("Copying weights...")

    # Embeddings
    copy_weight("model.embed_tokens.weight", "transformer.wte.weight")

    # Layers
    head_dim = optimus_cfg.n_embd // optimus_cfg.n_head
    for i in range(optimus_cfg.n_layer):
        # Attention
        copy_weight(
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"transformer.h.{i}.attn.wq.weight",
            permute=True,
            n_heads=optimus_cfg.n_head,
            head_dim=head_dim,
        )
        copy_weight(
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"transformer.h.{i}.attn.wk.weight",
            permute=True,
            n_heads=optimus_cfg.n_kv_head,
            head_dim=head_dim,
        )
        copy_weight(
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"transformer.h.{i}.attn.wv.weight",
        )
        copy_weight(
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"transformer.h.{i}.attn.wo.weight",
        )

        # MLP
        # w1 -> gate_proj, w2 -> up_proj, c_proj -> down_proj
        copy_weight(
            f"model.layers.{i}.mlp.gate_proj.weight", f"transformer.h.{i}.mlp.w1.weight"
        )
        copy_weight(
            f"model.layers.{i}.mlp.up_proj.weight", f"transformer.h.{i}.mlp.w2.weight"
        )
        copy_weight(
            f"model.layers.{i}.mlp.down_proj.weight",
            f"transformer.h.{i}.mlp.c_proj.weight",
        )

        # Layer Norms
        copy_weight(
            f"model.layers.{i}.input_layernorm.weight", f"transformer.h.{i}.ln_1.weight"
        )
        copy_weight(
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"transformer.h.{i}.ln_2.weight",
        )

    # Final Norm
    copy_weight("model.norm.weight", "transformer.ln_f.weight")

    # LM Head
    copy_weight("lm_head.weight", "lm_head.weight")

    # Validation
    expected_keys = set(sd.keys())
    # Exclude buffers like 'bias' in attention if any (Llama usually none)
    missing_keys = expected_keys - loaded_keys
    # Filter out ignored/optional keys
    missing_keys = {k for k in missing_keys if "inv_freq" not in k and "bias" not in k}

    if optimus_cfg.tie_word_embeddings:
        if "transformer.wte.weight" in loaded_keys and "lm_head.weight" in missing_keys:
            missing_keys.remove("lm_head.weight")
        if "lm_head.weight" in loaded_keys and "transformer.wte.weight" in missing_keys:
            missing_keys.remove("transformer.wte.weight")

    assert not missing_keys, f"Missing keys in loaded model: {missing_keys}"

    logger.info("Weights loaded successfully.")

    # Clean up HF model to free memory
    del hf_model
    del hf_sd
    import gc

    gc.collect()

    return model
