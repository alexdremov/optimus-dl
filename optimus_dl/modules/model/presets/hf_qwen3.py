"""Preset for loading Hugging Face Qwen3 models."""

import logging
from dataclasses import dataclass

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from optimus_dl.modules.model import register_model
from optimus_dl.modules.model.qwen3 import (
    Qwen3,
    Qwen3Config,
)

logger = logging.getLogger(__name__)


@dataclass
class HFQwen3Config(Qwen3Config):
    hf_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    load_weights: bool = True


@register_model("preset_hfqwen3", HFQwen3Config)
def make_hf_qwen3_model(cfg: HFQwen3Config, **_):
    """Create a Qwen3 model loaded with weights from Hugging Face."""
    logger.info(f"Loading HF model: {cfg.hf_model_name}")

    # Load HF config
    hf_config = AutoConfig.from_pretrained(cfg.hf_model_name, trust_remote_code=True)

    # Update local config from HF config
    cfg.n_layer = hf_config.num_hidden_layers
    cfg.n_head = hf_config.num_attention_heads
    cfg.n_embd = hf_config.hidden_size
    cfg.vocab_size = hf_config.vocab_size
    cfg.sequence_length = getattr(hf_config, "max_position_embeddings", 32768)
    cfg.block_size = cfg.sequence_length
    cfg.rmsnorm_eps = hf_config.rms_norm_eps
    cfg.intermediate_size = getattr(hf_config, "intermediate_size", None)
    cfg.attention_bias = getattr(
        hf_config, "attention_bias", False
    )  # Default to False if not present
    cfg.rope_theta = getattr(hf_config, "rope_theta", 5000000.0)
    cfg.head_dim = getattr(hf_config, "head_dim", cfg.n_embd // cfg.n_head)

    # Ensure correct tying behavior
    cfg.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # Handle GQA
    if hasattr(hf_config, "num_key_value_heads"):
        cfg.n_kv_head = hf_config.num_key_value_heads
    else:
        cfg.n_kv_head = hf_config.num_attention_heads

    # Initialize local Qwen3 model
    model = Qwen3(cfg)

    if not cfg.load_weights:
        return model

    # Load HF model weights
    logger.info("Loading HF model weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    hf_sd = hf_model.state_dict()

    # Map weights
    sd = model.state_dict()
    loaded_keys = set()

    # Helper to permute weights for RoPE (HF half-half -> Local interleaved)
    def permute_rope(w, n_heads, head_dim):
        # w shape: (n_heads * head_dim, ...)
        # or w shape: (head_dim,) for Norms
        original_shape = w.shape

        # Handle Norm weights which are (head_dim,)
        if w.ndim == 1:
            # Broadcast to 1 head effectively for reshaping logic
            w = w.view(1, head_dim, 1)  # (1, D, 1)
            w1 = w[:, : head_dim // 2, :]
            w2 = w[:, head_dim // 2 :, :]
            w_new = torch.stack((w1, w2), dim=2)  # (1, D/2, 2, 1)
            return w_new.reshape(original_shape)

        w = w.view(n_heads, head_dim, -1)
        w1 = w[:, : head_dim // 2, :]
        w2 = w[:, head_dim // 2 :, :]
        # Interleave: (x0, x_half, x1, x_half+1...)
        w_new = torch.stack((w1, w2), dim=2)
        return w_new.reshape(original_shape)

    # Helper to copy weights
    def copy_weight(src_key, dest_key, permute=False, n_heads=None, head_dim=None):
        nonlocal hf_sd
        if src_key not in hf_sd:
            if dest_key in sd:
                logger.warning(f"Missing key in HF model: {src_key}")
            return
        w = hf_sd[src_key]

        if permute:
            if n_heads is None and head_dim is None:
                # Infer for norm: w is (head_dim,)
                head_dim = w.shape[0]
                n_heads = 1  # Dummy
            assert head_dim is not None
            w = permute_rope(w, n_heads, head_dim)

        if sd[dest_key].shape != w.shape:
            logger.warning(
                f"Shape mismatch for {dest_key}: {sd[dest_key].shape} vs {w.shape}."
            )
            w = w.view(sd[dest_key].shape)

        sd[dest_key].copy_(w)
        loaded_keys.add(dest_key)

    logger.info("Copying weights...")

    # Embeddings
    copy_weight("model.embed_tokens.weight", "transformer.wte.weight")

    # Layers
    head_dim = cfg.head_dim
    for i in range(cfg.n_layer):
        # Attention
        copy_weight(
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"transformer.h.{i}.attn.wq.weight",
            permute=True,
            n_heads=cfg.n_head,
            head_dim=head_dim,
        )
        copy_weight(
            f"model.layers.{i}.self_attn.q_proj.bias",
            f"transformer.h.{i}.attn.wq.bias",
            permute=True,
            n_heads=cfg.n_head,
            head_dim=head_dim,
        )
        copy_weight(
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"transformer.h.{i}.attn.wk.weight",
            permute=True,
            n_heads=cfg.n_kv_head,
            head_dim=head_dim,
        )
        copy_weight(
            f"model.layers.{i}.self_attn.k_proj.bias",
            f"transformer.h.{i}.attn.wk.bias",
            permute=True,
            n_heads=cfg.n_kv_head,
            head_dim=head_dim,
        )
        copy_weight(
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"transformer.h.{i}.attn.wv.weight",
        )
        copy_weight(
            f"model.layers.{i}.self_attn.v_proj.bias",
            f"transformer.h.{i}.attn.wv.bias",
        )
        copy_weight(
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"transformer.h.{i}.attn.wo.weight",
        )
        copy_weight(
            f"model.layers.{i}.self_attn.o_proj.bias",
            f"transformer.h.{i}.attn.wo.bias",
        )

        # Q/K Norms - MUST PERMUTE because the input to them is permuted/interleaved!
        copy_weight(
            f"model.layers.{i}.self_attn.q_norm.weight",
            f"transformer.h.{i}.attn.q_norm.weight",
            permute=True,
            head_dim=head_dim,
        )
        copy_weight(
            f"model.layers.{i}.self_attn.k_norm.weight",
            f"transformer.h.{i}.attn.k_norm.weight",
            permute=True,
            head_dim=head_dim,
        )

        # MLP
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
    missing_keys = expected_keys - loaded_keys

    if cfg.tie_word_embeddings:
        if "transformer.wte.weight" in loaded_keys and "lm_head.weight" in missing_keys:
            missing_keys.remove("lm_head.weight")
        if "lm_head.weight" in loaded_keys and "transformer.wte.weight" in missing_keys:
            missing_keys.remove("transformer.wte.weight")

    if missing_keys:
        logger.warning(f"Missing keys in loaded model: {missing_keys}")

    logger.info("Weights loaded successfully.")

    del hf_model
    del hf_sd
    import gc

    gc.collect()

    return model
