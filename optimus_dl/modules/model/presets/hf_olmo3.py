"""Preset for loading Hugging Face Olmo3 models."""

import logging
from dataclasses import dataclass

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from optimus_dl.modules.model import register_model
from optimus_dl.modules.model.olmo3 import (
    Olmo3,
    Olmo3Config,
)
from optimus_dl.modules.model.presets.utils import (
    WeightMapper,
    update_config_from_hf,
)

logger = logging.getLogger(__name__)


@dataclass
class HFOlmo3Config(Olmo3Config):
    hf_model_name: str = "allenai/Olmo-3-1025-7B"
    load_weights: bool = True


@register_model("preset_hfolmo3", HFOlmo3Config)
def make_hf_olmo3_model(cfg: HFOlmo3Config, **_):
    """Create an Olmo3 model loaded with weights from Hugging Face."""
    logger.info(f"Loading HF model: {cfg.hf_model_name}")

    # Load HF config
    hf_config = AutoConfig.from_pretrained(cfg.hf_model_name, trust_remote_code=True)

    # Update local config from HF config
    update_config_from_hf(cfg, hf_config)
    cfg.sliding_window = getattr(hf_config, "sliding_window", 4096)
    cfg.layer_types = getattr(
        hf_config,
        "layer_types",
        [
            "full_attention",
        ]
        * cfg.n_layer,
    )
    cfg.rope_scaling = getattr(hf_config, "rope_scaling", None)
    cfg.attention_bias = getattr(hf_config, "attention_bias", False)

    # Initialize local Olmo3 model
    model = Olmo3(cfg)

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
    mapper = WeightMapper(hf_sd, model.state_dict())

    logger.info("Copying weights...")

    # Embeddings
    mapper.copy("model.embed_tokens.weight", "transformer.wte.weight")

    # Layers
    for i in range(cfg.n_layer):
        # Attention
        mapper.copy(
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"transformer.h.{i}.attn.wq.weight",
            permute=True,
            n_heads=cfg.n_head,
            head_dim=cfg.head_dim,
        )
        if cfg.attention_bias:
            mapper.copy(
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"transformer.h.{i}.attn.wq.bias",
                permute=True,
                n_heads=cfg.n_head,
                head_dim=cfg.head_dim,
            )

        mapper.copy(
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"transformer.h.{i}.attn.wk.weight",
            permute=True,
            n_heads=cfg.n_kv_head,
            head_dim=cfg.head_dim,
        )
        if cfg.attention_bias:
            mapper.copy(
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"transformer.h.{i}.attn.wk.bias",
                permute=True,
                n_heads=cfg.n_kv_head,
                head_dim=cfg.head_dim,
            )

        mapper.copy(
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"transformer.h.{i}.attn.wv.weight",
        )
        if cfg.attention_bias:
            mapper.copy(
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"transformer.h.{i}.attn.wv.bias",
            )

        mapper.copy(
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"transformer.h.{i}.attn.wo.weight",
        )
        if cfg.attention_bias:
            mapper.copy(
                f"model.layers.{i}.self_attn.o_proj.bias",
                f"transformer.h.{i}.attn.wo.bias",
            )

        # Q/K Norms
        mapper.copy(
            f"model.layers.{i}.self_attn.q_norm.weight",
            f"transformer.h.{i}.attn.q_norm.weight",
            permute=True,
            n_heads=cfg.n_head,
            head_dim=cfg.head_dim,
        )
        mapper.copy(
            f"model.layers.{i}.self_attn.k_norm.weight",
            f"transformer.h.{i}.attn.k_norm.weight",
            permute=True,
            n_heads=cfg.n_kv_head,
            head_dim=cfg.head_dim,
        )

        # MLP
        mapper.copy(
            f"model.layers.{i}.mlp.gate_proj.weight", f"transformer.h.{i}.mlp.w1.weight"
        )
        mapper.copy(
            f"model.layers.{i}.mlp.up_proj.weight", f"transformer.h.{i}.mlp.w2.weight"
        )
        mapper.copy(
            f"model.layers.{i}.mlp.down_proj.weight",
            f"transformer.h.{i}.mlp.c_proj.weight",
        )

        # Layer Norms
        mapper.copy(
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"transformer.h.{i}.ln_1.weight",
        )
        mapper.copy(
            f"model.layers.{i}.post_feedforward_layernorm.weight",
            f"transformer.h.{i}.ln_2.weight",
        )

    # Final Norm
    mapper.copy("model.norm.weight", "transformer.ln_f.weight")

    # LM Head
    mapper.copy("lm_head.weight", "lm_head.weight")

    # Validation
    mapper.validate(tie_word_embeddings=cfg.tie_word_embeddings)

    del hf_model
    del hf_sd
    import gc

    gc.collect()

    return model
