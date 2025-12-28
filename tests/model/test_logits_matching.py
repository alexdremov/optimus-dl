import numpy as np
import torch
import pytest
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from optimus_dl.core.registry import build
from optimus_dl.modules.model.presets.hf_llama import HFLlamaConfig


@pytest.mark.parametrize(
    "model_name",
    [
        "Intel/tiny-random-llama2",
        "yujiepan/llama-3-tiny-random",
        "AlignmentResearch/Llama-3.3-Tiny-Instruct-boolq",
    ],
)
def test_logits_matching(model_name):
    device = "cpu"  # Use CPU to avoid issues if GPU not available, models are small

    print(f"Loading HF model: {model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    hf_model.float()
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading Optimus model...")
    cfg = HFLlamaConfig(
        _name="preset_hfllama2", hf_model_name=model_name, load_weights=True
    )
    cfg = OmegaConf.structured(cfg)
    opt_model = build("model", cfg)
    opt_model.to(device)
    opt_model.float()
    opt_model.eval()

    print(hf_model.config)
    print(hf_model)
    print("=======")
    print(opt_model)

    input_text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    print("Running inference...")
    with torch.no_grad():
        hf_out = hf_model(**inputs)
        opt_out = opt_model(inputs.input_ids)

    hf_logits = hf_out.logits
    opt_logits = opt_out["logits"]

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Opt logits shape: {opt_logits.shape}")

    # Check mean diff
    diff = (hf_logits - opt_logits).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    print(f"Mean diff: {mean_diff}")
    print(f"Max diff: {max_diff}")

    # We expect very close match, float32 precision
    assert np.allclose(hf_logits.numpy(), opt_logits.numpy(), atol=1e-4, rtol=1e-4), (
        f"Logits mismatch! Max diff: {max_diff}, Mean diff: {mean_diff}",
    )
