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
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    ],
)
def test_logits_matching(model_name, device):
    print(f"Loading HF model: {model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.float()
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading Optimus model...")
    cfg = HFLlamaConfig(
        _name="preset_hfllama2", hf_model_name=model_name, load_weights=True
    )
    cfg = OmegaConf.structured(cfg)
    opt_model = build("model", cfg)
    # Both models on the SAME device: cross-device float noise (~2e-4 on MPS)
    # otherwise dominates and masks real mismatches.
    hf_model.to(device)
    opt_model.to(device)
    opt_model.float()
    opt_model.eval()

    print(hf_model.config)
    print("=======")
    print(opt_model)

    input_text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Running inference...")
    with torch.no_grad():
        hf_out = hf_model(**inputs)
        opt_out = opt_model(inputs["input_ids"])

    hf_logits = hf_out.logits.cpu()
    opt_logits = opt_out["logits"].cpu()

    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Opt logits shape: {opt_logits.shape}")

    # Check mean diff
    diff = (hf_logits - opt_logits).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    print(f"Mean diff: {mean_diff}")
    print(f"Max diff: {max_diff}")

    # We expect very close match. Measured backend noise levels (fp32):
    #   cpu: ~1e-4, mps: ~2.5e-4 for 30-layer models (both HF-vs-HF and
    #   HF-vs-Ours). Structural bugs (wrong RoPE/weights) produce O(0.1+).
    assert np.allclose(hf_logits.numpy(), opt_logits.numpy(), atol=1e-3, rtol=1e-3), (
        f"Logits mismatch! Max diff: {max_diff}, Mean diff: {mean_diff}",
    )
