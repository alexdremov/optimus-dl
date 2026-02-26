import torch

from optimus_dl.modules.model.llama2 import (
    Llama,
    LlamaConfig,
)
from optimus_dl.modules.model.olmo3 import (
    Olmo3,
    Olmo3Config,
)
from optimus_dl.modules.model.qwen3 import (
    Qwen3,
    Qwen3Config,
)


def _test_consistency(model, device):
    """Generic helper to test consistency between padded and flat layouts."""
    model.to(device)
    model.eval()

    torch.manual_seed(42)
    B = 3
    doc_lengths = [5, 12, 8]
    max_len = max(doc_lengths)
    vocab_size = model.config.vocab_size

    # 1. Prepare Padded Layout
    input_ids_padded = torch.zeros(B, max_len, dtype=torch.long)
    for i, length in enumerate(doc_lengths):
        input_ids_padded[i, :length] = torch.randint(0, vocab_size, (length,))
    seq_lens = torch.tensor(doc_lengths).to(device)
    input_ids_padded = input_ids_padded.to(device)

    # 2. Prepare Flat Layout
    input_ids_flat = []
    for i, length in enumerate(doc_lengths):
        input_ids_flat.append(input_ids_padded[i, :length])
    input_ids_flat = torch.cat(input_ids_flat).unsqueeze(0).to(device)

    # Metadata for Flat Layout
    document_ids = []
    position_ids = []
    for i, length in enumerate(doc_lengths):
        document_ids.extend([i] * length)
        position_ids.extend(range(length))

    document_ids = torch.tensor(document_ids).unsqueeze(0).to(device)
    position_ids = torch.tensor(position_ids).unsqueeze(0).to(device)

    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32).to(device)
    cu_seqlens[1:] = torch.tensor(doc_lengths).cumsum(0).to(device)
    max_seqlen = max(doc_lengths)

    with torch.no_grad():
        # A. Forward with Padded Layout
        # We use seq_lens to handle padding in attention
        out_padded = model(input_ids_padded, seq_lens=seq_lens)["logits"]

        # B. Forward with Flat Layout + Document/Position IDs
        # This uses the flex_attention/SDPA path
        out_flat_masked = model(
            input_ids_flat, document_ids=document_ids, position_ids=position_ids
        )["logits"]

        # C. Forward with Flat Layout + VarLen metadata
        # This uses the varlen_attn path (CUDA kernel or CPU fallback)
        out_flat_varlen = model(
            input_ids_flat,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )["logits"]

    # Comparison: logits_padded (B, T, V) vs logits_flat (1, sum_T, V)
    # We compare only the valid tokens in the padded output
    flat_idx = 0
    for i, length in enumerate(doc_lengths):
        padded_doc_logits = out_padded[i, :length, :]
        flat_masked_doc_logits = out_flat_masked[0, flat_idx : flat_idx + length, :]
        flat_varlen_doc_logits = out_flat_varlen[0, flat_idx : flat_idx + length, :]

        # Check Masked Flat vs Padded
        torch.testing.assert_close(
            padded_doc_logits, flat_masked_doc_logits, rtol=1e-5, atol=1e-5
        )

        # Check VarLen Flat vs Padded
        torch.testing.assert_close(
            padded_doc_logits, flat_varlen_doc_logits, rtol=1e-5, atol=1e-5
        )

        flat_idx += length


def test_llama_flat_consistency(device):
    config = LlamaConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        sequence_length=1024,
    )
    model = Llama(config)
    _test_consistency(model, device)


def test_olmo_flat_consistency(device):
    # Olmo3 alternating layers: 0=sliding, 1=full
    config = Olmo3Config(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        sequence_length=1024,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
    )
    model = Olmo3(config)
    _test_consistency(model, device)


def test_qwen_flat_consistency(device):
    config = Qwen3Config(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        sequence_length=1024,
    )
    model = Qwen3(config)
    _test_consistency(model, device)


def measure_activation_memory(model, input_ids, **kwargs):
    """Measures total size of activations (outputs of key modules) during a forward pass."""
    total_bytes = 0
    hooks = []

    def hook_fn(module, input, output):
        nonlocal total_bytes
        if isinstance(output, torch.Tensor):
            total_bytes += output.numel() * output.element_size()
        elif isinstance(output, (list, tuple)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    total_bytes += t.numel() * t.element_size()

    # Track outputs of attention and MLP, and transformer blocks
    # We target modules that tend to hold large activations
    for name, module in model.named_modules():
        # We target modules that tend to hold large activations
        if any(target in name for target in ["attn", "mlp"]):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_ids, **kwargs)

    for h in hooks:
        h.remove()
    return total_bytes


def test_flat_memory_efficiency(device):
    """Verify that flat forward uses less activation memory, proportional to token count."""
    config = LlamaConfig(
        vocab_size=1000, n_layer=4, n_head=8, n_embd=512, sequence_length=1024
    )
    model = Llama(config).to(device).eval()

    B = 4
    # One long document, three short ones
    # Token count padded: 4 * 128 = 512
    # Token count flat: 128 + 16 * 3 = 176
    doc_lengths = [128, 16, 16, 16]
    max_len = 128
    total_tokens_padded = B * max_len
    total_tokens_flat = sum(doc_lengths)

    # 1. Prepare Padded
    input_ids_padded = torch.randint(0, 1000, (B, max_len)).to(device)
    seq_lens = torch.tensor(doc_lengths).to(device)

    # 2. Prepare Flat
    input_ids_flat = []
    for i, length in enumerate(doc_lengths):
        input_ids_flat.append(input_ids_padded[i, :length])
    input_ids_flat = torch.cat(input_ids_flat).unsqueeze(0).to(device)

    # Flat metadata
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32).to(device)
    cu_seqlens[1:] = torch.tensor(doc_lengths).cumsum(0).to(device)
    max_seqlen = max(doc_lengths)

    # Measure memory
    mem_padded = measure_activation_memory(model, input_ids_padded, seq_lens=seq_lens)
    mem_flat = measure_activation_memory(
        model, input_ids_flat, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )

    # Ratio of tokens
    token_ratio = total_tokens_flat / total_tokens_padded  # 176 / 512 = 0.34375
    # Ratio of memory
    mem_ratio = mem_flat / mem_padded

    # Mem ratio should be very close to token ratio
    # Allow small tolerance for fixed activations if any
    assert (
        mem_ratio < token_ratio
    ), f"Memory ratio {mem_ratio:.4f} exceeds expected token ratio {token_ratio:.4f}"
    assert (
        mem_ratio < 0.5
    ), "Flat memory should be significantly less than padded memory in this scenario"
