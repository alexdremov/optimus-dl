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

    if device.type == "cuda":
        model = model.half()

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

        rtol = 1e-5
        atol = 1e-5
        if device.type == "cuda":
            rtol = 1e-3
            atol = 1e-3

        # Check Masked Flat vs Padded
        torch.testing.assert_close(
            padded_doc_logits, flat_masked_doc_logits, rtol=rtol, atol=atol
        )

        # Check VarLen Flat vs Padded
        torch.testing.assert_close(
            padded_doc_logits, flat_varlen_doc_logits, rtol=rtol, atol=atol
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
    if device.type == "cuda":
        model = model.half()

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
        mem_ratio - token_ratio
    ) / token_ratio < 0.05, (
        f"Memory ratio {mem_ratio:.4f} exceeds expected token ratio {token_ratio:.4f}"
    )
    assert (
        mem_ratio < 0.4
    ), "Flat memory should be significantly less than padded memory in this scenario"


def test_document_masking_independence(device):
    """Verify that changing tokens in one document does not affect outputs of another document in a flat batch."""
    config = LlamaConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        sequence_length=1024,
    )
    model = Llama(config).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    # Two documents: doc1 (10 tokens), doc2 (10 tokens)
    len1, len2 = 10, 10
    total_len = len1 + len2

    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (1, total_len)).to(device)
    document_ids = torch.tensor([0] * len1 + [1] * len2).unsqueeze(0).to(device)
    position_ids = (
        torch.tensor(list(range(len1)) + list(range(len2))).unsqueeze(0).to(device)
    )

    # 1. Forward pass 1 (Flex/SDPA path)
    with torch.no_grad():
        out1 = model(input_ids, document_ids=document_ids, position_ids=position_ids)[
            "logits"
        ]
        doc2_logits_orig = out1[0, len1:, :].clone()

    # 2. Modify doc1 tokens
    input_ids_modified = input_ids.clone()
    # Change first document tokens completely
    input_ids_modified[0, :len1] = (input_ids[0, :len1] + 1) % 1000

    # 3. Forward pass 2 (Flex/SDPA path)
    with torch.no_grad():
        out2 = model(
            input_ids_modified, document_ids=document_ids, position_ids=position_ids
        )["logits"]
        doc2_logits_new = out2[0, len1:, :]

    # Check independence for flex/SDPA path
    torch.testing.assert_close(doc2_logits_orig, doc2_logits_new)

    # 4. Repeat check for varlen path
    cu_seqlens = torch.tensor([0, len1, total_len], dtype=torch.int32).to(device)
    max_seqlen = max(len1, len2)

    with torch.no_grad():
        out_v1 = model(
            input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )["logits"]
        doc2_logits_v_orig = out_v1[0, len1:, :].clone()

        out_v2 = model(
            input_ids_modified,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )["logits"]
        doc2_logits_v_new = out_v2[0, len1:, :]

    torch.testing.assert_close(doc2_logits_v_orig, doc2_logits_v_new)


def test_causal_masking_within_document(device):
    """Verify that changing a token in a document affects only subsequent tokens (causal property)."""
    config = LlamaConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        sequence_length=1024,
    )
    model = Llama(config).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    length = 20
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (1, length)).to(device)
    document_ids = torch.zeros(1, length, dtype=torch.long).to(device)
    position_ids = torch.arange(length).unsqueeze(0).to(device)

    # Change token at index 10
    modify_idx = 10

    # 1. Forward pass 1
    with torch.no_grad():
        out1 = model(input_ids, document_ids=document_ids, position_ids=position_ids)[
            "logits"
        ]
        logits_orig = out1[0].clone()

    # 2. Modify token at modify_idx
    input_ids_modified = input_ids.clone()
    input_ids_modified[0, modify_idx] = (input_ids[0, modify_idx] + 1) % 1000

    # 3. Forward pass 2
    with torch.no_grad():
        out2 = model(
            input_ids_modified, document_ids=document_ids, position_ids=position_ids
        )["logits"]
        logits_new = out2[0]

    # Preceding tokens should be UNCHANGED
    torch.testing.assert_close(logits_orig[:modify_idx], logits_new[:modify_idx])

    # Token at modify_idx and subsequent tokens should be CHANGED
    # Note: logits at modify_idx depends on input at modify_idx, so it should change.
    # We check that the max difference is significant.
    diff = (logits_orig[modify_idx:] - logits_new[modify_idx:]).abs().max()
    assert diff > 1e-3, f"Logits after modified token did not change (diff={diff})"
