import numpy as np
import torch
from torchdata.nodes.base_node import BaseNode

from optimus_dl.modules.criterion.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)
from optimus_dl.modules.data.transforms.basic_batcher import (
    BasicBatcherConfig,
    BasicBatcherNode,
)
from optimus_dl.modules.distributed.fake import FakeCollective
from optimus_dl.modules.model.llama2 import (
    Llama,
    LlamaConfig,
)


class MockNode(BaseNode):
    def __init__(self, items):
        super().__init__()
        self.items = items
        self.idx = 0

    def next(self):
        if self.idx >= len(self.items):
            raise StopIteration
        item = self.items[self.idx]
        self.idx += 1
        return item

    def reset(self, initial_state=None):
        super().reset(initial_state)
        self.idx = 0


def test_loss_consistency_padded_vs_flat(device):
    """
    Verify that loss computed on a padded batch (internal shifting)
    matches loss computed on a flat batch (batcher-side shifting).
    """
    config = LlamaConfig(vocab_size=100, n_layer=2, n_head=4, n_embd=64)
    model = Llama(config).to(device).eval()

    pad_token_id = 0
    criterion_cfg = CrossEntropyCriterionConfig(padding_token_id=pad_token_id)
    criterion = CrossEntropyCriterion(criterion_cfg, FakeCollective(0, 1))

    # 1. Documents
    doc_lengths = [10, 25, 15]
    docs = [
        torch.randint(1, 100, (length,)).to(device) for length in doc_lengths
    ]  # tokens 1-99

    # 2. Padded Batch
    max_len = max(doc_lengths)
    input_ids_padded = torch.full(
        (len(docs), max_len), pad_token_id, dtype=torch.long, device=device
    )
    for i, doc in enumerate(docs):
        input_ids_padded[i, : len(doc)] = doc

    # In the padded case, we pass input_ids. Criterion will shift it.
    # Targets will be inputs[:, 1:].
    # Padding tokens (0) in inputs will become padding tokens in targets.
    # ignore_index=0 will mask them.
    seq_lens = torch.tensor(doc_lengths).to(device)
    padded_batch = {"input_ids": input_ids_padded, "seq_lens": seq_lens}
    with torch.no_grad():
        loss_padded, _ = criterion(model, padded_batch)

    # 3. Flat Batch (per-document shifting)
    # This simulates what BasicBatcher(flatten=True) now does
    input_segments = [d[:-1] for d in docs]
    label_segments = [d[1:] for d in docs]

    input_ids_flat = torch.cat(input_segments).unsqueeze(0)
    labels_flat = torch.cat(label_segments).unsqueeze(0)

    # Metadata for flat
    doc_ids_flat = (
        torch.cat([torch.full((len(s),), i) for i, s in enumerate(input_segments)])
        .unsqueeze(0)
        .to(device)
    )
    pos_ids_flat = (
        torch.cat([torch.arange(len(s)) for s in input_segments])
        .unsqueeze(0)
        .to(device)
    )

    # cu_seqlens for shifted documents
    shifted_lengths = [len(s) for s in input_segments]
    cu_seqlens = (
        torch.cumsum(torch.tensor([0] + shifted_lengths), dim=0)
        .to(torch.int32)
        .to(device)
    )
    max_seqlen = max(shifted_lengths)

    flat_batch = {
        "input_ids": input_ids_flat.to(device),
        "labels": labels_flat.to(device),
        "document_ids": doc_ids_flat,
        "position_ids": pos_ids_flat,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
    }

    with torch.no_grad():
        loss_flat, _ = criterion(model, flat_batch)

    print(f"Loss Padded: {loss_padded.item():.6f}")
    print(f"Loss Flat:   {loss_flat.item():.6f}")

    # They should be EXACTLY the same
    torch.testing.assert_close(loss_padded, loss_flat)


def test_batcher_shifting_logic():
    """Verify that BasicBatcher(flatten=True) produces correctly shifted inputs and labels."""
    data = [
        {"input_ids": np.array([1, 2, 3, 4, 5])},
        {"input_ids": np.array([10, 20, 30])},
    ]
    source = MockNode(data)
    cfg = BasicBatcherConfig(batch_size=2, pad_token_id=0, flatten=True)
    node = BasicBatcherNode(source, cfg)

    batch = node.next()

    # Doc 1: [1,2,3,4,5] -> in: [1,2,3,4], lab: [2,3,4,5]
    # Doc 2: [10,20,30] -> in: [10,20], lab: [20,30]
    # Concatenated:
    # in: [1,2,3,4,10,20]
    # lab: [2,3,4,5,20,30]

    expected_in = np.array([[1, 2, 3, 4, 10, 20]])
    expected_lab = np.array([[2, 3, 4, 5, 20, 30]])

    np.testing.assert_array_equal(batch["input_ids"], expected_in)
    np.testing.assert_array_equal(batch["labels"], expected_lab)

    # Position IDs should reset: [0,1,2,3, 0,1]
    expected_pos = np.array([[0, 1, 2, 3, 0, 1]])
    np.testing.assert_array_equal(batch["position_ids"], expected_pos)

    # Document IDs: [0,0,0,0, 1,1]
    expected_docs = np.array([[0, 0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(batch["document_ids"], expected_docs)

    # cu_seqlens: [0, 4, 6]
    expected_cu = np.array([0, 4, 6], dtype=np.int32)
    np.testing.assert_array_equal(batch["cu_seqlens"], expected_cu)


def test_exposed_protocols_consistency(device):
    """
    Verify that exposed protocols (LOGITS, CLASSIFICATION) return
    the same unflattened data for both padded and flat inputs.
    """
    from optimus_dl.modules.metrics.source import StandardProtocols

    config = LlamaConfig(vocab_size=100, n_layer=1, n_head=2, n_embd=32)
    model = Llama(config).to(device).eval()

    pad_token_id = 0
    criterion = CrossEntropyCriterion(
        CrossEntropyCriterionConfig(padding_token_id=pad_token_id), FakeCollective(0, 1)
    )

    # 1. Documents
    doc_lengths = [8, 15]
    docs = [torch.randint(1, 100, (length,)).to(device) for length in doc_lengths]

    # 2. Padded Batch
    max_len = max(doc_lengths)
    input_ids_padded = torch.full(
        (len(docs), max_len), pad_token_id, dtype=torch.long, device=device
    )
    for i, doc in enumerate(docs):
        input_ids_padded[i, : len(doc)] = doc

    seq_lens = torch.tensor(doc_lengths).to(device)
    padded_batch = {"input_ids": input_ids_padded, "seq_lens": seq_lens}

    protocols = {StandardProtocols.LOGITS, StandardProtocols.CLASSIFICATION}
    with torch.no_grad():
        _, exposed_padded = criterion(
            model, padded_batch, requested_protocols=protocols
        )

    # 3. Flat Batch
    input_segments = [d[:-1] for d in docs]
    label_segments = [d[1:] for d in docs]

    input_ids_flat = torch.cat(input_segments).unsqueeze(0)
    labels_flat = torch.cat(label_segments).unsqueeze(0)

    # Metadata for flat
    doc_ids_flat = (
        torch.cat([torch.full((len(s),), i) for i, s in enumerate(input_segments)])
        .unsqueeze(0)
        .to(device)
    )
    pos_ids_flat = (
        torch.cat([torch.arange(len(s)) for s in input_segments])
        .unsqueeze(0)
        .to(device)
    )

    shifted_lengths = [len(s) for s in input_segments]
    cu_seqlens = (
        torch.cumsum(torch.tensor([0] + shifted_lengths), dim=0)
        .to(torch.int32)
        .to(device)
    )
    max_seqlen = max(shifted_lengths)

    flat_batch = {
        "input_ids": input_ids_flat.to(device),
        "labels": labels_flat.to(device),
        "document_ids": doc_ids_flat,
        "position_ids": pos_ids_flat,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
    }

    with torch.no_grad():
        _, exposed_flat = criterion(model, flat_batch, requested_protocols=protocols)

    # 4. Comparison

    # Logits
    logits_p = exposed_padded[StandardProtocols.LOGITS]
    logits_f = exposed_flat[StandardProtocols.LOGITS]
    # Padded shape: (B, T-1, V)
    # Flat shape reconstructed: (B, max(T_shifted), V)
    # Note: padded batch shifting happens in criterion, so T-1 = max_len - 1
    # Flat batch shifting happened before, so T_flat = sum(T_i - 1)

    # Wait, the padded case in criterion produces (B, T-1, V)
    # The flat case reconstructed produces (B, max_shifted_len, V)
    # We should compare only valid tokens.

    for i in range(len(docs)):
        l_shifted = len(docs[i]) - 1
        torch.testing.assert_close(logits_p[i, :l_shifted], logits_f[i, :l_shifted])

    # Classification
    class_p = exposed_padded[StandardProtocols.CLASSIFICATION]
    class_f = exposed_flat[StandardProtocols.CLASSIFICATION]

    for i in range(len(docs)):
        l_shifted = len(docs[i]) - 1
        # Predictions
        torch.testing.assert_close(
            class_p["predictions"][i, :l_shifted], class_f["predictions"][i, :l_shifted]
        )
        # Targets
        torch.testing.assert_close(
            class_p["targets"][i, :l_shifted], class_f["targets"][i, :l_shifted]
        )
        # Mask
        torch.testing.assert_close(
            class_p["mask"][i, :l_shifted], class_f["mask"][i, :l_shifted]
        )


if __name__ == "__main__":
    # For manual debugging
    test_batcher_shifting_logic()
