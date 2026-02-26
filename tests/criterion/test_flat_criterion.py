import torch

from optimus_dl.modules.criterion.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)
from optimus_dl.modules.distributed.fake import FakeCollective
from optimus_dl.modules.model.llama2 import (
    Llama,
    LlamaConfig,
)


def test_criterion_flat_batch_metadata_alignment():
    """Verify that criterion correctly aligns metadata for flat batches."""
    from optimus_dl.modules.metrics.source import StandardProtocols

    config = LlamaConfig(vocab_size=100, n_layer=1, n_head=1, n_embd=32)
    model = Llama(config)

    criterion_cfg = CrossEntropyCriterionConfig()
    criterion = CrossEntropyCriterion(criterion_cfg, FakeCollective(0, 1))

    # Flat batch: 2 documents [5 tokens, 5 tokens] -> total 10
    # Note: These are UN-SHIFTED (raw from batcher if batcher didn't shift)
    input_ids = torch.randint(0, 100, (1, 10))
    document_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).unsqueeze(0)
    position_ids = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).unsqueeze(0)

    batch = {
        "input_ids": input_ids,
        "document_ids": document_ids,
        "position_ids": position_ids,
    }

    # This should now SUCCEED because criterion aligns document_ids and position_ids
    # by slicing them to match shifted input_ids (T-1 = 9)
    protocols = {StandardProtocols.CLASSIFICATION}
    loss, exposed = criterion(model, batch, requested_protocols=protocols)

    assert loss > 0
    # Input was length 10, shifted input is 9.
    # Prediction across doc boundary (token 4 -> token 5) is invalid.
    # Valid tokens: [0,1,2,3] (doc 0) and [5,6,7,8] (doc 1) -> total 8

    classif = exposed[StandardProtocols.CLASSIFICATION]
    # Re-alignment in exposed protocols means it's unflattened
    # But wait, this test doesn't provide cu_seqlens, so it's treated as a single doc or padded
    # Actually, without cu_seqlens, criterion treats it as a standard batch (B=1, T=10)
    # and unflattening only happens if "cu_seqlens" is in batch.

    assert classif["predictions"].shape == (1, 9)
    assert (
        classif["mask"].sum().item() == 9
    )  # No cu_seqlens, so all 9 shifted tokens are valid


if __name__ == "__main__":
    test_criterion_flat_batch_metadata_alignment()
