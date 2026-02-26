import torch
import pytest

from optimus_dl.modules.criterion.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)
from optimus_dl.modules.distributed.fake import FakeCollective
from optimus_dl.modules.model.llama2 import (
    Llama,
    LlamaConfig,
)


def test_criterion_flat_batch_inconsistency():
    config = LlamaConfig(vocab_size=100, n_layer=1, n_head=1, n_embd=32)
    model = Llama(config)

    criterion_cfg = CrossEntropyCriterionConfig()
    criterion = CrossEntropyCriterion(criterion_cfg, FakeCollective(0, 1))

    # Flat batch: 2 documents [5 tokens, 5 tokens] -> total 10
    input_ids = torch.randint(0, 100, (1, 10))
    document_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).unsqueeze(0)
    position_ids = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).unsqueeze(0)

    batch = {
        "input_ids": input_ids,
        "document_ids": document_ids,
        "position_ids": position_ids,
    }

    # This is expected to fail currently because:
    # 1. input_ids is sliced to 9, but document_ids/position_ids remain 10
    # 2. Assert in attention will catch the mismatch
    try:
        loss, _ = criterion(model, batch)
    except AssertionError as e:
        print(f"Caught expected assertion error: {e}")
        assert "document_ids shape" in str(e) or "position_ids shape" in str(e)
        return

    pytest.fail("Criterion should have failed with shape mismatch assertion")


if __name__ == "__main__":
    test_criterion_flat_batch_inconsistency()
