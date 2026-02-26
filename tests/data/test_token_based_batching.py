import numpy as np
from torchdata.nodes.base_node import BaseNode

from optimus_dl.modules.data.transforms.basic_batcher import (
    BasicBatcherConfig,
    BasicBatcherNode,
)
from optimus_dl.modules.data.transforms.flat_tokens_batcher import (
    FlatTokensBatcherConfig,
    FlatTokensBatcherNode,
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


def test_basic_batcher_max_tokens():
    # Mock data: lengths 5, 10, 3, 8
    data = [
        {"input_ids": np.arange(5)},
        {"input_ids": np.arange(10)},
        {"input_ids": np.arange(3)},
        {"input_ids": np.arange(8)},
    ]
    source = MockNode(data)

    # Case 1: max_tokens = 12
    # Batch 1: 5 + 10 > 12, so only 5. Wait, current implementation:
    # If 5 < 12, it takes 5. Next is 10. 5 + 10 = 15 > 12. So it stops at 5.
    # Actually, my implementation peeks and stops if it would exceed.
    # Let's verify.
    cfg = BasicBatcherConfig(max_tokens=12, pad_token_id=0, flatten=True)
    node = BasicBatcherNode(source, cfg)

    # Batch 1: should contain only first item (5 tokens)
    batch1 = node.next()
    assert batch1["input_ids"].shape[1] == 5

    # Batch 2: should contain second item (10 tokens)
    batch2 = node.next()
    assert batch2["input_ids"].shape[1] == 10

    # Batch 3: should contain 3 and 8 (3+8=11 <= 12)
    batch3 = node.next()
    assert batch3["input_ids"].shape[1] == 11
    assert len(batch3["cu_seqlens"]) == 3  # [0, 3, 11]


def test_basic_batcher_max_tokens_packing():
    # Mock data: 4, 4, 4, 4
    data = [{"input_ids": np.arange(4)} for _ in range(10)]
    source = MockNode(data)

    # max_tokens = 10
    # Batch 1: 4 + 4 = 8. (8 + 4 = 12 > 10). So 8 tokens.
    cfg = BasicBatcherConfig(max_tokens=10, pad_token_id=0, flatten=True)
    node = BasicBatcherNode(source, cfg)

    batch1 = node.next()
    assert batch1["input_ids"].shape[1] == 8

    batch2 = node.next()
    assert batch2["input_ids"].shape[1] == 8


def test_flat_batcher_max_tokens():
    # Mock data: 100 tokens
    data = [{"input_ids": np.arange(100)}]
    source = MockNode(data)

    # max_tokens = 32
    cfg = FlatTokensBatcherConfig(max_tokens=32, flatten=True)
    node = FlatTokensBatcherNode(source, cfg)

    batch1 = node.next()
    assert batch1["input_ids"].shape == (1, 32)

    batch2 = node.next()
    assert batch2["input_ids"].shape == (1, 32)

    batch3 = node.next()
    assert batch3["input_ids"].shape == (1, 32)


def test_basic_batcher_large_item():
    # Single item larger than max_tokens
    data = [{"input_ids": np.arange(20)}]
    source = MockNode(data)

    cfg = BasicBatcherConfig(max_tokens=10, pad_token_id=0, flatten=True)
    node = BasicBatcherNode(source, cfg)

    # Should still yield it to avoid deadlock
    batch1 = node.next()
    assert batch1["input_ids"].shape[1] == 20
