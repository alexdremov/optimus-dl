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
    # After shifting: 4, 9, 2, 7
    data = [
        {"input_ids": np.arange(5)},
        {"input_ids": np.arange(10)},
        {"input_ids": np.arange(3)},
        {"input_ids": np.arange(8)},
    ]
    source = MockNode(data)

    # Case 1: max_tokens = 12
    # Batch 1: 4 tokens (next is 9, 4+9=13 > 12)
    cfg = BasicBatcherConfig(max_tokens=12, pad_token_id=0, flatten=True)
    node = BasicBatcherNode(source, cfg)

    # Batch 1: should contain only first item (4 tokens after shift)
    batch1 = node.next()
    assert batch1["input_ids"].shape[1] == 4
    assert batch1["labels"].shape[1] == 4

    # Batch 2: should contain second item (9 tokens after shift)
    batch2 = node.next()
    assert batch2["input_ids"].shape[1] == 9

    # Batch 3: should contain 3 and 8 (shifted: 2 and 7 -> 2+7=9 <= 12)
    batch3 = node.next()
    assert batch3["input_ids"].shape[1] == 9
    assert len(batch3["cu_seqlens"]) == 3  # [0, 2, 9]


def test_basic_batcher_max_tokens_packing():
    # Mock data: 4, 4, 4, 4 -> shifted: 3, 3, 3, 3
    data = [{"input_ids": np.arange(4)} for _ in range(10)]
    source = MockNode(data)

    # max_tokens = 10
    # Logic: budget is checked against raw lengths.
    # Item 1: 4. Total: 4.
    # Item 2: 4. Total: 8.
    # Item 3: 4. Total: 12 > 10. Stop.
    # Result: 2 items. Total shifted tokens: 3 + 3 = 6.
    cfg = BasicBatcherConfig(max_tokens=10, pad_token_id=0, flatten=True)
    node = BasicBatcherNode(source, cfg)

    batch1 = node.next()
    assert batch1["input_ids"].shape[1] == 6

    batch2 = node.next()
    assert batch2["input_ids"].shape[1] == 6


def test_flat_batcher_max_tokens():
    # Mock data: 100 tokens
    data = [{"input_ids": np.arange(100)}]
    source = MockNode(data)

    # max_tokens = 32
    # In FlatTokensBatcher, we requested max_tokens=32
    # It consumes the document, shifts it (99 tokens), and adds to buffers.
    # Yields input length 32, labels length 32
    cfg = FlatTokensBatcherConfig(max_tokens=32, flatten=True)
    node = FlatTokensBatcherNode(source, cfg)

    batch1 = node.next()
    assert batch1["input_ids"].shape == (1, 32)
    assert batch1["labels"].shape == (1, 32)
    # Check shifting
    np.testing.assert_array_equal(batch1["input_ids"][0], np.arange(32))
    np.testing.assert_array_equal(batch1["labels"][0], np.arange(1, 33))

    batch2 = node.next()
    assert batch2["input_ids"].shape == (1, 32)


def test_flat_batcher_non_flatten():
    # Mock data: 100 tokens
    data = [{"input_ids": np.arange(100)}]
    source = MockNode(data)

    # batch_size=2, seq_len=10. Total inputs needed: 20 tokens.
    # Buffer will contain 99 shifted tokens from first item.
    cfg = FlatTokensBatcherConfig(batch_size=2, seq_len=10, flatten=False)
    node = FlatTokensBatcherNode(source, cfg)

    batch1 = node.next()
    assert batch1["input_ids"].shape == (2, 10)
    assert batch1["labels"].shape == (2, 10)

    # First row: 0..9 -> target 1..10
    np.testing.assert_array_equal(batch1["input_ids"][0], np.arange(10))
    np.testing.assert_array_equal(batch1["labels"][0], np.arange(1, 11))

    # Second row: 10..19 -> target 11..20
    np.testing.assert_array_equal(batch1["input_ids"][1], np.arange(10, 20))
    np.testing.assert_array_equal(batch1["labels"][1], np.arange(11, 21))


def test_basic_batcher_large_item():
    # Single item larger than max_tokens
    data = [{"input_ids": np.arange(20)}]  # Shifted length 19
    source = MockNode(data)

    cfg = BasicBatcherConfig(max_tokens=10, pad_token_id=0, flatten=True)
    node = BasicBatcherNode(source, cfg)

    # Should still yield it to avoid deadlock
    batch1 = node.next()
    assert batch1["input_ids"].shape[1] == 19
