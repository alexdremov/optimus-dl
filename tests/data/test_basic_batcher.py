from typing import Any

import numpy as np
import torch
import pytest
from torchdata.nodes.base_node import BaseNode

# Adjust these imports based on your actual project structure
from optimus_dl.modules.data.transforms.basic_batcher import (
    BasicBatcher,
    BasicBatcherConfig,
    BasicBatcherNode,
)


class MockSourceNode(BaseNode):
    """A simple mock node to feed data into the batcher for testing."""

    def __init__(self, data: list[dict[str, Any]]):
        super().__init__()
        self.data = data
        self.idx = 0

    def next(self) -> dict[str, Any]:
        if self.idx >= len(self.data):
            raise StopIteration
        item = self.data[self.idx]
        self.idx += 1
        return item

    def reset(self, initial_state: dict | None = None):
        self.idx = initial_state["idx"] if initial_state else 0

    def state_dict(self) -> dict[str, Any]:
        return {"idx": self.idx}


class TestBasicBatcher:
    """Test suite for the BasicBatcher and BasicBatcherNode."""

    def test_batching_python_lists(self):
        """Test batching and dynamic padding with standard Python lists."""
        data = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4]},  # Needs padding of 2
            {"input_ids": [5, 6]},  # Needs padding of 1
            {"input_ids": [7, 8, 9, 10]},  # Max len
        ]
        source = MockSourceNode(data)
        config = BasicBatcherConfig(batch_size=4, pad_token_id=0)
        batcher_node = BasicBatcherNode(source, config)

        batch = next(batcher_node)

        assert "input_ids" in batch
        assert "seq_lens" in batch

        # Output should be numpy arrays when inputs are lists
        assert isinstance(batch["input_ids"], np.ndarray)
        assert isinstance(batch["seq_lens"], np.ndarray)

        # Check padding
        expected_ids = np.array(
            [
                [1, 2, 3, 0],
                [4, 0, 0, 0],
                [5, 6, 0, 0],
                [7, 8, 9, 10],
            ]
        )
        np.testing.assert_array_equal(batch["input_ids"], expected_ids)

        # Check lengths
        expected_lens = np.array([3, 1, 2, 4])
        np.testing.assert_array_equal(batch["seq_lens"], expected_lens)

    def test_batching_numpy_arrays(self):
        """Test batching and dynamic padding with NumPy arrays."""
        data = [
            {"input_ids": np.array([1, 2])},
            {"input_ids": np.array([3, 4, 5])},
        ]
        source = MockSourceNode(data)
        config = BasicBatcherConfig(batch_size=2, pad_token_id=-1)  # Custom pad token
        batcher_node = BasicBatcherNode(source, config)

        batch = next(batcher_node)

        assert isinstance(batch["input_ids"], np.ndarray)

        expected_ids = np.array(
            [
                [1, 2, -1],
                [3, 4, 5],
            ]
        )
        np.testing.assert_array_equal(batch["input_ids"], expected_ids)
        np.testing.assert_array_equal(batch["seq_lens"], np.array([2, 3]))

    def test_batching_torch_tensors(self):
        """Test batching and dynamic padding natively with PyTorch tensors."""
        data = [
            {"input_ids": torch.tensor([10, 20, 30])},
            {"input_ids": torch.tensor([40])},
        ]
        source = MockSourceNode(data)
        config = BasicBatcherConfig(batch_size=2, pad_token_id=99)
        batcher_node = BasicBatcherNode(source, config)

        batch = next(batcher_node)

        # Output should be torch tensors
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["seq_lens"], torch.Tensor)

        expected_ids = torch.tensor([[10, 20, 30], [40, 99, 99]])
        expected_lens = torch.tensor([3, 1])

        torch.testing.assert_close(batch["input_ids"], expected_ids)
        torch.testing.assert_close(batch["seq_lens"], expected_lens)

    def test_partial_final_batch(self):
        """Test that the batcher yields a smaller final batch when the source is exhausted."""
        data = [
            {"input_ids": [1]},
            {"input_ids": [2]},
            {"input_ids": [3]},
        ]
        source = MockSourceNode(data)
        # Batch size is 2, so we expect one batch of 2, and one batch of 1
        config = BasicBatcherConfig(batch_size=2, pad_token_id=0)
        batcher_node = BasicBatcherNode(source, config)

        # First batch
        batch1 = next(batcher_node)
        assert batch1["input_ids"].shape == (2, 1)

        # Second (partial) batch
        batch2 = next(batcher_node)
        assert batch2["input_ids"].shape == (1, 1)
        np.testing.assert_array_equal(batch2["input_ids"], np.array([[3]]))

        # Exhausted
        with pytest.raises(StopIteration):
            next(batcher_node)

    def test_custom_field_name(self):
        """Test that the batcher works with custom dictionary field names."""
        data = [
            {"custom_tokens": [1, 2]},
            {"custom_tokens": [3]},
        ]
        source = MockSourceNode(data)
        config = BasicBatcherConfig(batch_size=2, pad_token_id=0, field="custom_tokens")
        batcher_node = BasicBatcherNode(source, config)

        batch = next(batcher_node)

        assert "custom_tokens" in batch
        assert "seq_lens" in batch
        np.testing.assert_array_equal(
            batch["custom_tokens"], np.array([[1, 2], [3, 0]])
        )

    def test_transform_build(self):
        """Test that the Transform wrapper correctly builds the Node."""
        data = [{"input_ids": [1]}]
        source = MockSourceNode(data)
        config = BasicBatcherConfig(batch_size=4)
        transform = BasicBatcher(config)

        node = transform.build(source)
        assert isinstance(node, BasicBatcherNode)
        assert node.cfg.batch_size == 4

    def test_state_checkpointing(self):
        """Test get_state and reset logic for checkpointing."""
        data = [{"input_ids": [1]}, {"input_ids": [2]}, {"input_ids": [3]}]
        source = MockSourceNode(data)
        config = BasicBatcherConfig(batch_size=1)
        batcher_node = BasicBatcherNode(source, config)

        # Consume first item
        next(batcher_node)

        # Save state
        state = batcher_node.get_state()
        assert state["source_state"]["idx"] == 1
        assert state["cfg"] == config

        # Create a new batcher and restore state
        new_source = MockSourceNode(data)
        new_batcher_node = BasicBatcherNode(new_source, config)
        new_batcher_node.reset(state)

        # Next item should be 2 (skipping 1)
        batch = next(new_batcher_node)
        np.testing.assert_array_equal(batch["input_ids"], np.array([[2]]))
