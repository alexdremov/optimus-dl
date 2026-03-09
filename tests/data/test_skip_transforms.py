from typing import Any

import pytest
from torchdata.nodes.base_node import BaseNode

from optimus_dl.modules.data.transforms.skip import (
    SkipInterleavedTransform,
    SkipInterleavedTransformConfig,
    SkipInterleavedTransformNode,
    SkipRandomTransform,
    SkipRandomTransformConfig,
    SkipRandomTransformNode,
)


class MockSourceNode(BaseNode):
    """A simple mock node to feed data for testing."""

    def __init__(self, data: list[Any]):
        super().__init__()
        self.data = data
        self.idx = 0

    def next(self) -> Any:
        if self.idx >= len(self.data):
            raise StopIteration
        item = self.data[self.idx]
        self.idx += 1
        return item

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        if initial_state:
            self.idx = initial_state.get("idx", 0)
        else:
            self.idx = 0

    def state_dict(self) -> dict[str, Any]:
        return {"idx": self.idx}


class TestSkipInterleavedTransform:
    def test_skipping_logic(self):
        data = list(range(10))
        source = MockSourceNode(data)
        # skip_count=1 means: 0, (skip 1), 2, (skip 3), 4, ... -> 0, 2, 4, 6, 8
        cfg = SkipInterleavedTransformConfig(skip_count=1)
        node = SkipInterleavedTransformNode(source, cfg)

        results = []
        try:
            while True:
                results.append(next(node))
        except StopIteration:
            pass

        assert results == [0, 2, 4, 6, 8]

    def test_skip_count_two(self):
        data = list(range(10))
        source = MockSourceNode(data)
        # skip_count=2 means: 0, (skip 1, 2), 3, (skip 4, 5), 6, (skip 7, 8), 9
        cfg = SkipInterleavedTransformConfig(skip_count=2)
        node = SkipInterleavedTransformNode(source, cfg)

        results = []
        try:
            while True:
                results.append(next(node))
        except StopIteration:
            pass

        assert results == [0, 3, 6, 9]

    def test_state_management(self):
        data = list(range(10))
        source = MockSourceNode(data)
        cfg = SkipInterleavedTransformConfig(skip_count=1)
        node = SkipInterleavedTransformNode(source, cfg)

        assert next(node) == 0  # counter=1
        assert next(node) == 2  # counter=3

        state = node.get_state()
        assert state["counter"] == 3
        assert state["source_state"]["idx"] == 3

        # New node with same data but restored state
        source2 = MockSourceNode(data)
        node2 = SkipInterleavedTransformNode(source2, cfg)
        node2.reset(state)

        assert node2.counter == 3
        assert next(node2) == 4
        assert next(node2) == 6

    def test_transform_builder(self):
        source = MockSourceNode(list(range(10)))
        cfg = SkipInterleavedTransformConfig(skip_count=1)
        transform = SkipInterleavedTransform(cfg)
        node = transform.build(source)
        assert isinstance(node, SkipInterleavedTransformNode)
        assert node.cfg == cfg
        assert node.node == source


class TestSkipRandomTransform:
    def test_deterministic_with_seed(self):
        data = list(range(100))
        cfg = SkipRandomTransformConfig(probability=0.5)

        source1 = MockSourceNode(data)
        node1 = SkipRandomTransformNode(source1, cfg, seed=42)
        res1 = []
        try:
            while True:
                res1.append(next(node1))
        except StopIteration:
            pass

        source2 = MockSourceNode(data)
        node2 = SkipRandomTransformNode(source2, cfg, seed=42)
        res2 = []
        try:
            while True:
                res2.append(next(node2))
        except StopIteration:
            pass

        assert res1 == res2
        assert len(res1) < 100  # Probabilistically true

    def test_state_management(self):
        data = list(range(100))
        cfg = SkipRandomTransformConfig(probability=0.5)

        source = MockSourceNode(data)
        node = SkipRandomTransformNode(source, cfg, seed=42)

        [next(node) for _ in range(10)]
        state = node.get_state()
        res_part2_orig = []
        try:
            while True:
                res_part2_orig.append(next(node))
        except StopIteration:
            pass

        # Restore state
        source2 = MockSourceNode(data)
        node2 = SkipRandomTransformNode(source2, cfg, seed=42)
        node2.reset(state)

        res_part2_restored = []
        try:
            while True:
                res_part2_restored.append(next(node2))
        except StopIteration:
            pass

        assert res_part2_orig == res_part2_restored

    def test_zero_probability(self):
        data = list(range(10))
        source = MockSourceNode(data)
        cfg = SkipRandomTransformConfig(probability=0.0)
        node = SkipRandomTransformNode(source, cfg, seed=42)

        results = []
        try:
            while True:
                results.append(next(node))
        except StopIteration:
            pass

        assert results == data

    def test_one_probability(self):
        data = list(range(10))
        source = MockSourceNode(data)
        cfg = SkipRandomTransformConfig(probability=1.0)
        node = SkipRandomTransformNode(source, cfg, seed=42)

        with pytest.raises(StopIteration):
            next(node)

    def test_transform_builder(self):
        source = MockSourceNode(list(range(10)))
        cfg = SkipRandomTransformConfig(probability=0.5)
        transform = SkipRandomTransform(cfg, seed=42)
        node = transform.build(source)
        assert isinstance(node, SkipRandomTransformNode)
        assert node.cfg == cfg
        assert node.node == source
        assert node.seed == 42
