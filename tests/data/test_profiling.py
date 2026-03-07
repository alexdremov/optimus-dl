import logging
import time

import torchdata.nodes

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.data import build_data_pipeline
from optimus_dl.modules.data.config import DataPipelineConfig
from optimus_dl.modules.data.datasets.base import BaseDataset
from optimus_dl.modules.data.profiling import (
    PipelineProfiler,
    PipelineTracer,
    ProfilingProxyNode,
    get_active_profiler,
)
from optimus_dl.modules.data.transforms.base import BaseTransform


class MockNode(torchdata.nodes.BaseNode):
    def __init__(self, delay=0.01, *args, **kwargs):
        super().__init__()
        self.delay = delay
        self.count = 0

    def next(self):
        if self.count >= 5:
            raise StopIteration
        time.sleep(self.delay)
        self.count += 1
        return self.count

    def reset(self, state=None):
        super().reset(state)
        self.count = 0


class MockTransform(BaseTransform):
    def __init__(self, delay=0.02):
        super().__init__()
        self.delay = delay

    def build(self, source):
        class TransformNode(torchdata.nodes.BaseNode):
            def __init__(self, inner, delay):
                super().__init__()
                self.inner = inner
                self.delay = delay

            def next(self):
                item = next(self.inner)
                time.sleep(self.delay)
                return item

            def reset(self, state=None):
                super().reset(state)
                self.inner.reset(state)

        return TransformNode(source, self.delay)


def test_tracer_state():
    assert get_active_profiler() is None
    with PipelineTracer() as profiler:
        assert get_active_profiler() is profiler
    assert get_active_profiler() is None


def test_profiling_proxy_timing():
    delay = 0.05
    node = MockNode(delay=delay)
    profiler = PipelineProfiler("test")
    proxy = ProfilingProxyNode(node, name="TestNode", profiler=profiler)

    # stats should be empty or zero
    assert "TestNode" not in profiler._analyze_bottlenecks()

    item = next(proxy)
    assert item == 1
    stats = profiler._analyze_bottlenecks()
    assert stats["TestNode"].calls == 1
    # Total time should be at least the delay
    assert stats["TestNode"].total_time >= delay
    # Self time should be approx total time since there are no children being profiled
    assert abs(stats["TestNode"].compute_time - stats["TestNode"].total_time) < 0.005


def test_nested_profiling_self_time():
    child_delay = 0.05
    parent_delay = 0.1
    profiler = PipelineProfiler("test")

    child_node = MockNode(delay=child_delay)
    child_proxy = ProfilingProxyNode(child_node, name="Child", profiler=profiler)

    class ParentNode(torchdata.nodes.BaseNode):
        def __init__(self, inner, delay):
            super().__init__()
            self.inner = inner
            self.delay = delay

        def next(self):
            item = next(self.inner)
            time.sleep(self.delay)
            return item

        def reset(self, state=None):
            super().reset(state)
            self.inner.reset(state)

    parent_proxy = ProfilingProxyNode(
        ParentNode(child_proxy, parent_delay), name="Parent", profiler=profiler
    )

    next(parent_proxy)

    stats = profiler._analyze_bottlenecks()
    # Child: total ~0.05, self ~0.05
    assert stats["Child"].total_time >= child_delay
    # Parent: total ~0.15 (0.05 child + 0.1 parent), self ~0.1
    assert stats["Parent"].total_time >= (child_delay + parent_delay)
    assert abs(stats["Parent"].compute_time - parent_delay) < 0.02


def test_base_transform_auto_wrapping():
    with PipelineTracer() as profiler:
        transform = MockTransform(delay=0.01)
        source = MockNode()
        # MockTransform.build is wrapped via __init_subclass__
        node = transform.build(source)

        assert isinstance(node, ProfilingProxyNode)
        assert "MockTransform" in node._name

        next(node)
        stats = profiler._analyze_bottlenecks()
        assert any("MockTransform" in name for name in stats)
        assert any(s.calls == 1 for n, s in stats.items() if "MockTransform" in n)


def test_build_data_pipeline_wrapping():
    # Register a mock dataset for this test
    from optimus_dl.modules.data.datasets import register_dataset

    @register_dataset("profile_mock_dataset", RegistryConfig)
    class ProfileMockDataset(BaseDataset):
        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg)
            self.delay = 0.01
            self.count = 0

        def next(self):
            if self.count >= 5:
                raise StopIteration
            time.sleep(self.delay)
            self.count += 1
            return self.count

        def reset(self, state=None):
            super().reset(state)
            self.count = 0

    cfg = DataPipelineConfig(
        source=RegistryConfig(_name="profile_mock_dataset"),
        transform=None,
        profile=False,
    )

    # Without profiling
    pipeline = build_data_pipeline(
        cfg, profile_name="test", rank=0, world_size=1, seed=42
    )
    assert not isinstance(pipeline.dataloader, ProfilingProxyNode)

    # With profiling
    cfg.profile = True
    cfg.report_freq = 10
    pipeline_p = build_data_pipeline(
        cfg, profile_name="test", rank=0, world_size=1, seed=42
    )
    assert isinstance(pipeline_p.dataloader, ProfilingProxyNode)
    assert "ProfileMockDataset" in pipeline_p.dataloader._name


def test_print_report(caplog):
    profiler = PipelineProfiler("test_report")
    node = ProfilingProxyNode(
        MockNode(delay=0.001), name="ReportNode", profiler=profiler
    )
    next(node)
    with caplog.at_level(logging.INFO):
        profiler.print_report()

    assert "Data Pipeline Profiling Report: test_report" in caplog.text
    assert "ReportNode" in caplog.text
    assert "1" in caplog.text  # Calls


def test_print_pipeline_tree(caplog):
    from optimus_dl.modules.data.datasets import register_dataset

    @register_dataset("tree_mock_dataset", RegistryConfig)
    class TreeMockDataset(BaseDataset):
        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg)

        def next(self):
            return 1

        def reset(self, state=None):
            super().reset(state)

    cfg = DataPipelineConfig(
        source=RegistryConfig(_name="tree_mock_dataset"),
        transform={
            "_name": "compose",
            "transforms": [{"_name": "prefetch", "prefetch_factor": 2}],
        },
        profile=True,
        report_freq=10,
    )

    pipeline = build_data_pipeline(
        cfg, profile_name="tree_mock_dataset", rank=0, world_size=1, seed=42
    )
    # The dataloader should be a proxy
    assert isinstance(pipeline.dataloader, ProfilingProxyNode)
    profiler = pipeline.dataloader._profiler
    with caplog.at_level(logging.INFO):
        profiler.print_pipeline_tree()

    assert "Data Pipeline Structure: tree_mock_dataset" in caplog.text
    assert "CompositeTransform" in caplog.text
    assert "Prefetcher" in caplog.text  # Internal node of PrefetchTransform
    assert "TreeMockDataset" in caplog.text


def test_report_freq(caplog):
    profiler = PipelineProfiler("freq_test", report_freq=2)
    node = ProfilingProxyNode(
        MockNode(delay=0.001), name="FreqNode", profiler=profiler, is_root=True
    )

    with caplog.at_level(logging.INFO):
        next(node)
        assert "Data Pipeline Profiling Report" not in caplog.text

        next(node)
        assert "Data Pipeline Profiling Report: freq_test" in caplog.text


def test_isolation():
    # Two pipelines in the same thread should not mix stats
    p1 = PipelineProfiler("p1")
    p2 = PipelineProfiler("p2")

    n1 = ProfilingProxyNode(MockNode(), name="Node", profiler=p1)
    n2 = ProfilingProxyNode(MockNode(), name="Node", profiler=p2)

    next(n1)
    next(n2)
    next(n2)

    stats1 = p1._analyze_bottlenecks()
    stats2 = p2._analyze_bottlenecks()

    assert stats1["Node"].calls == 1
    assert stats2["Node"].calls == 2
