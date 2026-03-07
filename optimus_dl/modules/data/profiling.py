import logging
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    Any,
)

import torchdata.nodes

logger = logging.getLogger(__name__)


class ProfilingStats:
    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0
        self.self_time = 0.0
        self.calls = 0

    def record(self, total_duration: float, self_duration: float):
        self.total_time += total_duration
        self.self_time += self_duration
        self.calls += 1


class StageAnalysis:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.compute_time = 0.0
        self.blocked_time = 0.0
        self.total_time = 0.0


def _get_upstream_proxies(node) -> list["ProfilingProxyNode"]:
    upstreams = []
    visited = set()

    def traverse(n):
        if id(n) in visited:
            return
        visited.add(id(n))

        if isinstance(n, ProfilingProxyNode):
            upstreams.append(n)
            return

        if hasattr(n, "node"):
            traverse(n.node)
        elif hasattr(n, "source"):
            traverse(n.source)
        elif hasattr(n, "inner_node"):
            traverse(n.inner_node)
        elif hasattr(n, "inner_dataset"):
            traverse(n.inner_dataset)
        elif hasattr(n, "inner"):
            traverse(n.inner)
        elif hasattr(n, "datasets") and isinstance(n.datasets, dict):
            for d in n.datasets.values():
                traverse(d)
        elif hasattr(n, "datasets") and isinstance(n.datasets, list):
            for d in n.datasets:
                traverse(d)

    if hasattr(node, "_inner_node"):
        traverse(node._inner_node)

    return upstreams


class PipelineProfiler:
    def __init__(self, name: str, report_freq: int | None = None):
        self.name = name
        self.report_freq = report_freq
        self.node_stats: dict[int, ProfilingStats] = {}
        self.iteration = 0
        self._local = threading.local()
        self.root_nodes: list[Any] = []
        self.all_proxies: list[ProfilingProxyNode] = []

    def add_proxy(self, proxy: "ProfilingProxyNode"):
        self.all_proxies.append(proxy)

    @property
    def _stack(self) -> list[float]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def record_call(self, proxy_node: "ProfilingProxyNode", func):
        start = time.perf_counter()
        stack = self._stack
        stack.append(0.0)
        try:
            return func()
        finally:
            duration = time.perf_counter() - start
            children_time = stack.pop()
            self_time = max(0.0, duration - children_time)

            node_id = id(proxy_node)
            if node_id not in self.node_stats:
                self.node_stats[node_id] = ProfilingStats(proxy_node._name)
            self.node_stats[node_id].record(duration, self_time)

            if stack:
                stack[-1] += duration

    def step(self):
        self.iteration += 1
        if self.report_freq and self.iteration % self.report_freq == 0:
            self.print_report()

    def _analyze_bottlenecks(self) -> dict[str, StageAnalysis]:
        analysis_by_stage: dict[str, StageAnalysis] = {}

        for proxy in self.all_proxies:
            node_stat = self.node_stats.get(id(proxy))
            if not node_stat or node_stat.calls == 0:
                continue

            upstreams = _get_upstream_proxies(proxy)
            upstream_total_time = sum(
                self.node_stats[id(u)].total_time
                for u in upstreams
                if id(u) in self.node_stats
            )

            children_time_recorded = node_stat.total_time - node_stat.self_time

            # Detect async boundary: upstream did work, but it wasn't recorded in this thread's stack
            is_async = False
            if upstreams and upstream_total_time > 0:
                if children_time_recorded < 0.1 * upstream_total_time:
                    is_async = True

            if is_async:
                compute_time = 0.0
                blocked_time = node_stat.self_time
            else:
                compute_time = node_stat.self_time
                blocked_time = 0.0

            name = proxy._name
            if name not in analysis_by_stage:
                analysis_by_stage[name] = StageAnalysis(name)

            analysis_by_stage[name].calls += node_stat.calls
            analysis_by_stage[name].total_time += node_stat.total_time
            analysis_by_stage[name].compute_time += compute_time
            analysis_by_stage[name].blocked_time += blocked_time

        return analysis_by_stage

    def print_report(self):
        if not self.node_stats:
            logger.info(f"No profiling data collected for pipeline: {self.name}")
            return

        analysis_by_stage = self._analyze_bottlenecks()
        sorted_stages = sorted(
            analysis_by_stage.values(), key=lambda x: x.compute_time, reverse=True
        )
        total_pipeline_compute = sum(s.compute_time for s in sorted_stages)
        if total_pipeline_compute == 0:
            total_pipeline_compute = 1e-9

        bottleneck = (
            sorted_stages[0]
            if sorted_stages and sorted_stages[0].compute_time > 0
            else None
        )

        if sys.stdout.isatty():
            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.table import Table

                console = Console()
                table = Table(title=f"Data Pipeline Profiling Report: {self.name}")
                table.add_column("Stage Name", style="cyan")
                table.add_column("Calls", justify="right")
                table.add_column("Compute (ms)", justify="right", style="green")
                table.add_column("Blocked (ms)", justify="right", style="red")
                table.add_column("% Compute", justify="right", style="magenta")
                table.add_column("Total Time (ms)", justify="right")

                for stat in sorted_stages:
                    compute_ms = stat.compute_time * 1000
                    blocked_ms = stat.blocked_time * 1000
                    total_ms = stat.total_time * 1000
                    percent = (stat.compute_time / total_pipeline_compute) * 100
                    table.add_row(
                        stat.name,
                        str(stat.calls),
                        f"{compute_ms:.2f}",
                        f"{blocked_ms:.2f}",
                        f"{percent:.1f}%",
                        f"{total_ms:.2f}",
                    )

                console.print(table)
                if bottleneck:
                    console.print(
                        Panel(
                            f"[bold red]🔥 Bottleneck Detected:[/bold red] [cyan]{bottleneck.name}[/cyan] is responsible for [bold]{(bottleneck.compute_time / total_pipeline_compute)*100:.1f}%[/bold] of total compute time.",
                            expand=False,
                        )
                    )
                return
            except ImportError:
                pass

        result = "=" * 85 + "\n"
        result += f"Data Pipeline Profiling Report: {self.name}\n"
        result += "=" * 85 + "\n"

        result += f"{'Stage Name':<30} | {'Calls':<8} | {'Compute (ms)':<15} | {'Blocked (ms)':<15} | {'% Compute':<10} | {'Total (ms)':<15}\n"
        result += "-" * 85 + "\n"
        for stat in sorted_stages:
            compute_ms = stat.compute_time * 1000
            blocked_ms = stat.blocked_time * 1000
            total_ms = stat.total_time * 1000
            percent = (stat.compute_time / total_pipeline_compute) * 100

            result += f"{stat.name:<30} | {stat.calls:<8} | {compute_ms:<15.2f} | {blocked_ms:<15.2f} | {percent:<10.1f} | {total_ms:<15.2f}\n"

        result += "=" * 85
        if bottleneck:
            result += f"\n🔥 Bottleneck Detected: {bottleneck.name} is responsible for {(bottleneck.compute_time / total_pipeline_compute)*100:.1f}% of total compute time."

        logger.info("\n" + result.strip())

    def print_pipeline_tree(self):
        if not self.root_nodes:
            logger.info(f"No pipeline structure captured for: {self.name}")
            return

        if sys.stdout.isatty():
            try:
                from rich.console import Console
                from rich.tree import Tree

                console = Console()
                root_tree = Tree(f"Data Pipeline Structure: {self.name}")

                def _build_rich_tree(node, rich_tree):
                    name = getattr(node, "_name", node.__class__.__name__)
                    branch = rich_tree.add(f"[cyan]{name}[/cyan]")

                    children = []
                    if isinstance(node, ProfilingProxyNode):
                        children.append(node._inner_node)
                    elif hasattr(node, "node"):
                        children.append(node.node)
                    elif hasattr(node, "source"):
                        children.append(node.source)
                    elif hasattr(node, "inner_node"):
                        children.append(node.inner_node)
                    elif hasattr(node, "datasets") and isinstance(node.datasets, dict):
                        children.extend(node.datasets.values())
                    elif hasattr(node, "inner_dataset"):
                        children.append(node.inner_dataset)

                    for child in children:
                        _build_rich_tree(child, branch)

                for node in self.root_nodes:
                    _build_rich_tree(node, root_tree)
                console.print(root_tree)
                return
            except ImportError:
                pass

        result = "=" * 85 + "\n"
        result += f"Data Pipeline Structure: {self.name}\n"
        result += "=" * 85 + "\n"

        def _print_node(node, indent="", is_last=True):
            nonlocal result
            marker = "└── " if is_last else "├── "

            name = getattr(node, "_name", node.__class__.__name__)
            result += f"{indent}{marker}{name}\n"

            new_indent = indent + ("    " if is_last else "│   ")

            # Find children/upstream nodes
            children = []
            if isinstance(node, ProfilingProxyNode):
                children.append(node._inner_node)
            elif hasattr(node, "node"):
                children.append(node.node)
            elif hasattr(node, "source"):
                children.append(node.source)
            elif hasattr(node, "inner_node"):
                children.append(node.inner_node)
            elif hasattr(node, "datasets") and isinstance(node.datasets, dict):
                children.extend(node.datasets.values())
            elif hasattr(node, "inner_dataset"):
                children.append(node.inner_dataset)

            for i, child in enumerate(children):
                _print_node(child, new_indent, is_last=(i == len(children) - 1))

        for node in self.root_nodes:
            _print_node(node)
        result += "=" * 85 + "\n"
        logger.info("\n" + result.strip())


# Context-local profiler for the build phase
_active_profiler: ContextVar[PipelineProfiler | None] = ContextVar(
    "active_profiler", default=None
)


def get_active_profiler() -> PipelineProfiler | None:
    return _active_profiler.get()


@contextmanager
def scope_profiler(profiler: PipelineProfiler | None):
    """Context manager to set the active profiler for the current context."""
    token = _active_profiler.set(profiler)
    try:
        yield
    finally:
        _active_profiler.reset(token)


class ProfilingProxyNode(torchdata.nodes.BaseNode):
    def __init__(
        self,
        inner_node: Any,
        name: str,
        profiler: PipelineProfiler,
        is_root: bool = False,
    ):
        super().__init__()
        self._inner_node = inner_node
        self._name = name
        self._profiler = profiler
        self._is_root = is_root

        if profiler:
            profiler.add_proxy(self)

    def next(self):
        # Required for BaseNode inheritance; this override wraps next() for profiling
        result = self._profiler.record_call(self, self._inner_node.next)
        if self._is_root:
            self._profiler.step()
        return result

    def reset(self, initial_state: Any = None):
        super().reset(initial_state)
        return self._inner_node.reset(initial_state)

    def get_state(self) -> dict[str, Any]:
        return self._inner_node.get_state()

    def __getattr__(self, attr):
        if attr in ["_inner_node", "_name", "_profiler", "_is_root"]:
            return super().__getattribute__(attr)
        return getattr(self._inner_node, attr)


class PipelineTracer:
    """Convenience context manager for enabling profiling during build phase."""

    def __init__(self, name: str = "manual", report_freq: int | None = None):
        self.profiler = PipelineProfiler(name, report_freq)
        self._scope = None

    def __enter__(self):
        self._scope = scope_profiler(self.profiler)
        self._scope.__enter__()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._scope:
            self._scope.__exit__(exc_type, exc_val, exc_tb)
