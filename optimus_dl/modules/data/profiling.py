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


class PipelineProfiler:
    def __init__(self, name: str, report_freq: int | None = None):
        self.name = name
        self.report_freq = report_freq
        self.stats: dict[str, ProfilingStats] = {}
        self.iteration = 0
        self._local = threading.local()
        self.root_nodes: list[Any] = []

    @property
    def _stack(self) -> list[float]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def record_call(self, stage_name: str, func):
        start = time.perf_counter()
        stack = self._stack
        stack.append(0.0)
        try:
            return func()
        finally:
            duration = time.perf_counter() - start
            children_time = stack.pop()
            self_time = max(0.0, duration - children_time)

            if stage_name not in self.stats:
                self.stats[stage_name] = ProfilingStats(stage_name)
            self.stats[stage_name].record(duration, self_time)

            if stack:
                stack[-1] += duration

    def step(self):
        self.iteration += 1
        if self.report_freq and self.iteration % self.report_freq == 0:
            self.print_report()

    def print_report(self):
        if not self.stats:
            logger.info(f"No profiling data collected for pipeline: {self.name}")
            return

        sorted_stats = sorted(
            self.stats.values(), key=lambda x: x.self_time, reverse=True
        )
        total_pipeline_time = sum(s.self_time for s in sorted_stats)
        if total_pipeline_time == 0:
            total_pipeline_time = 1e-9

        if sys.stdout.isatty():
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(title=f"Data Pipeline Profiling Report: {self.name}")
                table.add_column("Stage Name", style="cyan")
                table.add_column("Calls", justify="right")
                table.add_column("Self Time (ms)", justify="right")
                table.add_column("% Total", justify="right", style="magenta")
                table.add_column("Total Time (ms)", justify="right")

                for stat in sorted_stats:
                    self_ms = stat.self_time * 1000
                    total_ms = stat.total_time * 1000
                    percent = (stat.self_time / total_pipeline_time) * 100
                    table.add_row(
                        stat.name,
                        str(stat.calls),
                        f"{self_ms:.2f}",
                        f"{percent:.1f}%",
                        f"{total_ms:.2f}",
                    )
                console.print(table)
                return
            except ImportError:
                pass

        result = "=" * 85 + "\n"
        result += f"Data Pipeline Profiling Report: {self.name}\n"
        result += "=" * 85 + "\n"

        result += f"{'Stage Name':<30} | {'Calls':<8} | {'Self Time (ms)':<15} | {'% Total':<8} | {'Total Time (ms)':<15}\n"
        result += "-" * 85 + "\n"
        for stat in sorted_stats:
            self_ms = stat.self_time * 1000
            total_ms = stat.total_time * 1000
            percent = (stat.self_time / total_pipeline_time) * 100

            result += f"{stat.name:<30} | {stat.calls:<8} | {self_ms:<15.2f} | {percent:<8.1f} | {total_ms:<15.2f}\n"

        result += "=" * 85

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

    def next(self):
        # Required for BaseNode inheritance; this override wraps next() for profiling
        result = self._profiler.record_call(self._name, self._inner_node.next)
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
