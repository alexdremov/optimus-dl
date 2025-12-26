"""Performance profiling utilities.

This module provides functions for measuring execution time of code blocks,
iterators, and function calls. Supports both CPU timing (using perf_counter)
and GPU timing (using CUDA events for accurate GPU kernel timing).
"""

import time
from collections.abc import Callable, Iterator
from typing import TypeVar

import torch

T = TypeVar("T")


def measured_iter(itr: Iterator[T]) -> Iterator[tuple[float, T]]:
    """Measure time between iterations of an iterator.

    Yields tuples of (elapsed_time_ms, element) for each element in the iterator.
    The elapsed time is measured from the start of one iteration to the start
    of the next, providing per-iteration timing.

    Args:
        itr: Iterator to measure.

    Yields:
        Tuples of (elapsed_time_ms, element) where elapsed_time_ms is the
        time in milliseconds since the previous iteration.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> for elapsed, item in measured_iter(iter(data)):
        ...     print(f"Item {item} took {elapsed:.2f}ms")
    """
    start = time.perf_counter_ns()
    for elem in itr:
        elapsed = (time.perf_counter_ns() - start) / 1e6  # Convert to milliseconds
        yield elapsed, elem
        start = time.perf_counter_ns()


def measured_next(itr: Iterator[T]) -> tuple[float, T]:
    """Measure time to get the next element from an iterator.

    This is useful for measuring data loading time, as it measures the time
    to fetch a single batch from a data iterator.

    Args:
        itr: Iterator to get next element from.

    Returns:
        Tuple of (elapsed_time_ms, element) where elapsed_time_ms is the
        time in milliseconds to get the next element.

    Example:
        >>> data_iter = iter(dataloader)
        >>> elapsed, batch = measured_next(data_iter)
        >>> print(f"Data loading took {elapsed:.2f}ms")
    """
    start = time.perf_counter_ns()
    elem = next(itr)
    elapsed = (time.perf_counter_ns() - start) / 1e6  # Convert to milliseconds
    return elapsed, elem


def measured_lambda(
    f: Callable[[], T],
    cuda_events: bool = False,
    enabled: bool = True,
) -> tuple[float, T]:
    """Measure execution time of a callable function.

    Supports both CPU timing (using perf_counter) and GPU timing (using CUDA
    events). CUDA events provide more accurate timing for GPU operations as
    they measure actual GPU kernel execution time rather than wall-clock time.

    Args:
        f: Callable function to measure (takes no arguments).
        cuda_events: If True, use CUDA events for timing (more accurate for
            GPU operations). If False, use CPU perf_counter. Requires CUDA
            to be available.
        enabled: If False, skip timing and return (0, result). Useful for
            disabling profiling in production code.

    Returns:
        Tuple of (elapsed_time_ms, result) where:
        - elapsed_time_ms: Execution time in milliseconds
        - result: Return value of the function call

    Example:
        >>> def forward_pass():
        ...     return model(input_ids)
        >>>
        >>> # CPU timing
        >>> elapsed, output = measured_lambda(forward_pass, cuda_events=False)
        >>>
        >>> # GPU timing (more accurate for GPU ops)
        >>> elapsed, output = measured_lambda(forward_pass, cuda_events=True)
    """
    if not enabled:
        return 0, f()
    if cuda_events:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        elem = f()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        return elapsed, elem
    else:
        start = time.perf_counter_ns()
        elem = f()
        elapsed = (time.perf_counter_ns() - start) / 1e6  # Convert to milliseconds
        return elapsed, elem
