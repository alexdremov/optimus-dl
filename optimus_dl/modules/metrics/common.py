"""Common metric implementations and logging utilities.

This module provides standard metric types (averages, sums, frequencies, etc.)
and convenience functions for logging metrics during training. All metrics
support distributed aggregation and checkpointing.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .base import BaseMetric, log_metric


def safe_round(number: float | int | Any, ndigits: int | None) -> float | int:
    """Safely round a number, handling various numeric types.

    This function handles rounding for Python numbers, PyTorch tensors, and
    NumPy arrays. It recursively handles nested types (e.g., single-element
    tensors) until it reaches a roundable Python number.

    Args:
        number: The number to round. Can be a Python number, PyTorch tensor,
            or NumPy array.
        ndigits: Number of decimal places to round to. If None, returns the
            number unchanged.

    Returns:
        Rounded number as float or int (depending on whether rounding occurred).

    Example:
        >>> safe_round(3.14159, 2)  # 3.14
        >>> safe_round(torch.tensor(3.14159), 2)  # 3.14
        >>> safe_round(3.14159, None)  # 3.14159 (no rounding)
    """
    if ndigits is None:
        return number
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


class AverageMetric(BaseMetric):
    """Metric that computes a weighted average of logged values.

    This metric accumulates weighted values and computes the average when
    `compute()` is called. Useful for metrics like loss, accuracy, etc.
    that should be averaged over batches.

    Attributes:
        round: Number of decimal places to round the result to (None = no rounding).
        sum: Accumulated sum of (value * weight).
        count: Accumulated sum of weights.

    Example:
        >>> metric = AverageMetric(round=4)
        >>> metric.log(value=0.5, weight=32)  # Batch size 32
        >>> metric.log(value=0.6, weight=32)
        >>> metric.compute()  # (0.5*32 + 0.6*32) / (32+32) = 0.55
    """

    def __init__(self, round: int | None = None):
        """Initialize the average metric.

        Args:
            round: Number of decimal places to round results to. If None,
                results are not rounded.
        """
        self.round = round
        self.sum = 0
        self.count = 0

    def compute(self) -> float | int:
        """Compute the weighted average.

        Returns:
            Weighted average of all logged values, optionally rounded.

        Raises:
            ZeroDivisionError: If no values have been logged (count == 0).
        """
        return safe_round(self.sum / self.count, self.round)

    def log(self, value: float | int, weight: float | int) -> None:
        """Log a value with an associated weight.

        Args:
            value: The value to add to the average.
            weight: The weight for this value (typically batch size).
        """
        self.sum += value * weight
        self.count += weight

    def merge(self, other_state: dict[str, Any]) -> None:
        """Merge state from another metric instance (for distributed aggregation).

        Args:
            other_state: State dictionary from another AverageMetric instance.
        """
        self.sum += other_state["sum"]
        self.count += other_state["count"]


class SummedMetric(BaseMetric):
    """Metric that sums logged values.

    This metric simply accumulates values without averaging. Useful for
    metrics like total tokens processed, total examples seen, etc.

    Attributes:
        round: Number of decimal places to round results to (None = no rounding).
        sum: Accumulated sum of all logged values.

    Example:
        >>> metric = SummedMetric()
        >>> metric.log(100)
        >>> metric.log(200)
        >>> metric.compute()  # 300
    """

    def __init__(self, round: int | None = None):
        """Initialize the summed metric.

        Args:
            round: Number of decimal places to round results to. If None,
                results are not rounded.
        """
        self.round = round
        self.sum = 0

    def compute(self) -> float | int:
        """Compute the sum.

        Returns:
            Sum of all logged values, optionally rounded.
        """
        return self.sum

    def log(self, value: float | int) -> None:
        """Log a value to add to the sum.

        Args:
            value: The value to add to the sum.
        """
        self.sum += value

    def merge(self, other_state: dict[str, Any]) -> None:
        """Merge state from another metric instance (for distributed aggregation).

        Args:
            other_state: State dictionary from another SummedMetric instance.
        """
        self.sum += other_state["sum"]


class FrequencyMetric(BaseMetric):
    def __init__(self, round: int | None = None):
        self.round = round
        self.start = None
        self.elapsed = 0
        self.counter = 0

    def log(self):
        if self.start is None:
            self.start = time.perf_counter_ns()
            return
        self.counter += 1
        self.elapsed += time.perf_counter_ns() - self.start
        self.start = time.perf_counter_ns()

    def compute(self) -> float | int | dict[str, float | int]:
        if self.counter == 0:
            return 0
        return safe_round(self.elapsed / self.counter / 1e6, self.round)

    def merge(self, other_state):
        self.elapsed += other_state["elapsed"]
        self.counter += other_state["counter"]

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.start = None


class StopwatchMeter(BaseMetric):
    def __init__(self, round: int | None = None):
        self.round = round
        self._start: float | None = None
        self.elapsed = 0
        self.counter = 0

    def log(self, mode):
        if mode == "start":
            self.start()
        elif mode == "end":
            self.end()
        else:
            raise AssertionError("Unknown mode")

    def start(self):
        self._start = time.perf_counter_ns()

    def end(self):
        assert self._start is not None, "Was never started"
        self.elapsed += time.perf_counter_ns() - self._start
        self.counter += 1
        self._start = None

    def compute(self) -> float | int | dict[str, float | int]:
        if self.counter == 0:
            return 0
        return safe_round(self.elapsed / self.counter / 1e6, self.round)

    def merge(self, other_state):
        self.elapsed += other_state["elapsed"]
        self.counter += other_state["counter"]

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._start = None


class AveragedExponentMeter(BaseMetric):
    def __init__(self, round: int | None = None):
        self._internal = AverageMetric()
        self.round = round

    def log(self, value, weight):
        self._internal.log(value, weight)

    def compute(self):
        return safe_round(np.exp(self._internal.compute()), self.round)

    def merge(self, other_state):
        self._internal.merge(other_state["internal"])

    def load_state_dict(self, state_dict):
        self._internal.load_state_dict(state_dict["internal"])
        self.round = state_dict["round"]

    def state_dict(self):
        return {
            "internal": self._internal.state_dict(),
            "round": self.round,
        }


DelayedValue = Any | Callable[[], Any]


def log_averaged(
    name: str,
    value: DelayedValue,
    weight: DelayedValue = 1.0,
    round: int | None = None,
    reset: bool = True,
    priority: int = 100,
) -> None:
    """Log a value to an averaged metric.

    This is a convenience function for logging values to an AverageMetric.
    The value and weight can be callables (lambdas) that are only evaluated
    when the metric is actually logged (lazy evaluation).

    Args:
        name: Name of the metric (e.g., "train/loss").
        value: Value to log. Can be a number or a callable that returns a number.
        weight: Weight for this value (typically batch size). Can be a number
            or a callable. Defaults to 1.0.
        round: Number of decimal places to round the result to.
        reset: If True, the metric is reset after logging (for per-iteration metrics).
            If False, the metric accumulates across iterations.
        priority: Priority for metric ordering when logging. Higher priority
            metrics appear first.

    Example:
        >>> # Log a simple value
        >>> log_averaged("train/loss", 0.5, weight=32)
        >>>
        >>> # Log with lazy evaluation (only computed if metric is logged)
        >>> log_averaged("train/loss", lambda: compute_loss(), weight=lambda: batch_size)
        >>>
        >>> # Log with rounding
        >>> log_averaged("train/accuracy", 0.95, round=4)
    """
    log_metric(
        name=name,
        metric_factory=lambda: AverageMetric(round=round),
        reset=reset,
        priority=priority,
        value=value,
        weight=weight,
    )


def log_averaged_exponent(
    name: str,
    value: DelayedValue,
    weight: DelayedValue = 1.0,
    round: int | None = None,
    reset: bool = True,
    priority: int = 100,
):
    log_metric(
        name=name,
        metric_factory=lambda: AveragedExponentMeter(round=round),
        reset=reset,
        priority=priority,
        value=value,
        weight=weight,
    )


def log_summed(
    name: str,
    value: DelayedValue,
    round: int | None = None,
    reset: bool = True,
    priority: int = 100,
) -> None:
    """Log a value to a summed metric.

    This is a convenience function for logging values to a SummedMetric.
    The value can be a callable (lambda) that is only evaluated when the
    metric is actually logged (lazy evaluation).

    Args:
        name: Name of the metric (e.g., "train/tokens_processed").
        value: Value to add to the sum. Can be a number or a callable that
            returns a number.
        round: Number of decimal places to round the result to.
        reset: If True, the metric is reset after logging. If False, the
            metric accumulates across iterations.
        priority: Priority for metric ordering when logging.

    Example:
        >>> # Log total tokens processed
        >>> log_summed("train/tokens", batch_size * seq_len)
        >>>
        >>> # Log with lazy evaluation
        >>> log_summed("train/tokens", lambda: get_token_count())
    """
    log_metric(
        name=name,
        metric_factory=lambda: SummedMetric(round=round),
        reset=reset,
        priority=priority,
        value=value,
    )


def log_event_start(
    name: str,
    round: int | None = None,
    reset: bool = True,
    priority: int = 100,
) -> None:
    """Start timing an event.

    This function starts a stopwatch for measuring the duration of an event.
    Call `log_event_end()` with the same name to stop timing and record the
    duration. The duration is automatically averaged across multiple occurrences.

    Args:
        name: Name of the event to time (e.g., "perf/forward_pass").
        round: Number of decimal places to round the duration to (in milliseconds).
        reset: If True, the metric is reset after logging.
        priority: Priority for metric ordering when logging.

    Example:
        >>> log_event_start("perf/forward_pass")
        >>> # ... do work ...
        >>> log_event_end("perf/forward_pass")
        >>> # Metric will show average duration in milliseconds
    """
    log_metric(
        name=name,
        metric_factory=lambda: StopwatchMeter(round=round),
        reset=reset,
        priority=priority,
        mode="start",
        force_log=True,  # Always log event occurrences
    )


def log_event_end(
    name: str,
    round: int | None = None,
    reset: bool = True,
    priority: int = 100,
) -> None:
    """End timing an event.

    This function stops a stopwatch that was started with `log_event_start()`.
    The duration between start and end is recorded and averaged across
    multiple occurrences.

    Args:
        name: Name of the event (must match the name used in `log_event_start()`).
        round: Number of decimal places to round the duration to (in milliseconds).
        reset: If True, the metric is reset after logging.
        priority: Priority for metric ordering when logging.

    Raises:
        AssertionError: If `log_event_start()` was not called for this event name.

    Example:
        >>> log_event_start("perf/backward_pass")
        >>> # ... do backward pass ...
        >>> log_event_end("perf/backward_pass")
    """
    log_metric(
        name=name,
        metric_factory=lambda: StopwatchMeter(round=round),
        reset=reset,
        priority=priority,
        mode="end",
        force_log=True,  # Always log event occurrences
    )


def log_event_occurence(
    name: str,
    round: int | None = None,
    reset: bool = True,
    priority: int = 100,
):
    log_metric(
        name=name,
        metric_factory=lambda: FrequencyMetric(round=round),
        reset=reset,
        priority=priority,
        force_log=True,  # Always log event occurrences
    )


class CachedLambda:
    """Wrapper that caches the result of a callable function.

    This is useful for expensive computations that are used multiple times
    in metric logging. The function is only called once, and subsequent calls
    return the cached result.

    Example:
        >>> # Expensive computation
        >>> def compute_expensive_metric():
        ...     return complex_calculation()
        >>>
        >>> cached = CachedLambda(compute_expensive_metric)
        >>> value1 = cached()  # Computes and caches
        >>> value2 = cached()  # Returns cached value
    """

    def __init__(self, func: Callable[[], Any]):
        """Initialize the cached lambda.

        Args:
            func: Callable function that takes no arguments and returns a value.
        """
        self._func = func
        self._cache = None
        self._cached = False

    def __call__(self) -> Any:
        """Call the function, caching the result.

        Returns:
            The result of the function call. On first call, the function is
            executed and the result is cached. On subsequent calls, the cached value
            is returned.
        """
        if not self._cached:
            self._cache = self._func()
            self._cached = True
        return self._cache


def cached_lambda(x: Callable[[], Any]) -> CachedLambda:
    """Create a cached lambda wrapper.

    Convenience function for creating a CachedLambda instance.

    Args:
        x: Callable function to cache.

    Returns:
        CachedLambda instance that caches the function's result.

    Example:
        >>> expensive = cached_lambda(lambda: expensive_computation())
        >>> result = expensive()  # Computes once
        >>> result = expensive()  # Uses cache
    """
    return CachedLambda(x)
