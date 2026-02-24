"""Common meter implementations and logging utilities.

This module provides standard meter types (averages, sums, frequencies, etc.)
and convenience functions for logging meter values during training. All meters
support distributed aggregation and checkpointing.
"""

import time
from collections.abc import Callable
from typing import (
    Any,
)

import numpy as np
import torch

from .base import BaseMeter  # Changed from BaseMetric
from .base import (
    log_metric,
)


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
        ```python
        safe_round(3.14159, 2)  # 3.14
        safe_round(torch.tensor(3.14159), 2)  # 3.14
        safe_round(3.14159, None)  # 3.14159 (no rounding)

        ```"""
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


class AverageMeter(BaseMeter):  # Changed from AverageMetric
    """Meter that computes a weighted average of logged values.

    This meter accumulates weighted values and computes the average when
    `compute()` is called. Useful for values like loss, accuracy, etc.
    that should be averaged over batches.

    Attributes:
        round: Number of decimal places to round the result to (None = no rounding).
        sum: Accumulated sum of (value * weight).
        count: Accumulated sum of weights.

    Example:
        ```python
        meter = AverageMeter(round=4)
        meter.log(value=0.5, weight=32)  # Batch size 32
        meter.log(value=0.6, weight=32)
        meter.compute()  # (0.5*32 + 0.6*32) / (32+32) = 0.55

        ```"""

    def __init__(self, round: int | None = None):
        """Initialize the average meter.

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
        if self.count == 0:
            return 0  # Or raise an error, depending on desired behavior
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
        """Merge state from another meter instance (for distributed aggregation).

        Args:
            other_state: State dictionary from another AverageMeter instance.
        """
        self.sum += other_state["sum"]
        self.count += other_state["count"]


class SummedMeter(BaseMeter):  # Changed from SummedMetric
    """Meter that sums logged values.

    This meter simply accumulates values without averaging. Useful for
    values like total tokens processed, total examples seen, etc.

    Attributes:
        round: Number of decimal places to round results to (None = no rounding).
        sum: Accumulated sum of all logged values.

    Example:
        ```python
        meter = SummedMeter()
        meter.log(100)
        meter.log(200)
        meter.compute()  # 300

        ```"""

    def __init__(self, round: int | None = None):
        """Initialize the summed meter.

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
        return safe_round(self.sum, self.round)

    def log(self, value: float | int) -> None:
        """Log a value to add to the sum.

        Args:
            value: The value to add to the sum.
        """
        self.sum += value

    def merge(self, other_state: dict[str, Any]) -> None:
        """Merge state from another meter instance (for distributed aggregation).

        Args:
            other_state: State dictionary from another SummedMeter instance.
        """
        self.sum += other_state["sum"]


class FrequencyMeter(BaseMeter):  # Changed from FrequencyMetric
    """Meter that computes the frequency (duration per call) of an event.

    Measures time between successive calls to `log()`.

    Attributes:
        round: Rounding precision for the result.
        start: Internal timestamp of the last call to `log()`.
        elapsed: Total elapsed time in nanoseconds.
        counter: Number of events recorded.
    """

    def __init__(self, round: int | None = None):
        """Initialize the frequency meter.

        Args:
            round: Number of decimal places to round results to. If None,
                results are not rounded.
        """
        self.round = round
        self.start: Optional[int] = None
        self.elapsed: int = 0
        self.counter: int = 0

    def log(self):
        """Record an occurrence and compute elapsed time since the last call.

        If this is the first call, it initializes the start time.
        Subsequent calls update the elapsed time and increment the counter.
        """
        if self.start is None:
            self.start = time.perf_counter_ns()
            return
        self.counter += 1
        self.elapsed += time.perf_counter_ns() - self.start
        self.start = time.perf_counter_ns()

    def compute(self) -> float | int | dict[str, float | int]:
        """Compute average time per occurrence in milliseconds.

        Returns:
            The average time per occurrence in milliseconds, optionally rounded.
            Returns 0 if no events have been recorded.
        """
        if self.counter == 0:
            return 0
        return safe_round(self.elapsed / self.counter / 1e6, self.round)

    def merge(self, other_state: dict[str, Any]):
        """Merge state from another FrequencyMeter.

        Args:
            other_state: State dictionary from another FrequencyMeter instance.
        """
        self.elapsed += other_state["elapsed"]
        self.counter += other_state["counter"]

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restore state and reset the start timer.

        Args:
            state_dict: State dictionary to restore from.
        """
        super().load_state_dict(state_dict)
        self.start = None  # Reset start timer on load to avoid inaccurate timing across checkpoints


class StopwatchMeter(BaseMeter):  # Ensures it inherits from BaseMeter
    """Meter that acts as a manual stopwatch for measuring event durations.

    Expects pairs of `log(mode="start")` and `log(mode="end")` calls.

    Attributes:
        round: Rounding precision for the result.
        _start: Internal timestamp of when the stopwatch was started.
        elapsed: Total elapsed time in nanoseconds across all recorded intervals.
        counter: Number of completed intervals.
    """

    def __init__(self, round: int | None = None):
        """Initialize the stopwatch meter.

        Args:
            round: Number of decimal places to round results to. If None,
                results are not rounded.
        """
        self.round = round
        self._start: float | None = None
        self.elapsed: int = 0
        self.counter: int = 0

    def log(self, mode: str):
        """Start or stop the timer.

        Args:
            mode: "start" to begin timing, "end" to stop and record duration.

        Raises:
            AssertionError: If an unknown mode is provided.
        """
        if mode == "start":
            self.start()
        elif mode == "end":
            self.end()
        else:
            raise AssertionError("Unknown mode")

    def start(self):
        """Start the timer."""
        self._start = time.perf_counter_ns()

    def end(self):
        """Stop the timer and record the duration.

        Raises:
            AssertionError: If `start()` was not called before `end()`.
        """
        assert self._start is not None, "Stopwatch was never started"
        self.elapsed += time.perf_counter_ns() - self._start
        self.counter += 1
        self._start = None

    def compute(self) -> float | int | dict[str, float | int]:
        """Compute average duration in milliseconds.

        Returns:
            The average duration in milliseconds, optionally rounded.
            Returns 0 if no intervals have been recorded.
        """
        if self.counter == 0:
            return 0
        return safe_round(self.elapsed / self.counter / 1e6, self.round)

    def merge(self, other_state: dict[str, Any]):
        """Merge state from another StopwatchMeter.

        Args:
            other_state: State dictionary from another StopwatchMeter instance.
        """
        self.elapsed += other_state["elapsed"]
        self.counter += other_state["counter"]

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restore state and reset current timer.

        Args:
            state_dict: State dictionary to restore from.
        """
        super().load_state_dict(state_dict)
        self._start = None  # Reset current timer on load to avoid inaccurate timing across checkpoints


class AveragedExponentMeter(BaseMeter):  # Ensures it inherits from BaseMeter
    """Meter that computes the exponent of a weighted average.

    Commonly used for computing perplexity (exp(loss)).

    Attributes:
        _internal: An `AverageMeter` instance used to compute the weighted average.
        round: Number of decimal places to round the final result to.
    """

    def __init__(self, round: int | None = None):
        """Initialize the averaged exponent meter.

        Args:
            round: Number of decimal places to round results to. If None,
                results are not rounded.
        """
        self._internal = AverageMeter()  # Changed to AverageMeter
        self.round = round

    def log(self, value: float | int, weight: float | int):
        """Log a log-scale value with its weight to the internal average meter.

        Args:
            value: The log-scale value to add.
            weight: The weight for this value.
        """
        self._internal.log(value, weight)

    def compute(self) -> float | int:
        """Return the exponent of the average.

        Computes the weighted average of logged values using the internal
        `AverageMeter` and then returns `exp()` of that average.

        Returns:
            The exponent of the weighted average, optionally rounded.
        """
        return safe_round(np.exp(self._internal.compute()), self.round)

    def merge(self, other_state: dict[str, Any]):
        """Merge state from another AveragedExponentMeter.

        Args:
            other_state: State dictionary from another AveragedExponentMeter instance.
        """
        self._internal.merge(other_state["internal"])

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restore state.

        Args:
            state_dict: State dictionary to restore from.
        """
        self._internal.load_state_dict(state_dict["internal"])
        self.round = state_dict["round"]

    def state_dict(self) -> dict[str, Any]:
        """Collect state for checkpointing.

        Returns:
            A dictionary containing the internal state of the `AverageMeter`
            and the rounding precision.
        """
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
    """Log a value to an averaged meter.

    This is a convenience function for logging values to an `AverageMeter`.
    The value and weight can be callables (lambdas) that are only evaluated
    when the meter is actually logged (lazy evaluation).

    Args:
        name: Name of the meter (e.g., "train/loss").
        value: Value to log. Can be a number or a callable that returns a number.
        weight: Weight for this value (typically batch size). Can be a number
            or a callable. Defaults to 1.0.
        round: Number of decimal places to round the result to.
        reset: If True, the meter is reset after logging (for per-iteration meters).
            If False, the meter accumulates across iterations.
        priority: Priority for meter ordering when logging. Higher priority
            meters appear first.

    Example:
        ```python
        # Log a simple value
        log_averaged("train/loss", 0.5, weight=32)

        # Log with lazy evaluation (only computed if meter is logged)
        log_averaged("train/loss", lambda: compute_loss(), weight=lambda: batch_size)

        # Log with rounding
        log_averaged("train/accuracy", 0.95, round=4)

        ```"""
    log_metric(
        name=name,
        meter_factory=lambda: AverageMeter(round=round),  # Changed to AverageMeter
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
    """Log a value to an averaged exponent meter.

    This is a convenience function for logging values to an `AveragedExponentMeter`.
    The value and weight can be callables (lambdas) for lazy evaluation.

    Args:
        name: Name of the meter (e.g., "train/perplexity").
        value: Log-scale value to log. Can be a number or a callable.
        weight: Weight for this value. Can be a number or a callable. Defaults to 1.0.
        round: Number of decimal places to round the result to.
        reset: If True, the meter is reset after logging.
        priority: Priority for meter ordering.
    """
    log_metric(
        name=name,
        meter_factory=lambda: AveragedExponentMeter(round=round),
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
    """Log a value to a summed meter.

    This is a convenience function for logging values to a `SummedMeter`.
    The value can be a callable (lambda) that is only evaluated when the
    meter is actually logged (lazy evaluation).

    Args:
        name: Name of the meter (e.g., "train/tokens_processed").
        value: Value to add to the sum. Can be a number or a callable that
            returns a number.
        round: Number of decimal places to round the result to.
        reset: If True, the meter is reset after logging. If False, the
            meter accumulates across iterations.
        priority: Priority for meter ordering when logging.

    Example:
        ```python
        # Log total tokens processed
        log_summed("train/tokens", batch_size * seq_len)

        # Log with lazy evaluation
        log_summed("train/tokens", lambda: get_token_count())

        ```"""
    log_metric(
        name=name,
        meter_factory=lambda: SummedMeter(round=round),  # Changed to SummedMeter
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
        reset: If True, the meter is reset after logging.
        priority: Priority for meter ordering when logging.

    Example:
        ```python
        log_event_start("perf/forward_pass")
        # ... do work ...
        log_event_end("perf/forward_pass")
        # Meter will show average duration in milliseconds

        ```"""
    log_metric(
        name=name,
        meter_factory=lambda: StopwatchMeter(round=round),
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
        reset: If True, the meter is reset after logging.
        priority: Priority for meter ordering when logging.

    Raises:
        AssertionError: If `log_event_start()` was not called for this event name.

    Example:
        ```python
        log_event_start("perf/backward_pass")
        # ... do backward pass ...
        log_event_end("perf/backward_pass")

        ```"""
    log_metric(
        name=name,
        meter_factory=lambda: StopwatchMeter(round=round),
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
    """Log an occurrence of an event and track its frequency.

    This function uses a `FrequencyMeter` to measure the time between
    successive calls to `log_event_occurrence()` for a given `name`.
    It can be used to track the rate at which certain events happen.

    Args:
        name: Name of the event to track (e.g., "perf/dataloader_ready").
        round: Number of decimal places to round the duration to (in milliseconds).
        reset: If True, the meter is reset after logging.
        priority: Priority for meter ordering when logging.
    """
    log_metric(
        name=name,
        meter_factory=lambda: FrequencyMeter(round=round),  # Changed to FrequencyMeter
        reset=reset,
        priority=priority,
        force_log=True,  # Always log event occurrences
    )


class CachedLambda:
    """Wrapper that caches the result of a callable function.

    This is useful for expensive computations that are used multiple times
    in meter logging. The function is only called once, and subsequent calls
    return the cached result.

    Example:
        ```python
        # Expensive computation
        def compute_expensive_value():
            return complex_calculation()

        cached = CachedLambda(compute_expensive_value)
        value1 = cached()  # Computes and caches
        value2 = cached()  # Returns cached value

        ```"""

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
        ```python
        expensive = cached_lambda(lambda: expensive_computation())
        result = expensive()  # Computes once
        result = expensive()  # Uses cache

        ```"""
    return CachedLambda(x)
