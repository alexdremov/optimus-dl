import time
from typing import Any, Callable

import numpy as np
import torch

from .base import BaseMetric, log_metric


def safe_round(number, ndigits) -> float | int:
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
    def __init__(self, round: int | None = None):
        self.round = round
        self.sum = 0
        self.count = 0

    def compute(self) -> float | int:
        return safe_round(self.sum / self.count, self.round)

    def log(self, value, weight):
        self.sum += value * weight
        self.count += weight

    def merge(self, other_state):
        self.sum += other_state["sum"]
        self.count += other_state["count"]


class SummedMetric(BaseMetric):
    def __init__(self, round: int | None = None):
        self.round = round
        self.sum = 0

    def compute(self):
        return self.sum

    def log(self, value):
        self.sum += value

    def merge(self, other_state):
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
            assert False, "Unknown mode"

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
        return dict(
            internal=self._internal.state_dict(),
            round=self.round,
        )


DelayedValue = Any | Callable[[], Any]


def log_averaged(
    name: str,
    value: DelayedValue,
    weight: DelayedValue = 1.0,
    round: int | None = None,
    reset: bool = True,
    priority: int = 100,
):
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
):
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
):
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
):
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
    """
    A simple wrapper to cache the result of a lambda function.
    """

    def __init__(self, func: Callable[[], Any]):
        self._func = func
        self._cache = None
        self._cached = False

    def __call__(self):
        if not self._cached:
            self._cache = self._func()
            self._cached = True
        return self._cache


def cached_lambda(x):
    return CachedLambda(x)
