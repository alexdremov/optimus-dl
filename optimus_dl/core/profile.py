import time

import torch


def measured_iter(itr):
    start = time.perf_counter_ns()
    for elem in itr:
        elapsed = (time.perf_counter_ns() - start) / 1e6  # Convert to milliseconds
        yield elapsed, elem
        start = time.perf_counter_ns()


def measured_next(itr):
    start = time.perf_counter_ns()
    elem = next(itr)
    elapsed = (time.perf_counter_ns() - start) / 1e6  # Convert to milliseconds
    return elapsed, elem


def measured_lambda(f, cuda_events=False, enabled=True):
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
