"""Performance profiling utilities.

This module provides functions for measuring execution time of code blocks,
iterators, and function calls. Supports both CPU timing (using perf_counter)
and GPU timing (using CUDA events for accurate GPU kernel timing).
"""

import logging
import time
from collections.abc import (
    Callable,
    Iterator,
)
from typing import TypeVar

import torch

logger = logging.getLogger(__name__)

try:
    import dcgm_agent
    import dcgm_fields
    import dcgm_structs
except ImportError:
    dcgm_structs = None
    dcgm_agent = None
    dcgm_fields = None

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
        ```python
        data = [1, 2, 3, 4, 5]
        for elapsed, item in measured_iter(iter(data)):
            print(f"Item {item} took {elapsed:.2f}ms")
        ```
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
        ```python
        data_iter = iter(dataloader)
        elapsed, batch = measured_next(data_iter)
        print(f"Data loading took {elapsed:.2f}ms")
        ```
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
        ```python
        def forward_pass():
            return model(input_ids)

        # CPU timing
        elapsed, output = measured_lambda(forward_pass, cuda_events=False)

        # GPU timing (more accurate for GPU ops)
        elapsed, output = measured_lambda(forward_pass, cuda_events=True)
        ```
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


def setup_dcgm(gpu_id=0):
    """Initializes DCGM, starts the engine, and watches the SM metrics."""
    if dcgm_structs is None:
        return None

    # 1. Load the underlying C-library
    dcgm_structs._dcgmInit()

    # 2. Initialize the agent and start the embedded engine
    dcgm_agent.dcgmInit()
    handle = dcgm_agent.dcgmStartEmbedded(dcgm_structs.DCGM_OPERATION_MODE_AUTO)

    # 3. Create a tracking group for our specific GPU
    group_id = dcgm_agent.dcgmGroupCreate(
        handle, dcgm_structs.DCGM_GROUP_EMPTY, "sm_group"
    )
    dcgm_agent.dcgmGroupAddDevice(handle, group_id, gpu_id)

    # 4. Create a field group with our specific metrics
    field_ids = [
        dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
        dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
    ]
    field_group_id = dcgm_agent.dcgmFieldGroupCreate(handle, field_ids, "sm_fields")

    # 5. Start watching (Update every 1 second = 1,000,000 microseconds)
    dcgm_agent.dcgmWatchFields(handle, group_id, field_group_id, 1000000, 3600.0, 0)

    return handle


def get_sm_metrics(handle, gpu_id=0):
    """Queries the current SM metrics."""
    if dcgm_agent is None:
        return None
    try:
        values = dcgm_agent.dcgmEntityGetLatestValues(
            handle,
            dcgm_fields.DCGM_FE_GPU,
            gpu_id,
            [dcgm_fields.DCGM_FI_PROF_SM_ACTIVE, dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY],
        )

        # --- THE FIX ---
        # Extract the double (.dbl) field from the C-Union
        # We also check the .status to ensure DCGM successfully read the sensor.
        sm_active_val = (
            values[0].value.dbl if values[0].status == dcgm_structs.DCGM_ST_OK else 0.0
        )
        sm_occupancy_val = (
            values[1].value.dbl if values[1].status == dcgm_structs.DCGM_ST_OK else 0.0
        )

        # Filter out DCGM blank values (which are returned as massive negative numbers)
        sm_active = max(0.0, sm_active_val)
        sm_occupancy = max(0.0, sm_occupancy_val)

        return {
            "gpu_sm_active_pct": sm_active,
            "gpu_sm_occupancy_pct": sm_occupancy,
        }
    except Exception as e:
        # Actually print the error so it doesn't fail silently
        logger.warning(f"DEBUG ERROR: {e}")
        return None


def teardown_dcgm(handle):
    """Cleans up the DCGM engine."""
    if handle is None or dcgm_agent is None:
        return None
    dcgm_agent.dcgmStopEmbedded(handle)
    dcgm_agent.dcgmShutdown()
