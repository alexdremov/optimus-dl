import gc
import platform
import resource

import torch
import pytest
from _pytest.mark import Mark


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Global teardown to free memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_peak_rss_mb():
    """Get peak RSS memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Record memory usage during the test call."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    yield

    report_parts = []
    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
        report_parts.append(f"GPU: {peak_gpu:.1f}MB")

    # RSS peak is since process start, so it's not perfect for individual tests
    # but still gives a good idea of the growth.
    peak_rss = get_peak_rss_mb()
    report_parts.append(f"RSS: {peak_rss:.1f}MB")

    if report_parts:
        item.user_properties.append(("memory", " | ".join(report_parts)))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add memory usage summary to the terminal output."""
    if "passed" not in terminalreporter.stats:
        return

    terminalreporter.section("Memory Usage Report")
    for reports in terminalreporter.stats["passed"]:
        nodeid = reports.nodeid
        # Extract memory info from user_properties
        memory_info = "N/A"
        for key, value in reports.user_properties:
            if key == "memory":
                memory_info = value
                break

        # Shorten nodeid for better readability
        short_id = nodeid.split("/")[-1]
        terminalreporter.write_line(f"{short_id:<60} {memory_info}")


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return torch.device(request.param)


@pytest.fixture
def unique_port(worker_id):
    if worker_id == "master":
        return 29500
    worker_num = int(worker_id.replace("gw", ""))
    return 29500 + worker_num


empty_mark = Mark("", [], {})
slow_mark = Mark("slow", [], {})


def by_slow_marker(item):
    return item.get_closest_marker("slow", default=empty_mark).name == "slow"


def pytest_collection_modifyitems(items):
    items.sort(key=by_slow_marker, reverse=False)
