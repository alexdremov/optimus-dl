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
    """Add memory usage summary to the terminal output, sorted by RSS usage."""
    if "passed" not in terminalreporter.stats:
        return

    import re

    terminalreporter.section("Memory Usage Report")

    # Collect reports and extract memory values for sorting
    reports_with_mem = []
    for report in terminalreporter.stats["passed"]:
        memory_info = "N/A"
        rss_value = 0.0
        for key, value in report.user_properties:
            if key == "memory":
                memory_info = value
                # Extract numeric RSS for sorting
                match = re.search(r"RSS: ([\d.]+)MB", value)
                if match:
                    rss_value = float(match.group(1))
                break
        reports_with_mem.append((report.nodeid, memory_info, rss_value))

    # Sort by RSS value decreasing
    reports_with_mem.sort(key=lambda x: x[2], reverse=True)

    for nodeid, memory_info, _ in reports_with_mem:
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
