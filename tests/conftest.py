import torch
import pytest
from _pytest.mark import Mark


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


def by_slow_marker(item):
    return item.get_closest_marker("slow", default=empty_mark).name == "slow"


def pytest_collection_modifyitems(items):
    items.sort(key=by_slow_marker, reverse=False)
