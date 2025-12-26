import pytest


@pytest.fixture
def unique_port(worker_id):
    if worker_id == "master":
        return 29500
    worker_num = int(worker_id.replace("gw", ""))
    return 29500 + worker_num
