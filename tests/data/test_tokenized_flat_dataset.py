import tempfile

import numpy as np
import torchdata
import torchdata.nodes

from optimus_dl.modules.data.datasets.tokenized_flat_dataset import (
    TokenizedFlatDataset,
    TokenizedFlatDatasetConfig,
)


def make_simple_config(max_elem):
    """Create a simple test configuration with two data files containing sequential integers.

    Creates temporary files with sequential integer data split at 80% mark.
    """
    sep = int(max_elem * 0.8)
    with tempfile.NamedTemporaryFile("wb", delete=False) as file:
        file.write(np.arange(sep).astype(np.uint8).tobytes())
    with tempfile.NamedTemporaryFile("wb", delete=False) as file2:
        file2.write(np.arange(sep, max_elem).astype(np.uint8).tobytes())

    return TokenizedFlatDatasetConfig(
        dtype="np.uint8",
        files=[
            file.name,
            file2.name,
        ],
        seq_len=5,
        batch_size=1,
    )

    def test_basic():
        """Test basic functionality of TokenizedFlatDataset with concatenated input sequences."""
        config = make_simple_config(18)
        dataset = TokenizedFlatDataset(config, rank=0, world_size=1)
        dataset = torchdata.nodes.Loader(dataset)

    results = list(dataset)

    assert (
        np.concatenate([i["input_ids"].reshape(-1) for i in results])
        == np.arange(15, dtype=np.uint8)
    ).all()


def test_multiworld():
    """Test distributed training setup with multiple workers accessing different data shards."""
    max_elem = 100
    config = make_simple_config(max_elem + 4)
    results = []
    world_size = 5
    for rank in range(world_size):
        dataset = TokenizedFlatDataset(cfg=config, rank=rank, world_size=world_size)
        dataset = torchdata.nodes.Loader(dataset)
        results += list(dataset)

    res = np.concatenate([i["input_ids"].reshape(-1) for i in results])
    assert (res == np.arange(max_elem, dtype=np.uint8)).all()
