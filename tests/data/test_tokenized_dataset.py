import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from optimus_dl.modules.data.datasets.tokenized_dataset import (
    TokenizedDataset,
    TokenizedDatasetConfig,
)


def create_mock_data(
    data_dir: Path,
    num_shards: int,
    docs_per_shard: list[int],
    avg_doc_len: int,
    vocab_size: int = 100,
    dtype: np.dtype = np.uint16,
    doc_dtype: np.dtype = np.uint32,
) -> int:
    """Creates mock tokenized data and lens files, and an index.json."""
    total_docs = sum(docs_per_shard)
    total_tokens = 0

    files_meta = []

    for i in range(num_shards):
        num_docs_in_shard = docs_per_shard[i % len(docs_per_shard)]
        shard_doc_lens = np.random.randint(
            max(1, avg_doc_len // 2), avg_doc_len * 2, size=num_docs_in_shard
        ).astype(doc_dtype)

        shard_tokens_flat = np.concatenate(
            [
                np.random.randint(0, vocab_size, size=length, dtype=dtype)
                for length in shard_doc_lens
            ]
        )

        token_file = data_dir / f"test_data_{i:010d}.npy"
        lens_file = data_dir / f"test_data_{i:010d}_lens.npy"

        np.save(token_file, shard_tokens_flat)
        np.save(lens_file, shard_doc_lens)

        files_meta.append(
            {
                "file": token_file.name,
                "lens_file": lens_file.name,
                "num_tokens": len(shard_tokens_flat),
                "num_docs": num_docs_in_shard,
                "shard_idx": i,
            }
        )
        total_tokens += len(shard_tokens_flat)

    index_data = {
        "files": files_meta,
        "total_tokens": total_tokens,
        "config": {},  # Simplified config
    }

    with open(data_dir / "index.json", "w") as f:
        json.dump(index_data, f, indent=2)

    return total_docs


class TestTokenizedDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir()

        self.num_shards = 3
        self.docs_per_shard = [10, 5, 15]  # Different doc counts per shard
        self.avg_doc_len = 50
        self.vocab_size = 100
        self.total_docs_created = create_mock_data(
            self.data_dir,
            self.num_shards,
            self.docs_per_shard,
            self.avg_doc_len,
            self.vocab_size,
        )

        self.config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            dtype="np.uint16",
            doc_dtype="np.uint32",
            index_file="index.json",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init_and_load_index(self):
        dataset = TokenizedDataset(self.config, rank=0, world_size=1)
        dataset.reset()

        assert dataset.total_docs == self.total_docs_created
        assert len(dataset.shards) == self.num_shards
        assert dataset.shard_num_docs == self.docs_per_shard
        assert dataset.start_doc_idx == 0
        assert dataset.end_doc_idx == self.total_docs_created
        assert dataset.global_doc_idx == 0
        assert dataset.current_shard_idx == 0
        assert dataset.shard_doc_offset == 0
        assert dataset.shard_token_offset == 0

    def test_iteration_single_rank(self):
        dataset = TokenizedDataset(self.config, rank=0, world_size=1)
        dataset.reset()

        doc_count = 0
        tokens_seen = 0
        last_doc_id = -1

        for item in dataset:
            assert "input_ids" in item
            assert "document_id" in item
            assert isinstance(item["input_ids"], np.ndarray)
            assert isinstance(item["document_id"], int)

            assert item["input_ids"].dtype == dataset.dtype
            assert item["document_id"] == doc_count
            assert item["document_id"] > last_doc_id

            tokens_seen += len(item["input_ids"])
            last_doc_id = item["document_id"]
            doc_count += 1

        assert doc_count == self.total_docs_created

    def test_distributed_sharding(self):
        world_size = 3

        all_docs = []
        for rank in range(world_size):
            dataset = TokenizedDataset(self.config, rank=rank, world_size=world_size)
            dataset.reset()

            rank_docs = []
            for item in dataset:
                rank_docs.append(item["document_id"])
            all_docs.extend(rank_docs)

        # Check that all documents are present and unique
        assert len(all_docs) == self.total_docs_created
        assert len(set(all_docs)) == self.total_docs_created
        assert sorted(all_docs) == list(range(self.total_docs_created))

        # Check rank boundaries
        dataset_rank0 = TokenizedDataset(self.config, rank=0, world_size=world_size)
        dataset_rank0.reset()
        assert dataset_rank0.start_doc_idx == 0
        assert dataset_rank0.end_doc_idx == (self.total_docs_created // world_size)

        dataset_rank_last = TokenizedDataset(
            self.config, rank=world_size - 1, world_size=world_size
        )
        dataset_rank_last.reset()
        assert (
            dataset_rank_last.end_doc_idx == self.total_docs_created
        )  # Last rank gets remainder

    def test_state_restoration(self):
        dataset = TokenizedDataset(self.config, rank=0, world_size=1)
        dataset.reset()

        # Consume some documents
        num_to_consume = 5
        consumed_items = [dataset.next() for _ in range(num_to_consume)]

        # Save state
        state = dataset.get_state()

        # Create new dataset and load state
        dataset2 = TokenizedDataset(self.config, rank=0, world_size=1)
        dataset2.reset(state)

        # Verify state is restored correctly
        assert dataset2.global_doc_idx == state["global_doc_idx"]
        assert dataset2.global_doc_idx == num_to_consume
        assert dataset2.rank == state["rank"]
        assert dataset2.world_size == state["world_size"]

        # Continue iterating from dataset2
        remaining_items = []
        try:
            while True:
                remaining_items.append(dataset2.next())
        except StopIteration:
            pass

        # Verify total docs processed is correct
        total_processed_docs = num_to_consume + len(remaining_items)
        assert total_processed_docs == self.total_docs_created

        # Verify no duplication or missing docs
        all_doc_ids = [item["document_id"] for item in consumed_items] + [
            item["document_id"] for item in remaining_items
        ]
        assert sorted(all_doc_ids) == list(range(self.total_docs_created))

    def test_exact_reproducibility_after_restore(self):
        # Consume some documents in dataset1
        dataset1 = TokenizedDataset(self.config, rank=0, world_size=1)
        dataset1.reset()

        [dataset1.next() for _ in range(self.total_docs_created // 3)]

        # Save state
        state = dataset1.get_state()

        # Continue with dataset1
        items1_part2 = []
        try:
            while True:
                items1_part2.append(dataset1.next())
        except StopIteration:
            pass

        # Create dataset2, load state, and iterate
        dataset2 = TokenizedDataset(self.config, rank=0, world_size=1)
        dataset2.reset(state)

        items2_part2 = []
        try:
            while True:
                items2_part2.append(dataset2.next())
        except StopIteration:
            pass

        # Verify that the sequence after restoration is identical
        assert len(items1_part2) == len(items2_part2)
        for i in range(len(items1_part2)):
            np.testing.assert_array_equal(
                items1_part2[i]["input_ids"], items2_part2[i]["input_ids"]
            )
            assert items1_part2[i]["document_id"] == items2_part2[i]["document_id"]

    def test_empty_dataset(self):
        empty_data_dir = Path(self.temp_dir) / "empty_data"
        empty_data_dir.mkdir()

        # Create an empty index file
        with open(empty_data_dir / "index.json", "w") as f:
            json.dump({"files": [], "total_tokens": 0, "config": {}}, f)

        empty_config = TokenizedDatasetConfig(data_dir=str(empty_data_dir))
        dataset = TokenizedDataset(empty_config, rank=0, world_size=1)

        with pytest.raises(StopIteration):
            dataset.next()

    def test_data_corruption_lens_mismatch(self):
        # Create mock data but corrupt a lens file
        corrupt_data_dir = Path(self.temp_dir) / "corrupt_data"
        corrupt_data_dir.mkdir()

        # Create normal data first
        create_mock_data(corrupt_data_dir, 1, [10], 50)

        # Corrupt lens file: make it shorter than tokens suggest
        lens_file = corrupt_data_dir / "test_data_0000000000.npy"
        corrupt_data_dir / "test_data_0000000000_lens.npy"

        corrupted_lens = np.array(
            [5, 5, 5], dtype=np.uint32
        )  # Fewer docs than expected
        np.save(lens_file, corrupted_lens)

        config = TokenizedDatasetConfig(data_dir=str(corrupt_data_dir))
        dataset = TokenizedDataset(config, rank=0, world_size=1)
        dataset.reset()

        # Iterate until error or exhaustion
        with pytest.raises(RuntimeError, match="Data corruption"):
            for _ in range(self.total_docs_created + 5):  # Iterate past expected
                dataset.next()

    def test_limit_parameter(self):
        self.config.limit = 10  # Limit to 10 documents
        dataset = TokenizedDataset(self.config, rank=0, world_size=1)
        dataset.reset()

        doc_count = 0
        try:
            while True:
                dataset.next()
                doc_count += 1
        except StopIteration:
            pass

        assert doc_count == 10

    def test_load_config_type_coercion(self):
        # Test that dtype and doc_dtype are correctly resolved from string
        config_str = TokenizedDatasetConfig(
            data_dir=str(self.data_dir), dtype="int32", doc_dtype="int64"
        )
        dataset = TokenizedDataset(config_str, rank=0, world_size=1)
        dataset.reset()

        item = dataset.next()
        assert item["input_ids"].dtype == np.int32
