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
) -> int:
    """Creates mock tokenized data and lens files, and an index.json."""
    total_docs = sum(docs_per_shard)
    total_tokens = 0
    dtype: np.dtype = np.uint16
    doc_dtype: np.dtype = np.uint32

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
        "config": {
            "dtype": "np.uint16",
        },
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
            index_file="index.json",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init_and_load_index(self):
        dataset = TokenizedDataset(self.config, rank=0, world_size=1, seed=42)
        dataset.reset()

        assert dataset.total_docs == self.total_docs_created
        assert len(dataset.shards) == self.num_shards
        assert dataset.shard_num_docs == self.docs_per_shard
        # Strategy specific checks
        assert len(dataset.strategy.indices) == self.total_docs_created

    def test_iteration_single_rank(self):
        dataset = TokenizedDataset(self.config, rank=0, world_size=1, seed=42)
        dataset.reset()

        doc_count = 0
        tokens_seen = 0
        last_doc_id = -1

        for item in dataset:
            assert "input_ids" in item
            assert "document_id" in item
            assert "seq_lens" in item

            assert isinstance(item["input_ids"], np.ndarray)
            assert isinstance(item["seq_lens"], np.ndarray)
            # Default strategy yields full docs, so document_id should be scalar int
            assert isinstance(item["document_id"], (int, np.integer))

            assert item["input_ids"].dtype == dataset.dtype
            assert item["document_id"] == doc_count
            assert item["document_id"] > last_doc_id

            # Verify seq_lens logic: should contain single length matching input_ids
            assert len(item["seq_lens"]) == 1
            assert item["seq_lens"][0] == len(item["input_ids"])

            tokens_seen += len(item["input_ids"])
            last_doc_id = item["document_id"]
            doc_count += 1

        assert doc_count == self.total_docs_created

    def test_distributed_sharding(self):
        world_size = 3

        all_docs = []
        for rank in range(world_size):
            dataset = TokenizedDataset(
                self.config, rank=rank, world_size=world_size, seed=42
            )
            dataset.reset()

            rank_docs = []
            for item in dataset:
                rank_docs.append(item["document_id"])
            all_docs.extend(rank_docs)

        # Check that all documents are present and unique
        assert len(all_docs) == self.total_docs_created
        assert len(set(all_docs)) == self.total_docs_created
        assert sorted(all_docs) == list(range(self.total_docs_created))

        # We implicitly verify partitioning by checking all docs are covered and unique

    def test_state_restoration(self):
        dataset = TokenizedDataset(self.config, rank=0, world_size=1, seed=42)
        dataset.reset()

        # Consume some documents
        num_to_consume = 5
        consumed_items = [next(dataset) for _ in range(num_to_consume)]

        # Save state
        state = dataset.get_state()

        # Create new dataset and load state
        dataset2 = TokenizedDataset(self.config, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        # Verify state is restored correctly (via strategy check)
        assert dataset2.strategy.ptr == num_to_consume
        assert dataset2.rank == state["rank"]
        assert dataset2.world_size == state["world_size"]

        # Continue iterating from dataset2
        remaining_items = []
        try:
            while True:
                remaining_items.append(next(dataset2))
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
        dataset1 = TokenizedDataset(self.config, rank=0, world_size=1, seed=42)
        dataset1.reset()

        [next(dataset1) for _ in range(self.total_docs_created // 3)]

        # Save state
        state = dataset1.get_state()

        # Continue with dataset1
        items1_part2 = []
        try:
            while True:
                items1_part2.append(next(dataset1))
        except StopIteration:
            pass

        # Create dataset2, load state, and iterate
        dataset2 = TokenizedDataset(self.config, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        items2_part2 = []
        try:
            while True:
                items2_part2.append(next(dataset2))
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

        # Create an empty index file with required metadata
        with open(empty_data_dir / "index.json", "w") as f:
            json.dump(
                {"files": [], "total_tokens": 0, "config": {"dtype": "np.uint16"}}, f
            )

        empty_config = TokenizedDatasetConfig(data_dir=str(empty_data_dir))
        dataset = TokenizedDataset(empty_config, rank=0, world_size=1, seed=42)
        dataset.reset()

        with pytest.raises(StopIteration):
            next(dataset)

    def test_data_corruption_lens_mismatch(self):
        # Create mock data but corrupt a lens file
        corrupt_data_dir = Path(self.temp_dir) / "corrupt_data"
        corrupt_data_dir.mkdir()

        # Create normal data first
        create_mock_data(corrupt_data_dir, 1, [10], 50)

        # Corrupt lens file: make it shorter than tokens suggest
        # Wait, if we make lens file imply MORE tokens than exist, we trigger the check.
        # Original test made lens imply FEWER docs?
        # "make it shorter than tokens suggest" -> if lens says 5, 5, 5 (15 tokens)
        # but token file has 500 tokens. This is fine, we just ignore extra tokens.
        # But if lens says 1000 tokens, and file has 500, we crash.

        lens_file = corrupt_data_dir / "test_data_0000000000.npy"

        # Make lens imply huge length
        corrupted_lens = np.array([10000], dtype=np.uint32)
        np.save(lens_file, corrupted_lens)

        config = TokenizedDatasetConfig(data_dir=str(corrupt_data_dir))
        dataset = TokenizedDataset(config, rank=0, world_size=1, seed=42)
        dataset.reset()

        # Iterate until error or exhaustion
        with pytest.raises(RuntimeError, match="Data corruption"):
            next(dataset)

    def test_limit_parameter(self):
        self.config.limit = 10  # Limit to 10 documents
        dataset = TokenizedDataset(self.config, rank=0, world_size=1, seed=42)
        dataset.reset()

        doc_count = 0
        try:
            while True:
                next(dataset)
                doc_count += 1
        except StopIteration:
            pass

        assert doc_count == 10

    def test_load_config_type_coercion(self):
        # Test that dtype and doc_dtype are correctly resolved from string
        config_str = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
        )
        dataset = TokenizedDataset(config_str, rank=0, world_size=1, seed=42)
        dataset.reset()

        item = next(dataset)
        assert item["input_ids"].dtype == np.uint16
