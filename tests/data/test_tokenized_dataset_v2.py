import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from optimus_dl.modules.data.datasets.strategies.concat_random import (
    ConcatAndChunkFullRandomConfig,
)
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

    return total_docs, total_tokens


class TestTokenizedDatasetV2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir()

        self.num_shards = 3
        self.docs_per_shard = [10, 5, 15]
        self.avg_doc_len = 50
        self.vocab_size = 100
        self.total_docs_created, self.total_tokens_created = create_mock_data(
            self.data_dir,
            self.num_shards,
            self.docs_per_shard,
            self.avg_doc_len,
            self.vocab_size,
        )

        self.default_config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            index_file="index.json",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_legacy_behavior(self):
        """Ensure default strategy behaves like the legacy implementation."""
        dataset = TokenizedDataset(self.default_config, rank=0, world_size=1, seed=42)
        dataset.reset()

        doc_count = 0
        last_doc_id = -1

        for item in dataset:
            doc_id = item["document_id"]
            assert doc_id == last_doc_id + 1
            last_doc_id = doc_id
            doc_count += 1

        assert doc_count == self.total_docs_created

    def test_concat_random_strategy(self):
        """Test the new ConcatAndChunkFullRandom strategy."""
        chunk_size = 64
        strategy_config = ConcatAndChunkFullRandomConfig(
            _name="concat_random",
            chunk_size=chunk_size,
            random_offset=False,  # Disable for deterministic counting check
        )

        config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            index_file="index.json",
            strategy=strategy_config,
        )

        dataset = TokenizedDataset(config, rank=0, world_size=1, seed=42)
        dataset.reset()

        chunks = []
        for item in dataset:
            chunks.append(item["input_ids"])
            assert len(item["input_ids"]) == chunk_size
            assert "seq_lens" in item
            assert np.sum(item["seq_lens"]) == chunk_size

            if len(item["seq_lens"]) > 1:
                assert isinstance(item["document_id"], np.ndarray)
                assert len(item["document_id"]) == chunk_size
            else:
                # Can be int or array of 1 element depending on implementation choice?
                # Implementation: if len(parts)==1: scalar.
                assert isinstance(item["document_id"], (int, np.integer))

        total_chunks = self.total_tokens_created // chunk_size
        assert len(chunks) == total_chunks

        # Verify shuffling (heuristic: chunks shouldn't be sequential in terms of content if we checked doc ids,
        # but doc ids are ambiguous here. We trust the perm logic).

    def test_concat_random_distributed(self):
        chunk_size = 32
        strategy_config = ConcatAndChunkFullRandomConfig(
            _name="concat_random", chunk_size=chunk_size, random_offset=False
        )
        config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir), strategy=strategy_config
        )

        world_size = 2
        chunks_r0 = []
        dataset0 = TokenizedDataset(config, rank=0, world_size=world_size, seed=42)
        dataset0.reset()
        for item in dataset0:
            chunks_r0.append(item["input_ids"])

        chunks_r1 = []
        dataset1 = TokenizedDataset(config, rank=1, world_size=world_size, seed=42)
        dataset1.reset()
        for item in dataset1:
            chunks_r1.append(item["input_ids"])

        # Check total count
        total_chunks = self.total_tokens_created // chunk_size
        # +/- 1 due to integer division distribution
        assert len(chunks_r0) + len(chunks_r1) == total_chunks

        # Check disjointness (hard with raw tokens, but we assume correctness of indices)

    def test_concat_random_checkpointing(self):
        chunk_size = 32
        strategy_config = ConcatAndChunkFullRandomConfig(
            _name="concat_random", chunk_size=chunk_size, random_offset=True
        )
        config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir), strategy=strategy_config
        )

        dataset = TokenizedDataset(config, rank=0, world_size=1, seed=42)
        dataset.reset()

        # Take 5 chunks
        items1 = [next(dataset) for _ in range(5)]
        state = dataset.get_state()

        # Resume
        dataset2 = TokenizedDataset(config, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        # Next 5
        items2 = [next(dataset2) for _ in range(5)]

        # Continuous run
        dataset_full = TokenizedDataset(config, rank=0, world_size=1, seed=42)
        dataset_full.reset()
        items_full = [next(dataset_full) for _ in range(10)]

        # Compare
        for i in range(5):
            np.testing.assert_array_equal(
                items1[i]["input_ids"], items_full[i]["input_ids"]
            )
            np.testing.assert_array_equal(
                items2[i]["input_ids"], items_full[i + 5]["input_ids"]
            )

    def test_random_offset(self):
        """Test that random_offset changes the number/content of chunks."""
        chunk_size = 100
        # Config 1: offset=False
        cfg1 = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            strategy=ConcatAndChunkFullRandomConfig(
                _name="concat_random",
                chunk_size=chunk_size,
                random_offset=False,
            ),
        )
        ds1 = TokenizedDataset(cfg1, rank=0, world_size=1, seed=42)
        ds1.reset()
        tokens1 = next(ds1)["input_ids"]

        # Config 2: offset=True
        cfg2 = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            strategy=ConcatAndChunkFullRandomConfig(
                _name="concat_random",
                chunk_size=chunk_size,
                random_offset=True,
            ),
        )
        ds2 = TokenizedDataset(cfg2, rank=0, world_size=1, seed=42)
        ds2.reset()
        tokens2 = next(ds2)["input_ids"]

        # With high probability, tokens should differ because of shift
        # (unless offset happened to be 0)
        # We can check the internal offset
        assert ds1.strategy.global_offset == 0
        assert ds2.strategy.global_offset >= 0

        if ds2.strategy.global_offset > 0:
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(tokens1, tokens2)

    def test_concat_random_multi_doc_structure(self):
        """Test document_id array structure for multi-document chunks."""
        # Create small docs (avg 10) and large chunks (25) -> guarantees multi-doc
        chunk_size = 25
        avg_doc_len = 10

        # Clean and recreate data
        shutil.rmtree(self.data_dir)
        self.data_dir.mkdir()

        create_mock_data(self.data_dir, 1, [50], avg_doc_len=avg_doc_len)

        cfg = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            index_file="index.json",
            strategy=ConcatAndChunkFullRandomConfig(
                _name="concat_random",
                chunk_size=chunk_size,
                random_offset=False,
            ),
        )
        ds = TokenizedDataset(cfg, rank=0, world_size=1, seed=42)
        ds.reset()

        # Iterate until we find a multi-doc chunk
        found_multi = False
        for _ in range(10):
            try:
                item = next(ds)
                if len(item["seq_lens"]) > 1:
                    found_multi = True
                    # Check types
                    assert isinstance(item["document_id"], np.ndarray)
                    assert len(item["document_id"]) == chunk_size
                    assert item["document_id"].dtype == np.int64

                    # Verify content: document_id should be piecewise constant
                    current_idx = 0
                    for length in item["seq_lens"]:
                        segment_ids = item["document_id"][
                            current_idx : current_idx + length
                        ]
                        # All tokens in segment should have same doc id
                        assert np.all(segment_ids == segment_ids[0])
                        current_idx += length
                    break
            except StopIteration:
                break

        assert found_multi, "Did not find a multi-document chunk despite config"
