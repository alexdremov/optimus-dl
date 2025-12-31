import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optimus_dl.modules.data.datasets.strategies.concat_random import (
    ConcatAndChunkFullRandomConfig,
)
from optimus_dl.modules.data.datasets.strategies.document import (
    DocumentStrategyConfig,
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
) -> tuple[int, int, np.ndarray]:
    """Creates mock tokenized data and returns (total_docs, total_tokens, all_tokens)."""
    total_docs = sum(docs_per_shard)
    total_tokens = 0
    dtype: np.dtype = np.uint16
    doc_dtype: np.dtype = np.uint32

    files_meta = []
    all_tokens_list = []

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
        all_tokens_list.append(shard_tokens_flat)

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

    return total_docs, total_tokens, np.concatenate(all_tokens_list)


class TestDatasetsTotality(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir()

        self.num_shards = 4
        self.docs_per_shard = [15, 20, 10, 25]
        self.avg_doc_len = 50
        self.vocab_size = 1000
        self.total_docs, self.total_tokens, self.all_tokens_gt = create_mock_data(
            self.data_dir,
            self.num_shards,
            self.docs_per_shard,
            self.avg_doc_len,
            self.vocab_size,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _collect_distributed(self, config, world_size):
        collected_tokens = []
        collected_docs_count = 0

        for rank in range(world_size):
            dataset = TokenizedDataset(config, rank=rank, world_size=world_size)
            dataset.reset()

            rank_tokens = []
            for item in dataset:
                tokens = item["input_ids"]
                rank_tokens.append(tokens)
                collected_docs_count += (
                    1  # Rough count, assumes 1 item = 1 unit of work
                )

            if rank_tokens:
                collected_tokens.append(np.concatenate(rank_tokens))

        if collected_tokens:
            return np.concatenate(collected_tokens), collected_docs_count
        return np.array([], dtype=np.uint16), collected_docs_count

    def test_document_strategy_sequential_totality(self):
        # Sequential should yield exactly all tokens in order (conceptually, if we sort by doc id)
        # But even simpler: sum of all ranks should equal total tokens
        config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            strategy=DocumentStrategyConfig(_name="document", shuffle=False),
        )

        world_size = 3
        collected, doc_count = self._collect_distributed(config, world_size)

        assert len(collected) == self.total_tokens
        assert doc_count == self.total_docs

        # Verify content match (Sequential yields in order of shards/docs)
        np.testing.assert_array_equal(collected, self.all_tokens_gt)

    def test_document_strategy_shuffled_totality(self):
        # Shuffled should yield all tokens, but order will be permuted at document level
        config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            strategy=DocumentStrategyConfig(_name="document", shuffle=True, seed=123),
        )

        world_size = 3
        collected, doc_count = self._collect_distributed(config, world_size)

        assert len(collected) == self.total_tokens
        assert doc_count == self.total_docs

        # Content verification: Sort both and compare (Histogram match is sufficient for tokens,
        # but exact match sorted is stronger)
        # Note: Sorting 100k tokens is fast.
        collected_sorted = np.sort(collected)
        gt_sorted = np.sort(self.all_tokens_gt)
        np.testing.assert_array_equal(collected_sorted, gt_sorted)

    def test_concat_random_strategy_totality(self):
        # ConcatRandom with offset=False should yield all tokens (truncated to chunk size if remainder exists)
        chunk_size = 64
        config = TokenizedDatasetConfig(
            data_dir=str(self.data_dir),
            strategy=ConcatAndChunkFullRandomConfig(
                _name="concat_random",
                chunk_size=chunk_size,
                random_offset=False,
                seed=42,
            ),
        )

        world_size = 2
        collected, chunks_count = self._collect_distributed(config, world_size)

        # ConcatRandom drops the last incomplete chunk?
        # Let's check implementation.
        # `available_tokens // chunk_size`. Yes, truncates.

        expected_len = (self.total_tokens // chunk_size) * chunk_size

        assert len(collected) == expected_len
        assert chunks_count == (self.total_tokens // chunk_size)

        # Since it's shuffled chunks, we sort to compare content
        # We compare against the first `expected_len` tokens of GT?
        # No, because chunks are shuffled globally.
        # We compare against the FULL GT, but we might be missing some tokens from the end of the stream?
        # Actually, ConcatRandom treats stream as one big array.
        # It takes `num_chunks` chunks.
        # It permutes them.
        # So we get `num_chunks * chunk_size` tokens.
        # These tokens correspond to `[0 : expected_len]` of the global stream if offset=0?
        # Wait, `global_starts = global_offset + (chunk_indices * chunk_size)`.
        # `chunk_indices` are `0..num_chunks-1`.
        # So we cover `[0, chunk_size)`, `[chunk_size, 2*chunk_size)`, ...
        # Yes, we cover exactly the first `expected_len` tokens of the global concatenated stream.

        gt_truncated = self.all_tokens_gt[:expected_len]

        collected_sorted = np.sort(collected)
        gt_sorted = np.sort(gt_truncated)

        np.testing.assert_array_equal(collected_sorted, gt_sorted)
