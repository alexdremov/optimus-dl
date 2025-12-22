import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.recipe.data_prep.config import (
    DataPrepConfig,
    DatasetConfig,
    OutputConfig,
    ProcessingConfig,
)
from optimus_dl.recipe.data_prep.recipe import DataPrepRecipe


class MockTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=1000):
        self._vocab_size = vocab_size
        self.config = {"_name": "mock_tokenizer"}

    def encode(self, text: str) -> list[int]:
        # Simple mock encoding: ascii values of characters
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([chr(i) for i in ids])

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class TestDataPrep(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.output_dir.mkdir()
        self.cache_dir.mkdir()

        # Mock dependencies
        try:
            from optimus_dl.modules.tokenizer import register_tokenizer

            register_tokenizer("mock_tokenizer")(MockTokenizer)
        except AssertionError:
            pass  # Already registered

        self.config = DataPrepConfig(
            dataset=DatasetConfig(
                repo_id="test/dataset",
                split="train",
                cache_dir=str(self.cache_dir),
            ),
            processing=ProcessingConfig(
                shard_size_mb=1,  # Small shard size for testing
                shuffle_buffer_size=10,
                text_column="text",
                seed=42,
                num_proc=1,
            ),
            output=OutputConfig(
                dir=str(self.output_dir),
                name="test_data",
            ),
            tokenizer={"_name": "mock_tokenizer"},  # Use a serializable dict
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("optimus_dl.recipe.data_prep.recipe.build")
    @patch("optimus_dl.recipe.data_prep.source.list_repo_files")
    @patch("optimus_dl.recipe.data_prep.source.hf_hub_download")
    def test_basic_processing(self, mock_download, mock_list_files, mock_build):
        # Mock dependencies
        try:
            from optimus_dl.modules.tokenizer import register_tokenizer

            register_tokenizer("mock_tokenizer")(MockTokenizer)
        except AssertionError:
            pass  # Already registered

        mock_tokenizer = MockTokenizer()
        mock_build.return_value = mock_tokenizer
        mock_list_files.return_value = [
            "data/train-00001.jsonl",
            "data/train-00002.jsonl",
        ]

        # Create dummy local files
        file1_name = "train-00001.jsonl"
        file1_path = self.cache_dir / file1_name
        with open(file1_path, "w") as f:
            f.write(json.dumps({"text": "abc"}) + "\n")
            f.write(json.dumps({"text": "def"}) + "\n")

        file2_name = "train-00002.jsonl"
        file2_path = self.cache_dir / file2_name
        with open(file2_path, "w") as f:
            f.write(json.dumps({"text": "ghi"}) + "\n")

        # Mock download to return local paths
        mock_download.side_effect = [str(file1_path), str(file2_path)]

        recipe = DataPrepRecipe(self.config)
        recipe.run()
        # Verify output
        index_path = self.output_dir / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            index = json.load(f)

        assert index["total_tokens"] == 9  # 3+3+3
        assert len(index["files"]) >= 1

        # Check shards
        shard_file = self.output_dir / index["files"][0]["file"]
        lens_file = self.output_dir / index["files"][0]["lens_file"]

        assert shard_file.exists()
        assert lens_file.exists()

        tokens = np.load(shard_file)
        lens = np.load(lens_file)

        # Expected tokens: ord('a').. etc.
        # Since buffer is shuffled, order might vary, but set of tokens should match
        expected_tokens = {ord(c) for c in "abcdefghi"}
        assert set(tokens.tolist()) == expected_tokens
        assert len(tokens) == 9
        assert len(lens) == 3  # 3 documents

    @patch("optimus_dl.recipe.data_prep.recipe.build")
    @patch("optimus_dl.recipe.data_prep.source.list_repo_files")
    @patch("optimus_dl.recipe.data_prep.source.hf_hub_download")
    def test_resumption(self, mock_download, mock_list_files, mock_build):
        # Mock dependencies
        try:
            from optimus_dl.modules.tokenizer import register_tokenizer

            register_tokenizer("mock_tokenizer")(MockTokenizer)
        except AssertionError:
            pass  # Already registered

        mock_tokenizer = MockTokenizer()
        mock_build.return_value = mock_tokenizer
        mock_list_files.return_value = [
            "data/train-f1.jsonl",
            "data/train-f2.jsonl",
            "data/train-f3.jsonl",
        ]

        # Create many small files to force checkpoints
        local_mock_files = []
        for i in range(3):
            file_name = f"train-f{i+1}.jsonl"
            p = self.cache_dir / file_name
            with open(p, "w") as f:
                f.write(json.dumps({"text": "a" * 1000}) + "\n")
            local_mock_files.append(str(p))
        mock_download.side_effect = local_mock_files

        # 1. Run partially (simulate interruption)
        recipe = DataPrepRecipe(self.config)

        # Set shard size such that it can hold one document (2KB) but not two (4KB)
        # 0.0025 MB = 2.5 KB
        recipe.config.processing.shard_size_mb = 0.0025
        recipe.sharder.max_shard_bytes = 0.0025 * 1024 * 1024

        # Mock _save_checkpoint to raise KeyboardInterrupt after the first save
        original_save_checkpoint = recipe.checkpointer.save
        call_count = 0

        def interrupting_save_checkpoint(*args, **kwargs):
            nonlocal call_count
            original_save_checkpoint(*args, **kwargs)
            call_count += 1
            if call_count == 1:  # Interrupt after the first checkpoint is saved
                raise KeyboardInterrupt("Simulated interruption")

        recipe.checkpointer.save = interrupting_save_checkpoint

        try:
            recipe.run()
        except KeyboardInterrupt:
            pass

        # Checkpoint should exist now
        assert (self.output_dir / "checkpoint.pkl").exists()

        # 2. Resume
        mock_download.reset_mock()
        mock_download.side_effect = local_mock_files

        recipe2 = DataPrepRecipe(self.config)
        recipe2.run()

        # Verify completion
        index_path = self.output_dir / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            index = json.load(f)

        # Should have processed all 3 files * 1000 chars = 3000 tokens
        assert index["total_tokens"] == 3000
        # Clean up
        assert not (self.output_dir / "checkpoint.pkl").exists()

    @patch("optimus_dl.recipe.data_prep.source.list_repo_files")
    @patch("optimus_dl.recipe.data_prep.source.hf_hub_download")
    @patch("optimus_dl.recipe.data_prep.processor.multiprocessing.get_context")
    def test_parallel_processing(
        self, mock_get_context, mock_download, mock_list_files
    ):
        # Enable parallel processing
        self.config.processing.num_proc = 2

        # Mock `list_repo_files` to return names matching the filtering logic
        mock_list_files.return_value = ["data/train-f1.jsonl", "data/train-f2.jsonl"]

        # Create dummy local files for mock_download
        local_mock_files = []
        for i in range(2):
            file_name = f"train-f{i+1}.jsonl"
            p = self.cache_dir / file_name
            with open(p, "w") as f:
                f.write(json.dumps({"text": f"doc{i+1}"}) + "\n")
            local_mock_files.append(str(p))

        mock_download.side_effect = local_mock_files

        # Setup pool mock
        pool_instance = MagicMock()
        context_instance = MagicMock()
        mock_get_context.return_value = context_instance
        context_instance.Pool.return_value = pool_instance

        # imap needs to yield results.
        # Result format: list of list of ints (tokens)
        pool_instance.imap.return_value = iter(
            [
                [[ord(c) for c in "doc1"]],  # Tokenized "doc1"
                [[ord(c) for c in "doc2"]],  # Tokenized "doc2"
            ]
        )

        recipe = DataPrepRecipe(self.config)
        recipe.run()

        # Verify pool usage
        mock_get_context.assert_called_once_with("spawn")
        context_instance.Pool.assert_called_once_with(2)
        pool_instance.imap.assert_called_once()

        # Check output
        index_path = self.output_dir / "index.json"
        assert index_path.exists()
        with open(index_path) as f:
            index = json.load(f)

        # "doc1" + "doc2" -> 4+4=8 tokens
        assert index["total_tokens"] == 8
