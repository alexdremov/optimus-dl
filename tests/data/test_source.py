"""Tests for the data source components: FileFinder and FileReader."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from optimus_dl.recipe.data_prep.config import DatasetConfig, ProcessingConfig
from optimus_dl.recipe.data_prep.source import FileFinder, FileReader


class TestFileFinder(unittest.TestCase):
    def setUp(self):
        self.all_files = [
            "data/train/0001.jsonl",
            "data/train/0002.parquet",
            "data/validation/0001.jsonl",
            "README.md",
            "config.json",
        ]

    @patch("optimus_dl.recipe.data_prep.source.list_repo_files")
    def test_get_all_files(self, mock_list_repo_files):
        """Test that all supported file types are returned when split is 'all'."""
        mock_list_repo_files.return_value = self.all_files
        config = DatasetConfig(repo_id="test/repo", split="all")
        finder = FileFinder(config, seed=42)
        files = finder.get_files()
        self.assertEqual(len(files), 3)
        self.assertIn("data/train/0001.jsonl", files)
        self.assertIn("data/validation/0001.jsonl", files)

    @patch("optimus_dl.recipe.data_prep.source.list_repo_files")
    def test_filter_by_split(self, mock_list_repo_files):
        """Test filtering files by a specific split name."""
        mock_list_repo_files.return_value = self.all_files
        config = DatasetConfig(repo_id="test/repo", split="validation")
        finder = FileFinder(config, seed=42)
        files = finder.get_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], "data/validation/0001.jsonl")

    @patch("optimus_dl.recipe.data_prep.source.list_repo_files")
    def test_filter_by_pattern(self, mock_list_repo_files):
        """Test filtering files using a glob pattern."""
        mock_list_repo_files.return_value = self.all_files
        config = DatasetConfig(
            repo_id="test/repo", split="all", file_pattern="**/*.parquet"
        )
        finder = FileFinder(config, seed=42)
        files = finder.get_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], "data/train/0002.parquet")

    @patch("optimus_dl.recipe.data_prep.source.list_repo_files")
    def test_shuffling_is_deterministic(self, mock_list_repo_files):
        """Test that shuffling with the same seed produces the same order."""
        mock_list_repo_files.return_value = self.all_files
        config = DatasetConfig(repo_id="test/repo", split="all")

        finder1 = FileFinder(config, seed=42)
        files1 = finder1.get_files()

        finder2 = FileFinder(config, seed=42)
        files2 = finder2.get_files()

        finder3 = FileFinder(config, seed=123)
        files3 = finder3.get_files()

        self.assertEqual(files1, files2)
        self.assertNotEqual(files1, files3)


class TestFileReader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_config = DatasetConfig(
            repo_id="test/repo", cache_dir=self.temp_dir
        )
        self.proc_config = ProcessingConfig(text_column="text")
        self.reader = FileReader(self.proc_config, self.dataset_config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("optimus_dl.recipe.data_prep.source.hf_hub_download")
    def test_read_jsonl(self, mock_download):
        """Test reading texts from a JSONL file."""
        # Create a dummy jsonl file
        jsonl_path = Path(self.temp_dir) / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "Hello world"}) + "\n")
            f.write(json.dumps({"text": "This is a test."}) + "\n")
            f.write(json.dumps({"other_col": "ignore me"}) + "\n")

        mock_download.return_value = str(jsonl_path)

        texts = list(self.reader.read_texts("test.jsonl"))
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "Hello world")
        self.assertEqual(texts[1], "This is a test.")

    @patch("optimus_dl.recipe.data_prep.source.hf_hub_download")
    def test_read_parquet(self, mock_download):
        """Test reading texts from a Parquet file."""
        # Create a dummy parquet file
        df = pd.DataFrame({"text": ["Alpha", "Bravo"], "id": [1, 2]})
        parquet_path = Path(self.temp_dir) / "test.parquet"
        df.to_parquet(parquet_path)

        mock_download.return_value = str(parquet_path)

        texts = list(self.reader.read_texts("test.parquet"))
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "Alpha")
        self.assertEqual(texts[1], "Bravo")

    @patch("optimus_dl.recipe.data_prep.source.hf_hub_download")
    def test_download_failure(self, mock_download):
        """Test that the reader handles download errors gracefully."""
        mock_download.side_effect = Exception("Download failed!")
        texts = list(self.reader.read_texts("non_existent.jsonl"))
        self.assertEqual(len(texts), 0)


if __name__ == "__main__":
    unittest.main()
