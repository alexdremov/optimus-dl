"""Tests for the Sharder."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optimus_dl.recipe.pretokenize.config import OutputConfig, ProcessingConfig
from optimus_dl.recipe.pretokenize.sharder import Sharder


class TestSharder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

        self.output_config = OutputConfig(dir=str(self.output_dir), name="test_shard")
        # Use a tiny shard size to make flushing easy to trigger
        self.proc_config = ProcessingConfig(shard_size_mb=0.001)  # 1KB

        self.sharder = Sharder(self.output_config, self.proc_config)
        # Set dtype for predictable size calculations
        self.sharder.dtype = np.uint16

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_add_single_doc(self):
        """Test adding a single document to the sharder."""
        doc = [1, 2, 3, 4, 5]
        self.sharder.add(doc)

        self.assertEqual(len(self.sharder.current_shard_tokens), 5)
        self.assertEqual(self.sharder.current_shard_doc_lens, [5])
        self.assertEqual(
            self.sharder.current_shard_size_bytes, 10
        )  # 5 tokens * 2 bytes/token

    def test_flush_shard(self):
        """Test that flushing writes the correct files to disk."""
        self.sharder.add([1, 2, 3])
        self.sharder.add([4, 5])

        self.sharder.flush()

        # Verify files were created
        shard_path = self.output_dir / "test_shard_0000000000.npy"
        lens_path = self.output_dir / "test_shard_0000000000_lens.npy"
        self.assertTrue(shard_path.exists())
        self.assertTrue(lens_path.exists())

        # Verify content
        tokens = np.load(shard_path)
        lens = np.load(lens_path)
        np.testing.assert_array_equal(tokens, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(lens, [3, 2])

        # Verify state was updated
        self.assertEqual(self.sharder.shard_idx, 1)
        self.assertEqual(len(self.sharder.file_metadata), 1)
        self.assertEqual(self.sharder.total_tokens, 5)
        self.assertEqual(len(self.sharder.current_shard_tokens), 0)

    def test_shard_flush_on_add(self):
        """Test that `add` automatically flushes when shard size is exceeded."""
        # Each token is 2 bytes. Max size is 1000 bytes. 500 tokens = 1000 bytes.
        doc1 = list(range(400))  # 800 bytes
        doc2 = list(range(200))  # 400 bytes. This should trigger a flush.

        flushed1 = self.sharder.add(doc1)
        self.assertFalse(flushed1)
        self.assertEqual(self.sharder.shard_idx, 0)

        flushed2 = self.sharder.add(doc2)
        self.assertTrue(flushed2)
        self.assertEqual(self.sharder.shard_idx, 1)

        # Check that the first shard was written
        shard0_path = self.output_dir / "test_shard_0000000000.npy"
        self.assertTrue(shard0_path.exists())
        self.assertEqual(len(np.load(shard0_path)), 400)

        # Check that the second doc is now in the *new* current shard
        self.assertEqual(len(self.sharder.current_shard_tokens), 200)

    def test_finalize(self):
        """Test that finalize flushes remaining data and writes the index."""
        self.sharder.add([1, 2, 3])

        final_config = {"dataset": "test", "split": "train"}
        self.sharder.finalize(final_config)

        # Check that the last shard was written
        shard0_path = self.output_dir / "test_shard_0000000000.npy"
        self.assertTrue(shard0_path.exists())

        # Check that index.json was written
        index_path = self.output_dir / "index.json"
        self.assertTrue(index_path.exists())

        with open(index_path) as f:
            index_data = json.load(f)

        self.assertEqual(index_data["total_tokens"], 3)
        self.assertEqual(len(index_data["files"]), 1)
        self.assertEqual(index_data["config"]["dataset"], "test")

    def test_state_management(self):
        """Test saving and loading the sharder's state."""
        self.sharder.add([1, 2])
        self.sharder.flush()
        self.sharder.add([3, 4, 5])

        state = self.sharder.get_state()

        # Create a new sharder and load the state
        new_sharder = Sharder(self.output_config, self.proc_config)
        new_sharder.load_state(state)

        self.assertEqual(new_sharder.shard_idx, 1)
        self.assertEqual(new_sharder.total_tokens, 2)
        self.assertEqual(len(new_sharder.file_metadata), 1)
        self.assertEqual(new_sharder.current_shard_tokens, [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
