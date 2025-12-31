"""Tests for the CheckpointManager."""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from optimus_dl.recipe.pretokenize.checkpoint import (
    CheckpointManager,
    CheckpointState,
)


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.checkpointer = CheckpointManager(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load(self):
        """Test that a checkpoint can be saved and then loaded correctly."""
        state = CheckpointState(
            processor_state={"file_idx": 5, "buffer": [1, 2]},
            sharder_state={"shard_idx": 1, "total_tokens": 1000},
        )
        self.checkpointer.save(state)

        # Verify the file exists
        self.assertTrue(self.checkpointer.checkpoint_path.exists())

        # Load and verify content
        loaded_state = self.checkpointer.load()
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.processor_state["file_idx"], 5)
        self.assertEqual(loaded_state.sharder_state["shard_idx"], 1)

    def test_load_non_existent(self):
        """Test that loading returns None when no checkpoint exists."""
        loaded_state = self.checkpointer.load()
        self.assertIsNone(loaded_state)

    def test_clean(self):
        """Test that the checkpoint file is removed."""
        # Create a dummy checkpoint file
        self.checkpointer.checkpoint_path.touch()
        self.assertTrue(self.checkpointer.checkpoint_path.exists())

        self.checkpointer.clean()
        self.assertFalse(self.checkpointer.checkpoint_path.exists())

    def test_atomic_save(self):
        """Test the atomicity of the save operation."""
        state = CheckpointState({}, {})
        self.checkpointer.save(state)

        # The temporary file should not exist after a successful save
        self.assertFalse(self.checkpointer.tmp_path.exists())
        self.assertTrue(self.checkpointer.checkpoint_path.exists())

    def test_load_corrupted_file(self):
        """Test that loading a corrupted or invalid pickle file returns None."""
        # Create a file with invalid content
        with open(self.checkpointer.checkpoint_path, "wb") as f:
            f.write(b"this is not a pickle file")

        loaded_state = self.checkpointer.load()
        self.assertIsNone(loaded_state)

    def test_load_outdated_state_format(self):
        """Test that loading an old checkpoint format (not a CheckpointState obj) returns None."""
        # Old format as a simple dict
        old_state = {
            "generator_state": {"file_idx": 1},
            "shard_idx": 0,
        }
        with open(self.checkpointer.checkpoint_path, "wb") as f:
            pickle.dump(old_state, f)

        loaded_state = self.checkpointer.load()
        self.assertIsNone(loaded_state)


if __name__ == "__main__":
    unittest.main()
