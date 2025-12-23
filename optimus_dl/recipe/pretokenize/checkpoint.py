"""Manages saving and loading of data preparation checkpoints."""

import logging
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """
    Represents the state to be saved in a checkpoint.
    This provides a clear structure for what is being saved and loaded.
    """

    processor_state: dict[str, Any]
    sharder_state: dict[str, Any]
    rng_state: Any


class CheckpointManager:
    """Handles the loading and saving of checkpoints to ensure atomicity."""

    def __init__(self, output_dir: Path):
        self.checkpoint_path = output_dir / "checkpoint.pkl"
        self.tmp_path = output_dir / "checkpoint.tmp"

    def save(self, state: CheckpointState):
        """Saves the current processing state to disk atomically."""
        with open(self.tmp_path, "wb") as f:
            pickle.dump(state, f)
        shutil.move(self.tmp_path, self.checkpoint_path)
        logger.debug(f"Saved checkpoint to {self.checkpoint_path}")

    def load(self) -> CheckpointState | None:
        """Loads the processing state from disk if a checkpoint exists."""
        if self.checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            try:
                with open(self.checkpoint_path, "rb") as f:
                    state = pickle.load(f)
                if isinstance(state, CheckpointState):
                    return state
                logger.warning("Checkpoint file is invalid or outdated, ignoring.")
                return None
            except Exception as e:
                logger.warning(
                    f"Failed to load checkpoint: {e}. Starting from scratch."
                )
                return None
        return None

    def clean(self):
        """Removes the checkpoint file if it exists."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.debug("Removed checkpoint file.")
