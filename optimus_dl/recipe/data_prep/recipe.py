"Main recipe for preparing and tokenizing datasets from the Hugging Face Hub."

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import omegaconf
from tqdm import tqdm

from optimus_dl.core.registry import build
from optimus_dl.modules.tokenizer.base import BaseTokenizer

from .checkpoint import CheckpointManager, CheckpointState
from .config import DataPrepConfig
from .processor import TokenProcessor
from .sharder import Sharder
from .source import FileFinder

logger = logging.getLogger(__name__)


class DataPrepRecipe:
    """
    Orchestrates the data preparation process, including finding files,
    tokenizing them in parallel, and writing them to numpy shards.
    Supports resumption from checkpoints.
    """

    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.output_dir = Path(config.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_rng()
        self.sharder = Sharder(config.output, config.processing)
        self.tokenizer = self._setup_tokenizer()
        self.checkpointer = CheckpointManager(self.output_dir)

    def _setup_rng(self):
        """Initializes random number generators for reproducibility."""
        random.seed(self.config.processing.seed)
        np.random.seed(self.config.processing.seed)

    def _setup_tokenizer(self) -> BaseTokenizer:
        """Builds the tokenizer and validates its vocab size against the chosen dtype."""
        tokenizer = build("tokenizer", self.config.tokenizer)
        assert isinstance(tokenizer, BaseTokenizer)

        # Validate that the tokenizer vocab size fits within the chosen dtype
        max_val = np.iinfo(self.sharder.dtype).max
        if tokenizer.vocab_size > max_val:
            raise ValueError(
                f"Tokenizer vocab size ({tokenizer.vocab_size}) exceeds the maximum value "
                f"for the chosen dtype '{self.sharder.dtype}' ({max_val}). "
                "Please use a larger dtype (e.g., uint32)."
            )
        return tokenizer

    def run(self):
        """Executes the entire data preparation pipeline."""
        file_finder = FileFinder(self.config.dataset, self.config.processing.seed)
        files = file_finder.get_files()
        if not files:
            logger.error("No files found to process. Aborting.")
            return

        logger.info(f"Found {len(files)} files to process: {files}")
        processor = TokenProcessor(files, self.config)

        # Load checkpoint if one exists
        checkpoint = self.checkpointer.load()
        if checkpoint:
            logger.info("Resuming from a checkpoint.")
            processor.load_state(checkpoint.processor_state)
            self.sharder.load_state(checkpoint.sharder_state)
            # Restore python's random state for the main thread
            random.setstate(checkpoint.rng_state)

        # Setup progress bars
        file_pbar = tqdm(
            total=len(files), desc="Files", unit="file", initial=processor.progress
        )
        token_pbar = tqdm(desc="Tokens", unit="tok", initial=self.sharder.total_tokens)

        last_file_progress = processor.progress

        try:
            for doc_tokens in processor:
                # Update file progress bar
                new_file_progress = processor.progress
                if new_file_progress > last_file_progress:
                    file_pbar.update(new_file_progress - last_file_progress)
                    last_file_progress = new_file_progress

                initial_total_tokens = self.sharder.total_tokens

                # Add document to sharder and check if a flush occurred
                shard_was_flushed = self.sharder.add(doc_tokens)

                # Update token progress bar
                token_pbar.update(self.sharder.total_tokens - initial_total_tokens)

                if shard_was_flushed:
                    # A shard was just written, which is a good time to save a checkpoint
                    logger.debug(f"Shard flushed at file index {processor.progress}.")
                    state = CheckpointState(
                        processor_state=processor.get_state(),
                        sharder_state=self.sharder.get_state(),
                        rng_state=random.getstate(),
                    )
                    self.checkpointer.save(state)

            # Finalize the process
            self.sharder.finalize(self._get_final_config())
            self.checkpointer.clean()

        except KeyboardInterrupt:
            logger.info("\nInterruption detected. Saving final checkpoint...")
            # Ensure the current state is saved upon interruption
            state = CheckpointState(
                processor_state=processor.get_state(),
                sharder_state=self.sharder.get_state(),
                rng_state=random.getstate(),
            )
            self.checkpointer.save(state)
            logger.info("Checkpoint saved. To resume, run the script again.")
        finally:
            file_pbar.close()
            token_pbar.close()

    def _get_final_config(self) -> dict[str, Any]:
        """Constructs the configuration to be saved in the final index.json."""
        return {
            "dataset": self.config.dataset.repo_id,
            "split": self.config.dataset.split,
            "dtype": self.config.processing.dtype,
            "tokenizer": (
                omegaconf.OmegaConf.to_container(self.tokenizer.config, resolve=True)
                if omegaconf.OmegaConf.is_config(self.tokenizer.config)
                else omegaconf.OmegaConf.to_container(
                    omegaconf.OmegaConf.structured(self.tokenizer.config), resolve=True
                )
            ),
        }
