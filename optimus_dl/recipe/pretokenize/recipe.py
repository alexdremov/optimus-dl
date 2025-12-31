"Main recipe for preparing and tokenizing datasets from the Hugging Face Hub."

import logging
from pathlib import Path
from typing import Any

import numpy as np
import omegaconf
from tqdm import tqdm

from optimus_dl.core.registry import build
from optimus_dl.modules.tokenizer.base import BaseTokenizer

from .checkpoint import (
    CheckpointManager,
    CheckpointState,
)
from .config import DataPrepConfig
from .processor import TokenProcessor
from .sharder import Sharder
from .source import FileFinder

logger = logging.getLogger(__name__)


class DataPrepRecipe:
    """Recipe for preparing and tokenizing datasets.

    Orchestrates the entire ETL pipeline:
    1.  **Extract**: Finds files from a Hugging Face Hub repository using `FileFinder`.
    2.  **Transform**: Tokenizes text documents in parallel using `TokenProcessor`.
    3.  **Load**: Writes tokenized data into sharded numpy files using `Sharder`.

    Handles resumption from interruptions via atomic checkpointing.

    Args:
        config: Data preparation configuration.
    """

    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.output_dir = Path(config.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sharder = Sharder(config.output, config.processing)
        self.checkpointer = CheckpointManager(self.output_dir)

        self._check_tokenizer()

    def _check_tokenizer(self):
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

    def run(self):
        """Executes the data preparation pipeline.

        Finds files, resumes from checkpoint if available, and processes data
        until completion. Finalizes by writing the `index.json`.
        """
        file_finder = FileFinder(self.config.dataset, self.config.processing.seed)
        files = file_finder.get_files()
        if not files:
            logger.error("No files found to process. Aborting.")
            return

        logger.info(f"Found {len(files)} files to process.")
        processor = TokenProcessor(files, self.config)

        # Load checkpoint if one exists
        checkpoint = self.checkpointer.load()
        if checkpoint:
            logger.info("Resuming from a checkpoint.")
            processor.load_state(checkpoint.processor_state)
            self.sharder.load_state(checkpoint.sharder_state)

        # Setup progress bars
        file_pbar = tqdm(
            total=len(files),
            desc="Files",
            unit="file",
            initial=processor.progress,
            position=0,
        )
        token_pbar = tqdm(
            desc="Tokens", unit="tok", initial=self.sharder.total_tokens, position=1
        )

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
                    file_pbar.set_description(
                        f"Files (Saved shard {self.sharder.shard_idx-1})"
                    )
                    logger.debug(f"Shard flushed at file index {processor.progress}.")
                    state = CheckpointState(
                        processor_state=processor.get_state(),
                        sharder_state=self.sharder.get_state(),
                    )
                    self.checkpointer.save(state)

            # Finalize the process
            file_pbar.set_description("Finalizing index...")
            self.sharder.finalize(self._get_final_config())
            self.checkpointer.clean()
            file_pbar.set_description("Processing Complete")

        except KeyboardInterrupt:
            logger.info("Interruption detected. Saving final checkpoint...")
            # Ensure the current state is saved upon interruption
            state = CheckpointState(
                processor_state=processor.get_state(),
                sharder_state=self.sharder.get_state(),
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
                omegaconf.OmegaConf.to_container(self.config.tokenizer, resolve=True)
                if omegaconf.OmegaConf.is_config(self.config.tokenizer)
                else omegaconf.OmegaConf.to_container(
                    omegaconf.OmegaConf.structured(self.config.tokenizer), resolve=True
                )
            ),
        }
