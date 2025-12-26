"""
Handles the tokenization of source files, including parallel processing and resumption.
"""

import logging
import multiprocessing
import random
from collections.abc import Generator
from typing import Any

from optimus_dl.core.registry import build

from .config import DataPrepConfig
from .source import FileReader

logger = logging.getLogger(__name__)


def _tokenize_file_worker(args: tuple) -> list[list[int]]:
    """
    Worker function to be executed in a separate process.
    It reads texts from a file and tokenizes them.

    Args:
        args: A tuple containing (file_path, tokenizer_cfg, dataset_cfg, proc_cfg).
    Returns:
        A list of tokenized documents.
    """
    file_path, tokenizer_cfg, dataset_cfg, proc_cfg = args
    tokenized_docs = []
    try:
        tokenizer = build("tokenizer", tokenizer_cfg)
        file_reader = FileReader(proc_cfg, dataset_cfg)

        for text in file_reader.read_texts(file_path):
            tokens = tokenizer.encode(text)
            if tokens:
                tokenized_docs.append(tokens)

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)

    return tokenized_docs


class TokenProcessor:
    """A resumable generator that yields tokenized documents from a list of files.

    Manages a pool of workers to tokenize files in parallel. Features:

    - **Parallelism**: Uses multiprocessing to speed up tokenization.
    - **Buffering**: Accumulates a buffer of documents for local shuffling.
    - **Resumability**: Tracks file progress and buffer state to allow
      checkpointing and resuming after interruptions.

    Args:
        files: List of file paths to process.
        config: Data preparation configuration.
    """

    def __init__(self, files: list[str], config: DataPrepConfig):
        self.files = files
        self.tokenizer_config = config.tokenizer
        self.dataset_config = config.dataset
        self.processing_config = config.processing
        self.num_proc = self.processing_config.num_proc

        # State
        self.file_idx = 0
        self.buffer: list[list[int]] = []
        self.rng_state = random.getstate()

        # Multiprocessing
        self.pool: multiprocessing.Pool | None = None
        self._iterator: Generator | None = None

    def get_state(self) -> dict[str, Any]:
        """Returns the current state for checkpointing."""
        return {
            "file_idx": self.file_idx,
            "buffer": self.buffer,
            "rng_state": random.getstate(),
        }

    def load_state(self, state: dict[str, Any]):
        """Restores the state from a checkpoint."""
        self.file_idx = state.get("file_idx", 0)
        self.buffer = state.get("buffer", [])
        rng_state = state.get("rng_state")
        if rng_state:
            random.setstate(rng_state)
        self._iterator = None  # Force re-initialization of the iterator

    def _init_iterator(self):
        """Initializes the file iterator (parallel or sequential)."""
        files_to_process = self.files[self.file_idx :]
        if not files_to_process:
            self._iterator = iter([])
            return

        args_gen = (
            (f, self.tokenizer_config, self.dataset_config, self.processing_config)
            for f in files_to_process
        )

        if self.num_proc > 1:
            if self.pool is None:
                ctx = multiprocessing.get_context("spawn")
                self.pool = ctx.Pool(self.num_proc)
                logger.info(
                    f"Initialized processing pool with {self.num_proc} workers."
                )
            self._iterator = self.pool.imap(_tokenize_file_worker, args_gen)
        else:
            self._iterator = map(_tokenize_file_worker, args_gen)

    def _fill_buffer(self):
        """Refills the internal token buffer by consuming from the iterator."""
        if self._iterator is None:
            self._init_iterator()

        target_size = self.processing_config.shuffle_buffer_size
        while len(self.buffer) < target_size:
            try:
                # The iterator yields a list of documents from one file
                file_docs = next(self._iterator)
                if file_docs:
                    self.buffer.extend(file_docs)
                self.file_idx += 1
            except StopIteration:
                # Iterator is exhausted
                break
            except Exception as e:
                logger.error(f"Error during file processing iteration: {e}")
                self.file_idx += 1  # Skip the problematic file

    def __iter__(self) -> Generator[list[int], None, None]:
        """Yields tokenized documents, handling buffering and shuffling."""
        try:
            # Yield documents already in the buffer from a previous run
            while self.buffer:
                yield self.buffer.pop(0)

            while self.file_idx < len(self.files):
                self._fill_buffer()
                if not self.buffer:
                    break

                random.shuffle(self.buffer)

                # Yield a portion of the buffer to avoid holding too much in memory
                # while still allowing for good shuffling.
                yield_count = (
                    len(self.buffer)
                    if self.file_idx >= len(self.files)
                    else max(1, len(self.buffer) // 2)
                )

                for _ in range(yield_count):
                    if self.buffer:
                        yield self.buffer.pop()
                    else:
                        break  # Buffer might be empty if yield_count was > len
        finally:
            if self.pool:
                self.pool.close()
                self.pool.join()
                self.pool = None

    @property
    def progress(self) -> int:
        """Returns the number of files that have been submitted for processing."""
        return self.file_idx
