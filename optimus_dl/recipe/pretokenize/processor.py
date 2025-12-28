"""
Handles the tokenization of source files, including parallel processing and resumption.
"""

import logging
import multiprocessing
import random
import time
from collections.abc import Generator
from typing import Any

from tqdm import tqdm

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
    logger.info(f"Worker: Starting processing of {file_path}")
    start_time = time.time()
    
    try:
        tokenizer = build("tokenizer", tokenizer_cfg)
        file_reader = FileReader(proc_cfg, dataset_cfg)

        doc_count = 0
        for text in file_reader.read_texts(file_path):
            tokens = tokenizer.encode(text)
            if tokens:
                tokenized_docs.append(tokens)
                doc_count += 1
                if doc_count % 1000 == 0:
                     logger.debug(f"Worker: Processed {doc_count} docs in {file_path}")

        elapsed = time.time() - start_time
        logger.info(f"Worker: Finished {file_path}. Extracted {len(tokenized_docs)} docs in {elapsed:.2f}s.")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)

    return tokenized_docs


class TokenProcessor:
    """A resumable generator that yields tokenized documents from a list of files.

    Manages a pool of workers to tokenize files in parallel. Features:

    - **Parallelism**: Uses multiprocessing to speed up tokenization.
    - **Buffering**: Accumulates a buffer of documents for local shuffling.
    - **Resumability**: Tracks file progress and buffer state to allow
      checkpointing and resuming after interruptions without saving the full buffer.

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
        self.total_docs_yielded = 0

        # Resumption state (Block-based)
        self.block_start_file_idx = 0
        self.block_rng_state = None
        self.docs_yielded_in_block = 0

        # Internal stream state
        self.pool: multiprocessing.Pool | None = None
        self._file_stream: Generator | None = None

    def get_state(self) -> dict[str, Any]:
        """Returns the current state for checkpointing.
        
        We save enough state to recompute the current block upon resumption.
        """
        return {
            "file_idx": self.file_idx,
            "block_start_file_idx": self.block_start_file_idx,
            "block_rng_state": self.block_rng_state,
            "docs_yielded_in_block": self.docs_yielded_in_block,
            "total_docs_yielded": self.total_docs_yielded,
        }

    def load_state(self, state: dict[str, Any]):
        """Restores the state from a checkpoint."""
        self.file_idx = state.get("file_idx", 0)
        self.block_start_file_idx = state.get("block_start_file_idx", 0)
        self.block_rng_state = state.get("block_rng_state")
        self.docs_yielded_in_block = state.get("docs_yielded_in_block", 0)
        self.total_docs_yielded = state.get("total_docs_yielded", 0)
        
        self._file_stream = None  # Force re-initialization

    def __iter__(self) -> Generator[list[int], None, None]:
        """Yields tokenized documents, handling buffering, shuffling, and resumption."""
        self._start_pool()
        try:
            # 1. Handle Resumption if needed (partial block)
            if self.docs_yielded_in_block > 0:
                yield from self._process_block(resume=True)

            # 2. Main Processing Loop (new blocks)
            while self.file_idx < len(self.files):
                yield from self._process_block(resume=False)

        finally:
            self._stop_pool()

    def _process_block(self, resume: bool) -> Generator[list[int], None, None]:
        """Generates, shuffles, and yields a block of documents.

        Args:
            resume: If True, reconstructs the previous block and fast-forwards.
        """
        if resume:
            logger.info(
                f"Resuming block from file_idx {self.block_start_file_idx}, "
                f"skipping {self.docs_yielded_in_block} documents."
            )
            # Rewind file index to the start of the block
            self.file_idx = self.block_start_file_idx
            self._reset_file_stream()
            
            # Re-fetch buffer
            buffer = self._fill_buffer()
            
            # Restore RNG state
            if self.block_rng_state:
                random.setstate(self.block_rng_state)
            
            # Shuffle and skip
            random.shuffle(buffer)
            yield from self._yield_from_buffer(buffer, skip=self.docs_yielded_in_block)
            
            # Reset resumption flag for subsequent blocks
            self.docs_yielded_in_block = 0
        else:
            # Record start of a new block
            self.block_start_file_idx = self.file_idx
            if self._file_stream is None:
                self._reset_file_stream()

            buffer = self._fill_buffer()
            if not buffer:
                return

            # Save RNG state for this block
            self.block_rng_state = random.getstate()
            random.shuffle(buffer)
            
            self.docs_yielded_in_block = 0
            yield from self._yield_from_buffer(buffer)

    def _fill_buffer(self) -> list[list[int]]:
        """Consumes files from the stream until the buffer is full."""
        buffer = []
        target_size = self.processing_config.shuffle_buffer_size
        
        with tqdm(total=target_size, desc="Refilling Buffer", unit="doc", leave=False) as pbar:
            while len(buffer) < target_size and self.file_idx < len(self.files):
                try:
                    # Get documents from the next file in the stream
                    file_docs = next(self._file_stream)
                    self.file_idx += 1
                    if file_docs:
                        doc_count = len(file_docs)
                        buffer.extend(file_docs)
                        pbar.update(doc_count)
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Error reading file stream: {e}")
                    self.file_idx += 1  # Skip file on error
        
        return buffer

    def _yield_from_buffer(self, buffer: list[list[int]], skip: int = 0) -> Generator[list[int], None, None]:
        """Yields items from the buffer, updating state counters.

        Args:
            buffer: List of documents.
            skip: Number of documents to skip (already yielded).
        """
        # Fast-forward: remove 'skip' items from the end (since pop() takes from end)
        if skip > 0:
            del buffer[len(buffer) - skip :]
        
        while buffer:
            self.docs_yielded_in_block += 1
            self.total_docs_yielded += 1
            yield buffer.pop()

    def _start_pool(self):
        """Initializes the multiprocessing pool if needed."""
        if self.num_proc > 1 and self.pool is None:
            ctx = multiprocessing.get_context("spawn")
            self.pool = ctx.Pool(self.num_proc)
            logger.info(f"Initialized processing pool with {self.num_proc} workers.")

    def _stop_pool(self):
        """Closes the multiprocessing pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
            self._file_stream = None

    def _reset_file_stream(self):
        """Creates a new iterator over the files starting from self.file_idx."""
        files_to_process = self.files[self.file_idx :]
        if not files_to_process:
            self._file_stream = iter([])
            return

        args_gen = (
            (f, self.tokenizer_config, self.dataset_config, self.processing_config)
            for f in files_to_process
        )

        if self.pool:
            self._file_stream = self.pool.imap(_tokenize_file_worker, args_gen)
        else:
            self._file_stream = map(_tokenize_file_worker, args_gen)

    @property
    def progress(self) -> int:
        """Returns the number of files that have been submitted for processing."""
        return self.file_idx
