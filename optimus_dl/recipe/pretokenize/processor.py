"""
Handles the tokenization of source files, including parallel processing and resumption.
"""

import logging
import multiprocessing
import random
from collections.abc import Generator
from typing import Any

from tqdm import tqdm

from optimus_dl.core.registry import build

from .config import DataPrepConfig
from .source import FileReader

logger = logging.getLogger(__name__)

# Global variables in worker processes
_worker_tokenizer = None


def _init_worker_context(tokenizer_cfg):
    """Initializes the tokenizer and progress bar in the worker process."""
    global _worker_tokenizer

    _worker_tokenizer = build("tokenizer", tokenizer_cfg)


def _tokenize_batch_worker(text_batch) -> list[list[int]]:
    """
    Worker function to tokenize a batch of texts.

    Args:
        args: A tuple containing (text_batch, tokenizer_cfg).
    Returns:
        A list of tokenized documents.
    """
    tokenized_docs = []

    global _worker_tokenizer

    for text in text_batch:
        tokens = _worker_tokenizer.encode(text)
        if tokens:
            tokenized_docs.append(tokens)

    return tokenized_docs


class TokenProcessor:
    """A resumable generator that yields tokenized documents from a list of files.

    Manages a pool of workers to tokenize files in parallel. Features:

    - **Parallelism**: Uses multiprocessing to speed up tokenization.
    - **Buffering**: Accumulates a buffer of documents for local shuffling.
    - **Resumability**: Tracks file progress and buffer state to allow
      checkpointing and resuming after interruptions without saving the full buffer.
    - **Streaming**: Reads files in batches to support large files without OOM.

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
        self.batch_size = 1000  # Number of docs per batch sent to workers

        # State
        self.file_idx = 0
        self.doc_idx_in_file = 0  # Track progress within the current file
        self.total_docs_yielded = 0

        # Resumption state (Block-based)
        self.block_start_file_idx = 0
        self.block_start_doc_idx = 0
        self.block_rng_state = None
        self.docs_yielded_in_block = 0

        # Internal stream state
        self.pool: multiprocessing.Pool | None = None
        self._batch_stream: Generator | None = None

    def get_state(self) -> dict[str, Any]:
        """Returns the current state for checkpointing."""
        return {
            "file_idx": self.file_idx,
            "doc_idx_in_file": self.doc_idx_in_file,
            "block_start_file_idx": self.block_start_file_idx,
            "block_start_doc_idx": self.block_start_doc_idx,
            "block_rng_state": self.block_rng_state,
            "docs_yielded_in_block": self.docs_yielded_in_block,
            "total_docs_yielded": self.total_docs_yielded,
        }

    def load_state(self, state: dict[str, Any]):
        """Restores the state from a checkpoint."""
        self.file_idx = state.get("file_idx", 0)
        self.doc_idx_in_file = state.get("doc_idx_in_file", 0)
        self.block_start_file_idx = state.get("block_start_file_idx", 0)
        self.block_start_doc_idx = state.get("block_start_doc_idx", 0)
        self.block_rng_state = state.get("block_rng_state")
        self.docs_yielded_in_block = state.get("docs_yielded_in_block", 0)
        self.total_docs_yielded = state.get("total_docs_yielded", 0)

        self._batch_stream = None

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
        """Generates, shuffles, and yields a block of documents."""
        if resume:
            logger.info(
                f"Resuming block from file {self.block_start_file_idx} (offset {self.block_start_doc_idx}), "
                f"skipping {self.docs_yielded_in_block} documents."
            )
            # Restore iteration state to the start of the block
            self.file_idx = self.block_start_file_idx
            self.doc_idx_in_file = self.block_start_doc_idx
            self._reset_batch_stream()

            buffer = self._fill_buffer()

            if self.block_rng_state:
                random.setstate(self.block_rng_state)

            random.shuffle(buffer)
            yield from self._yield_from_buffer(buffer, skip=self.docs_yielded_in_block)

            self.docs_yielded_in_block = 0
        else:
            # Record start of a new block
            self.block_start_file_idx = self.file_idx
            self.block_start_doc_idx = self.doc_idx_in_file

            if self._batch_stream is None:
                self._reset_batch_stream()

            buffer = self._fill_buffer()
            if not buffer:
                return

            self.block_rng_state = random.getstate()
            random.shuffle(buffer)

            self.docs_yielded_in_block = 0
            yield from self._yield_from_buffer(buffer)

    def _fill_buffer(self) -> list[list[int]]:
        """Consumes batches from the stream until the buffer is full."""
        buffer = []
        target_size = self.processing_config.shuffle_buffer_size

        with tqdm(
            total=target_size,
            desc="Refilling Buffer",
            unit="doc",
            leave=False,
            disable=True,
        ) as pbar:
            while len(buffer) < target_size:
                try:
                    # Get next batch of tokenized docs
                    batch_docs = next(self._batch_stream)
                    if batch_docs:
                        doc_count = len(batch_docs)
                        buffer.extend(batch_docs)
                        pbar.update(doc_count)
                except StopIteration:
                    break

        return buffer

    def _yield_from_buffer(
        self, buffer: list[list[int]], skip: int = 0
    ) -> Generator[list[int], None, None]:
        """Yields items from the buffer, updating state counters."""
        if skip > 0:
            del buffer[len(buffer) - skip :]

        while buffer:
            self.docs_yielded_in_block += 1
            self.total_docs_yielded += 1
            yield buffer.pop()

    def _generate_batches(self) -> Generator[str]:
        """Reads files and yields batches of raw text for processing."""
        # Start from the current file_idx
        # If we are resuming, we might need to skip docs in the current file
        skip_count = self.doc_idx_in_file

        while self.file_idx < len(self.files):
            file_path = self.files[self.file_idx]
            logger.info(f"Reading file: {file_path}")

            file_reader = FileReader(self.processing_config, self.dataset_config)
            text_iterator = file_reader.read_texts(file_path)

            # Fast-forward if needed
            if skip_count > 0:
                logger.info(f"Skipping {skip_count} docs in {file_path}...")
                for _ in range(skip_count):
                    next(text_iterator)
                skip_count = 0  # Only skip for the first file in the sequence

            batch = []
            for text in text_iterator:
                batch.append(text)
                self.doc_idx_in_file += 1

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

            # File finished
            self.file_idx += 1
            self.doc_idx_in_file = 0

    def _start_pool(self):
        """Initializes the multiprocessing pool if needed."""
        if self.num_proc > 1 and self.pool is None:
            ctx = multiprocessing.get_context("spawn")

            self.pool = ctx.Pool(
                self.num_proc,
                initializer=_init_worker_context,
                initargs=(self.tokenizer_config,),
            )
            logger.info(f"Initialized processing pool with {self.num_proc} workers.")

    def _stop_pool(self):
        """Closes the multiprocessing pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
            self._batch_stream = None

    def _reset_batch_stream(self):
        """Creates a new iterator over batches."""
        if self.pool:
            _init_worker_context(None)
            self._batch_stream = self.pool.imap(
                _tokenize_batch_worker, self._generate_batches()
            )
        else:
            _init_worker_context(self.tokenizer_config)
            self._batch_stream = map(_tokenize_batch_worker, self._generate_batches())

    @property
    def progress(self) -> int:
        """Returns the number of files that have been submitted for processing."""
        return self.file_idx
