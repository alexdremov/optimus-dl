"""
Handles the tokenization of source files using a high-performance parallel pipeline.
Architecture:
  [Downloader Process] -> (File Paths) -> [Reader Process] -> (Raw Batches) -> [Tokenizer Processes] -> (Token Batches) -> [Main Process]
"""

import heapq
import logging
import multiprocessing
import queue
import random
from collections.abc import Generator
from typing import (
    Any,
    NamedTuple,
)

from huggingface_hub import hf_hub_download

from optimus_dl.core.registry import build

from .config import DataPrepConfig
from .source import FileReader

logger = logging.getLogger(__name__)


# --- Worker Functions ---


class DownloaderMessage(NamedTuple):
    file_idx: int
    file_path: str


def _downloader_worker(
    files: list[str],
    dataset_config: Any,
    output_queue: multiprocessing.Queue,
    yielded_file_idx: int | None,
):
    """
    Worker 1: Pre-loads (downloads) files.
    Ensures files are present in the local HF cache before the Reader needs them.
    """
    logger.setLevel(logging.INFO)
    # We start from start_idx, assuming strict order is required
    try:
        yielded_file_idx = yielded_file_idx or 0
        for i in range(yielded_file_idx, len(files)):
            file_path = files[i]
            # Ensure file is downloaded/cached
            # This is the network-bound part.
            hf_hub_download(
                repo_id=dataset_config.repo_id,
                filename=file_path,
                repo_type="dataset",
                cache_dir=dataset_config.cache_dir,
            )
            output_queue.put(DownloaderMessage(i, file_path))
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Downloader worker failed: {e}")
        output_queue.put(e)
        return
    finally:
        output_queue.put(None)  # Sentinel


def _shuffled_reader(
    reader: Generator[str, None, None], buffer_size: int | None, init_seed: int
):
    buffer_index = 0
    if buffer_size is None:
        yield from reader
    else:
        buffer = []
        for text in reader:
            buffer.append(text)
            if len(buffer) >= buffer_size:
                random.seed(init_seed + buffer_index)
                random.shuffle(buffer)
                yield from buffer
                buffer_index += 1
                buffer = []
        if buffer:
            random.seed(init_seed + buffer_index)
            random.shuffle(buffer)
            yield from buffer
            buffer_index += 1


class ReaderMessage(NamedTuple):
    file_idx: int
    doc_idx: int
    sort_doc_id: int
    text: str


def _reader_worker(
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    processing_config: Any,
    dataset_config: Any,
    yielded_doc_idx: int | None,
    shuffle_buffer_size: int | None,
    num_workers: int,
    init_seed: int,
):
    """
    Worker 2: Reads files and produces raw text batches.
    """
    logger.setLevel(logging.INFO)
    if yielded_doc_idx is not None:
        logger.info(f"Reader: Skipping first {yielded_doc_idx} docs (idx)")

    try:
        reader = FileReader(processing_config, dataset_config)
        sort_doc_id = 0

        while True:
            item = input_queue.get()
            if item is None:
                break

            if isinstance(item, Exception | KeyboardInterrupt):
                raise RuntimeError("Worker failed") from item

            doc_idx = 0
            file_path = item.file_path
            file_idx = item.file_idx
            for text in _shuffled_reader(
                reader.read_texts(file_path),
                buffer_size=shuffle_buffer_size,
                init_seed=file_idx * 100 + init_seed,
            ):
                if yielded_doc_idx is not None and doc_idx <= yielded_doc_idx:
                    doc_idx += 1
                    continue
                else:
                    yielded_doc_idx = None

                output_queue.put(
                    ReaderMessage(
                        file_idx=file_idx,
                        doc_idx=doc_idx,
                        sort_doc_id=sort_doc_id,
                        text=text,
                    )
                )
                doc_idx += 1
                sort_doc_id += 1

    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Reader worker failed: {e}")
        output_queue.put(e)
        return
    finally:
        for _ in range(num_workers):
            output_queue.put(None)  # Sentinel


class TokenizedMessage(NamedTuple):
    file_idx: int
    doc_idx: int
    sort_doc_id: int
    tokens: list[int]


def _tokenizer_worker(
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    tokenizer: Any,
):
    """
    Worker 3 (Pool): Tokenizes text batches.
    """
    logger.setLevel(logging.INFO)
    try:
        while True:
            item = input_queue.get()
            if item is None:
                break

            if isinstance(item, Exception | KeyboardInterrupt):
                raise RuntimeError("Worker failed") from item

            text = item.text
            tokens = tokenizer.encode(text)

            output_queue.put(
                TokenizedMessage(
                    file_idx=item.file_idx,
                    doc_idx=item.doc_idx,
                    sort_doc_id=item.sort_doc_id,
                    tokens=tokens,
                )
            )
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Tokenizer worker failed: {e}")
        output_queue.put(e)
        return
    finally:
        output_queue.put(None)


class TokenProcessor:
    """A resumable, parallel tokenization pipeline.

    Manages a multi-stage pipeline:
    1. Downloader (Pre-fetch)
    2. Reader (Disk I/O)
    3. Tokenizers (CPU)

    Outputs are re-ordered to ensure determinism for resumption.
    """

    def __init__(self, files: list[str], config: DataPrepConfig):
        self.files = files
        self.config = config
        self.tokenizer = build("tokenizer", config.tokenizer)
        self.dataset_config = config.dataset
        self.processing_config = config.processing
        self.num_proc = self.processing_config.num_proc
        self.shuffle_buffer_size = self.processing_config.shuffle_buffer_size
        self.seed = self.processing_config.seed

        # Pipeline internals
        self.ctx = multiprocessing.get_context("spawn")
        self.processes = []
        self.queues = {}

        self.yielded_doc_idx = None
        self.yielded_file_idx = None
        self.total_docs_yielded = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "yielded_doc_idx": self.yielded_doc_idx,
            "yielded_file_idx": self.yielded_file_idx,
            "total_docs_yielded": self.total_docs_yielded,
        }

    def load_state(self, state: dict[str, Any]):
        self.yielded_doc_idx = state.get("yielded_doc_idx", 0)
        self.yielded_file_idx = state.get("yielded_file_idx", 0)
        self.total_docs_yielded = state.get("total_docs_yielded", 0)

        logger.info(
            f"Resuming from checkpoint. {self.yielded_doc_idx = } {self.yielded_file_idx = } {self.total_docs_yielded = }"
        )

    def __iter__(self) -> Generator[list[int], None, None]:
        # Clean up any previous run
        self._stop_pipeline()

        # Generator that yields ordered batches from the parallel pipeline
        if self.num_proc == 0:
            pipeline_gen = self._run_sequential()
        else:
            pipeline_gen = self._start_pipeline_generator()

        try:
            for result in pipeline_gen:
                self.total_docs_yielded += 1
                self.yielded_doc_idx = result.doc_idx
                self.yielded_file_idx = result.file_idx
                yield result.tokens
        finally:
            self._stop_pipeline()

    def _start_pipeline_generator(self) -> Generator[TokenizedMessage, None, None]:
        """
        Sets up the multiprocessing pipeline and yields re-ordered batches.
        Updates `self.file_idx` and `self.doc_idx_in_file` as data flows through.
        """
        # 1. Create Queues
        # Limited size to control RAM
        self.queues["files"] = self.ctx.Queue(maxsize=3)
        self.queues["documents"] = self.ctx.Queue(maxsize=1024)
        self.queues["tokens"] = self.ctx.Queue(maxsize=1024)

        # 2. Start Downloader
        p_down = self.ctx.Process(
            target=_downloader_worker,
            args=(
                self.files,
                self.dataset_config,
                self.queues[
                    "files"
                ],  # Start from the beginning of the current block context
                self.yielded_file_idx,
            ),
            name="Downloader",
            daemon=True,
        )
        p_down.start()
        self.processes.append(p_down)

        # 3. Start Reader
        actual_num_tok = max(1, self.num_proc)
        p_read = self.ctx.Process(
            target=_reader_worker,
            args=(
                self.queues["files"],
                self.queues["documents"],
                self.processing_config,
                self.dataset_config,
                self.yielded_doc_idx,  # Skip docs if resuming within a file
                self.shuffle_buffer_size,
                actual_num_tok,
                self.seed,
            ),
            name="Reader",
            daemon=True,
        )
        p_read.start()
        self.processes.append(p_read)

        # 4. Start Tokenizers
        for i in range(actual_num_tok):
            p_tok = self.ctx.Process(
                target=_tokenizer_worker,
                args=(
                    self.queues["documents"],
                    self.queues["tokens"],
                    self.tokenizer,
                ),
                name=f"Tokenizer-{i}",
                daemon=True,
            )
            p_tok.start()
            self.processes.append(p_tok)

        # 5. Re-ordering Loop (Generator)
        return self._consume_and_reorder(actual_num_tok)

    def _run_sequential(self) -> Generator[TokenizedMessage, None, None]:
        """
        Runs the pipeline sequentially in the main process (for testing/debugging).
        Reuses the exact same worker functions but with standard Queues and direct calls.
        """
        # 1. Create Queues (Standard queue.Queue for sequential execution)
        self.queues["files"] = queue.Queue()
        self.queues["documents"] = queue.Queue()
        self.queues["tokens"] = queue.Queue()

        # 2. Run Downloader
        _downloader_worker(
            self.files,
            self.dataset_config,
            self.queues["files"],
            self.yielded_file_idx,
        )
        logger.info("Downloader completed.")

        # 3. Run Reader
        _reader_worker(
            self.queues["files"],
            self.queues["documents"],
            self.processing_config,
            self.dataset_config,
            self.yielded_doc_idx,
            self.shuffle_buffer_size,
            1,
            self.seed,
        )
        logger.info("Reader completed.")

        # 4. Run Tokenizer (Single worker for sequential)
        _tokenizer_worker(
            self.queues["documents"],
            self.queues["tokens"],
            self.tokenizer,
        )
        logger.info("Tokenizer completed.")

        # 5. Consume Results (1 worker means 1 sentinel)
        return self._consume_and_reorder(num_workers=1)

    def _consume_and_reorder(
        self, num_workers: int
    ) -> Generator[TokenizedMessage, None, None]:
        """
        Consumes from token_batches queue.
        Ensures strict ordering by batch_id: 0, 1, 2...
        """
        next_expected_id = 0
        reorder_heap = []  # Min-heap of (sort_doc_id, data)
        finished_workers = 0

        while True:
            # Check for dead workers
            # (Simplification: we assume they handle their own errors or we catch the None sentinel)

            item = self.queues["tokens"].get()

            if isinstance(item, Exception | KeyboardInterrupt):
                logger.warning(f"Got exception from worker. {item!r}")
                raise item

            if item is None:
                logger.info(f"Worker {finished_workers} completed.")
                finished_workers += 1
                if finished_workers >= num_workers:
                    logger.info("All workers completed.")
                    break
                continue

            heapq.heappush(reorder_heap, (item.sort_doc_id, item))

            while reorder_heap and reorder_heap[0][0] == next_expected_id:
                _, item = heapq.heappop(reorder_heap)
                next_expected_id += 1

                yield item

        logger.info("Pipeline completed.")

    def _stop_pipeline(self):
        """Terminates all workers."""
        logger.info("Stopping pipeline...")
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()
        self.processes = []

        # Clear queues
        for q in self.queues.values():
            if not isinstance(q, queue.Queue):
                q.close()
                q.join_thread()
        self.queues = {}

    @property
    def progress(self) -> int:
        return self.yielded_file_idx or 0
