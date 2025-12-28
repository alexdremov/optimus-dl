"""Tests for the TokenProcessor."""

import random
import shutil
import tempfile
import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

from optimus_dl.modules.tokenizer.base import BaseTokenizer
from optimus_dl.recipe.pretokenize.config import DataPrepConfig
from optimus_dl.recipe.pretokenize.processor import TokenProcessor


class MockTokenizer(BaseTokenizer):
    def __init__(self, config=None):
        self.config = config

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(map(chr, ids))

    @property
    def vocab_size(self) -> int:
        return 256


class TestTokenProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataPrepConfig(
            tokenizer={"_name": "mock_tokenizer"},
        )
        self.files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("optimus_dl.recipe.pretokenize.processor.FileReader")
    @patch("optimus_dl.recipe.pretokenize.processor.build")
    def test_single_process_iteration(self, mock_build, mock_file_reader_cls):
        """Test the processor in single-process mode."""
        self.config.processing.num_proc = 1

        # Mock tokenizer build
        mock_build.return_value = MockTokenizer(config=self.config.tokenizer)

        # Mock file reader to yield predictable text
        mock_file_reader_instance = mock_file_reader_cls.return_value
        mock_file_reader_instance.read_texts.side_effect = [
            ["doc1", "doc2"],
            ["doc3"],
            ["doc4", "doc5", "doc6"],
        ]

        processor = TokenProcessor(self.files, self.config)
        all_tokens = list(processor)

        self.assertEqual(len(all_tokens), 6)
        self.assertEqual(processor.progress, 3)

    @patch("optimus_dl.recipe.pretokenize.processor.multiprocessing.get_context")
    @patch("optimus_dl.recipe.pretokenize.processor._tokenize_batch_worker")
    def test_multi_process_iteration(self, mock_worker, mock_get_context):
        """Test that the processor correctly uses a multiprocessing pool."""
        self.config.processing.num_proc = 2

        mock_pool_instance = MagicMock()
        mock_context_instance = MagicMock()
        mock_get_context.return_value = mock_context_instance
        mock_context_instance.Pool.return_value = mock_pool_instance

        processor = TokenProcessor(self.files, self.config)

        # Mock the results from the worker using a generator that also updates state
        def mock_results_generator():
            yield [[1, 2]]  # Tokens from file 1
            yield [[3]]  # Tokens from file 2
            yield [[4, 5]]  # Tokens from file 3
            # Simulate the side effect of _generate_batches completing files
            processor.file_idx = 3

        mock_pool_instance.imap.return_value = mock_results_generator()

        all_tokens = list(processor)

        mock_get_context.assert_called_once_with("spawn")
        # Check that initializer was called
        mock_context_instance.Pool.assert_called_once()
        mock_pool_instance.imap.assert_called_once()

        self.assertEqual(len(all_tokens), 3)
        self.assertEqual(processor.progress, 3)

    def test_state_management(self):
        """Test saving and loading of the processor's state."""
        processor = TokenProcessor(self.files, self.config)
        processor.file_idx = 2
        processor.docs_yielded_in_block = 3
        processor.total_docs_yielded = 10

        state = processor.get_state()
        self.assertNotIn("buffer", state)

        new_processor = TokenProcessor(self.files, self.config)
        self.assertEqual(new_processor.file_idx, 0)

        new_processor.load_state(state)

        self.assertEqual(new_processor.file_idx, 2)
        self.assertEqual(new_processor.docs_yielded_in_block, 3)
        self.assertEqual(new_processor.total_docs_yielded, 10)

    @patch("optimus_dl.recipe.pretokenize.processor.FileReader")
    @patch("optimus_dl.recipe.pretokenize.processor.build")
    def test_resumption_recomputation(self, mock_build, mock_file_reader_cls):
        """Test that the processor correctly recomputes the buffer upon resumption."""
        self.config.processing.num_proc = 1
        self.config.processing.shuffle_buffer_size = 4  # Small buffer for testing

        # Mock tokenizer build
        mock_build.return_value = MockTokenizer(config=self.config.tokenizer)

        # Mock file reader to yield predictable text
        # Each file has 2 documents.
        file_data_map = {
            "file1.jsonl": ["doc1", "doc2"],
            "file2.jsonl": ["doc3", "doc4"],
            "file3.jsonl": ["doc5", "doc6"],
        }

        def mock_read_texts(file_path):
            return file_data_map[file_path]

        mock_file_reader_instance = mock_file_reader_cls.return_value
        mock_file_reader_instance.read_texts.side_effect = mock_read_texts

        # Use a fixed seed for predictable shuffling
        random.seed(42)
        processor = TokenProcessor(self.files, self.config)

        iterator = iter(processor)
        # Yield first 2 docs from the first block of 4
        # (The block will be from file1 and file2)
        yielded_first = [next(iterator) for _ in range(2)]

        # Save state
        state = processor.get_state()
        self.assertNotIn("buffer", state)
        self.assertEqual(state["docs_yielded_in_block"], 2)
        self.assertEqual(state["block_start_file_idx"], 0)

        # Create a new processor and load state
        new_processor = TokenProcessor(self.files, self.config)
        new_processor.load_state(state)

        # It should yield the remaining 2 docs from the first block, then the next block
        yielded_rest = list(new_processor)

        self.assertEqual(len(yielded_first) + len(yielded_rest), 6)

        # Verify all documents are present (in some order)
        all_docs_yielded = yielded_first + yielded_rest
        all_docs_tokens = [[ord(c) for c in f"doc{i}"] for i in range(1, 7)]

        for tokens in all_docs_tokens:
            self.assertIn(tokens, all_docs_yielded)

        # Also check that we didn't get duplicates (unless there were duplicates in source)
        self.assertEqual(len(all_docs_yielded), 6)
        unique_yielded = [list(x) for x in {tuple(x) for x in all_docs_yielded}]
        self.assertEqual(len(unique_yielded), 6)

    @patch("optimus_dl.recipe.pretokenize.processor.FileReader")
    @patch("optimus_dl.recipe.pretokenize.processor.build")
    def test_empty_files_handling(self, mock_build, mock_file_reader_cls):
        """Test processing when some files are empty or yield no documents."""
        self.config.processing.num_proc = 1
        mock_build.return_value = MockTokenizer(config=self.config.tokenizer)

        mock_file_reader_instance = mock_file_reader_cls.return_value
        mock_file_reader_instance.read_texts.side_effect = [
            ["doc1"],
            [],  # Empty file
            ["doc2"],
        ]

        processor = TokenProcessor(self.files, self.config)
        all_tokens = list(processor)

        self.assertEqual(len(all_tokens), 2)
        self.assertEqual(processor.progress, 3)

    @patch("optimus_dl.recipe.pretokenize.processor.FileReader")
    @patch("optimus_dl.recipe.pretokenize.processor.build")
    def test_worker_error_resilience(self, mock_build, mock_file_reader_cls):
        """Test that worker errors (e.g. read failure) are skipped gracefully."""
        self.config.processing.num_proc = 1
        mock_build.return_value = MockTokenizer(config=self.config.tokenizer)

        mock_file_reader_instance = mock_file_reader_cls.return_value

        def generator_side_effect(file_path):
            if file_path == "file2.jsonl":
                raise ValueError("Corrupt file")
            yield "doc_from_" + file_path

        mock_file_reader_instance.read_texts.side_effect = generator_side_effect

        processor = TokenProcessor(self.files, self.config)
        all_tokens = list(processor)

        self.assertEqual(len(all_tokens), 2)
        doc_strings = ["".join(map(chr, tokens)) for tokens in all_tokens]
        self.assertIn("doc_from_file1.jsonl", doc_strings)
        self.assertIn("doc_from_file3.jsonl", doc_strings)
        self.assertNotIn("doc_from_file2.jsonl", doc_strings)

    @patch("optimus_dl.recipe.pretokenize.processor.FileReader")
    @patch("optimus_dl.recipe.pretokenize.processor.build")
    def test_resumption_at_block_boundary(self, mock_build, mock_file_reader_cls):
        """Test resumption exactly when a block is finished."""
        self.config.processing.num_proc = 1
        self.config.processing.shuffle_buffer_size = 2

        mock_build.return_value = MockTokenizer(config=self.config.tokenizer)

        # 4 files, 1 doc each
        files = ["f1", "f2", "f3", "f4"]
        file_data_map = {f: [f"doc_{f}"] for f in files}

        def mock_read_texts(file_path):
            return file_data_map[file_path]

        mock_file_reader_instance = mock_file_reader_cls.return_value
        mock_file_reader_instance.read_texts.side_effect = mock_read_texts

        processor = TokenProcessor(files, self.config)
        iterator = iter(processor)

        # Yield 2 docs (1 block of size 2)
        yielded_first = [next(iterator) for _ in range(2)]

        state = processor.get_state()
        self.assertEqual(state["docs_yielded_in_block"], 2)

        # Resume
        new_processor = TokenProcessor(files, self.config)
        new_processor.load_state(state)

        yielded_rest = list(new_processor)

        self.assertEqual(len(yielded_rest), 2)  # f3, f4
        all_docs = yielded_first + yielded_rest

        doc_strings = ["".join(map(chr, tokens)) for tokens in all_docs]
        self.assertEqual(sorted(doc_strings), ["doc_f1", "doc_f2", "doc_f3", "doc_f4"])


if __name__ == "__main__":
    unittest.main()
