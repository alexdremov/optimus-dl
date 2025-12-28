"""Tests for the TokenProcessor."""

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
        # The order is not guaranteed due to shuffling, so we can't assert the first element reliably
        # without controlling the random seed within the test.
        # self.assertEqual(all_tokens[0], [ord(c) for c in "doc1"])
        self.assertEqual(processor.progress, 3)

    @patch("optimus_dl.recipe.pretokenize.processor.multiprocessing.get_context")
    @patch("optimus_dl.recipe.pretokenize.processor._tokenize_file_worker")
    def test_multi_process_iteration(self, mock_worker, mock_get_context):
        """Test that the processor correctly uses a multiprocessing pool."""
        self.config.processing.num_proc = 2

        mock_pool_instance = MagicMock()
        mock_context_instance = MagicMock()
        mock_get_context.return_value = mock_context_instance
        mock_context_instance.Pool.return_value = mock_pool_instance

        # Mock the results from the worker
        mock_pool_instance.imap.return_value = iter(
            [
                [[1, 2]],  # Tokens from file 1
                [[3]],  # Tokens from file 2
                [[4, 5]],  # Tokens from file 3
            ]
        )

        processor = TokenProcessor(self.files, self.config)
        all_tokens = list(processor)

        mock_get_context.assert_called_once_with("spawn")
        mock_context_instance.Pool.assert_called_once_with(2)
        mock_pool_instance.imap.assert_called_once()

        self.assertEqual(len(all_tokens), 3)
        self.assertEqual(processor.progress, 3)

    def test_state_management(self):
        """Test saving and loading of the processor's state."""
        processor = TokenProcessor(self.files, self.config)
        processor.file_idx = 2
        processor.buffer = [[1], [2], [3]]

        state = processor.get_state()

        new_processor = TokenProcessor(self.files, self.config)
        self.assertEqual(new_processor.file_idx, 0)

        new_processor.load_state(state)

        self.assertEqual(new_processor.file_idx, 2)
        self.assertEqual(new_processor.buffer, [[1], [2], [3]])


if __name__ == "__main__":
    unittest.main()
