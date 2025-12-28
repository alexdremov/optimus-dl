from unittest.mock import (
    Mock,
    patch,
)

import datasets

from optimus_dl.modules.data.datasets.huggingface import (
    HuggingFaceDataset,
    HuggingFaceDatasetConfig,
)


class TestHuggingFaceDataset:
    """Test suite for HuggingFace dataset integration."""

    def test_config_creation(self):
        """Test that HuggingFaceDatasetConfig can be created with dataset load kwargs."""
        config = HuggingFaceDatasetConfig(
            dataset_load_kwargs={
                "path": "test/dataset",
                "split": "train",
            }
        )
        assert config.dataset_load_kwargs["path"] == "test/dataset"
        assert config.dataset_load_kwargs["split"] == "train"

    @patch("optimus_dl.modules.data.datasets.huggingface.load_dataset")
    def test_dataset_initialization(self, mock_load_dataset):
        """Test that dataset initializes correctly with mocked HuggingFace dataset."""
        # Create a mock dataset that behaves like an IterableDataset
        mock_dataset = Mock(spec=datasets.IterableDataset)
        mock_dataset.shard.return_value = mock_dataset
        mock_dataset.__iter__ = Mock(
            return_value=iter(
                [
                    {"text": "sample text 1"},
                    {"text": "sample text 2"},
                ]
            )
        )
        mock_dataset._distributed = False
        mock_load_dataset.return_value = mock_dataset

        config = HuggingFaceDatasetConfig(
            dataset_load_kwargs={"path": "test/dataset", "split": "train"}
        )

        dataset = HuggingFaceDataset(config, rank=0, world_size=1)
        dataset.reset()

        # Verify load_dataset was called with streaming=True
        expected_kwargs = {
            "path": "test/dataset",
            "split": "train",
            "streaming": True,
        }
        mock_load_dataset.assert_called_once_with(**expected_kwargs)

    @patch("optimus_dl.modules.data.datasets.huggingface.load_dataset")
    def test_next_iteration(self, mock_load_dataset):
        """Test that next() method correctly iterates through dataset items."""
        sample_data = [
            {"text": "first item"},
            {"text": "second item"},
        ]

        mock_dataset = Mock(spec=datasets.IterableDataset)
        mock_dataset.shard.return_value = mock_dataset
        mock_dataset.__iter__ = Mock(return_value=iter(sample_data))
        mock_load_dataset.return_value = mock_dataset

        config = HuggingFaceDatasetConfig(
            dataset_load_kwargs={"path": "test/dataset", "split": "train"}
        )

        dataset = HuggingFaceDataset(config, rank=0, world_size=1)
        dataset.reset()

        # Test iteration
        first_item = dataset.next()
        assert first_item == {"text": "first item"}
        assert dataset.position == 1

        second_item = dataset.next()
        assert second_item == {"text": "second item"}
        assert dataset.position == 2

    @patch("optimus_dl.modules.data.datasets.huggingface.load_dataset")
    def test_streaming_default_behavior(self, mock_load_dataset):
        """Test that streaming is enabled by default even if not specified."""
        mock_dataset = Mock(spec=datasets.IterableDataset)
        mock_dataset.shard.return_value = mock_dataset
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset

        # Config without streaming parameter
        config = HuggingFaceDatasetConfig(
            dataset_load_kwargs={"path": "test/dataset", "split": "train"}
        )

        dataset = HuggingFaceDataset(config, rank=0, world_size=1)
        dataset.reset()

        # Should have added streaming=True
        expected_kwargs = {
            "path": "test/dataset",
            "split": "train",
            "streaming": True,
        }
        mock_load_dataset.assert_called_once_with(**expected_kwargs)
