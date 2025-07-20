import tempfile
from unittest.mock import Mock, patch

import pytest

from optimus_dl.modules.loggers.base import BaseMetricsLogger
from optimus_dl.modules.loggers.jsonl import JsonlLogger, JsonlLoggerConfig
from optimus_dl.modules.loggers.wandb import WandbLogger, WandbLoggerConfig


class TestJsonlLogger:
    """Tests for JSONL Logger"""

    def test_jsonl_logger_config(self):
        """Test JsonlLoggerConfig initialization with custom parameters."""
        config = JsonlLoggerConfig(
            output_dir="/tmp/logs",
            include_timestamp=True,
            max_file_size_mb=100,
            include_group_in_filename=True,
        )

        assert config.output_dir == "/tmp/logs"
        assert config.include_timestamp is True
        assert config.max_file_size_mb == 100
        assert config.include_group_in_filename is True

    def test_jsonl_logger_init(self):
        """Test JsonlLogger initialization with directory creation."""
        config = JsonlLoggerConfig(output_dir="/tmp/logs")
        with patch("pathlib.Path.mkdir"):
            logger = JsonlLogger(config)
            assert logger.output_dir.name == "logs"

    @patch("builtins.open")
    @patch("pathlib.Path.mkdir")
    def test_jsonl_logger_log_metrics(self, mock_mkdir, mock_open):
        """Test JSONL metric logging functionality with mocked file operations."""
        config = JsonlLoggerConfig(output_dir="/tmp/logs")
        logger = JsonlLogger(config)

        mock_file = Mock()
        mock_open.return_value = mock_file
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)

        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log_metrics(metrics, step=100, group="train")

        mock_file.write.assert_called()

    @patch("pathlib.Path.mkdir")
    def test_jsonl_logger_setup(self, mock_mkdir):
        """Test JsonlLogger setup method with experiment configuration."""
        config = JsonlLoggerConfig(output_dir="/tmp/logs")
        logger = JsonlLogger(config)

        with patch("builtins.open"), patch("json.dump"), patch("yaml.dump"):
            logger.setup("test_experiment", {"param1": "value1"})


class TestWandbLogger:
    """Tests for WandB Logger"""

    def test_wandb_logger_config(self):
        """Test WandbLoggerConfig initialization with custom parameters."""
        config = WandbLoggerConfig(
            project="test_project",
            entity="test_entity",
            mode="online",
            job_type="train",
        )

        assert config.project == "test_project"
        assert config.entity == "test_entity"
        assert config.mode == "online"
        assert config.job_type == "train"

    @patch("optimus_dl.modules.loggers.wandb.WANDB_AVAILABLE", True)
    @patch("optimus_dl.modules.loggers.wandb.wandb")
    def test_wandb_logger_init(self, mock_wandb):
        """Test WandbLogger initialization when W&B is available."""
        config = WandbLoggerConfig(project="test_project")
        logger = WandbLogger(config)

        assert logger.enabled is True

    @patch("optimus_dl.modules.loggers.wandb.WANDB_AVAILABLE", True)
    @patch("optimus_dl.modules.loggers.wandb.wandb")
    def test_wandb_logger_setup(self, mock_wandb):
        """Test WandbLogger setup method with experiment initialization."""
        config = WandbLoggerConfig(project="test_project")
        logger = WandbLogger(config)

        mock_run = Mock()
        mock_wandb.init.return_value = mock_run

        logger.setup("test_experiment", {"param1": "value1"})

        mock_wandb.init.assert_called_once()

    @patch("optimus_dl.modules.loggers.wandb.WANDB_AVAILABLE", True)
    @patch("optimus_dl.modules.loggers.wandb.wandb")
    def test_wandb_logger_log_metrics(self, mock_wandb):
        """Test WandbLogger metric logging with group prefixing."""
        config = WandbLoggerConfig(project="test_project")
        logger = WandbLogger(config)

        mock_run = Mock()
        logger.run = mock_run

        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log_metrics(metrics, step=100, group="train")

        expected_metrics = {"train/loss": 0.5, "train/accuracy": 0.95}
        mock_run.log.assert_called_once_with(expected_metrics, step=100)

    @patch("optimus_dl.modules.loggers.wandb.WANDB_AVAILABLE", False)
    def test_wandb_logger_disabled_when_unavailable(self):
        """Test that WandbLogger is disabled when W&B library is not available."""
        config = WandbLoggerConfig(project="test_project")
        logger = WandbLogger(config)

        assert logger.enabled is False


class TestLoggerIntegration:
    """Integration tests for logger usage with metrics"""

    @patch("builtins.open")
    @patch("pathlib.Path.mkdir")
    def test_jsonl_logger_real_workflow(self, mock_mkdir, mock_open):
        """Test JSONL logger in a realistic multi-step workflow with metric logging."""
        mock_file = Mock()
        mock_open.return_value = mock_file
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)

        config = JsonlLoggerConfig(output_dir="/tmp/logs")
        logger = JsonlLogger(config)

        # Simulate multiple metric logging calls
        for i in range(5):
            metrics = {"loss": 1.0 / (i + 1), "learning_rate": 0.001, "step": i}
            logger.log_metrics(metrics, step=i * 10, group="train")

        # Should have written 5 times
        assert mock_file.write.call_count == 5

    @patch("optimus_dl.modules.loggers.wandb.WANDB_AVAILABLE", True)
    @patch("optimus_dl.modules.loggers.wandb.wandb")
    def test_wandb_logger_real_workflow(self, mock_wandb):
        """Test WandB logger in a realistic workflow with multiple groups."""
        config = WandbLoggerConfig(project="test_project")
        logger = WandbLogger(config)

        mock_run = Mock()
        logger.run = mock_run

        # Simulate multiple metric logging calls
        for i in range(5):
            train_metrics = {"loss": 1.0 / (i + 1), "accuracy": 0.5 + i * 0.1}
            logger.log_metrics(train_metrics, step=i * 10, group="train")

            eval_metrics = {"val_loss": 1.2 / (i + 1), "val_accuracy": 0.4 + i * 0.1}
            logger.log_metrics(eval_metrics, step=i * 10, group="eval")

        # Should have called log 10 times (5 train + 5 eval)
        assert mock_run.log.call_count == 10


class MockLogger(BaseMetricsLogger):
    """Mock logger for testing base functionality"""

    def __init__(self, cfg=None):
        super().__init__(cfg or Mock())
        self.logged_metrics = []

    def setup(self, experiment_name, config):
        pass

    def log_metrics(self, metrics, step, group="train"):
        self.logged_metrics.append({"metrics": metrics, "step": step, "group": group})

    def close(self):
        pass


class TestBaseLogger:
    """Tests for BaseLogger interface"""

    def test_base_logger_interface(self):
        """Test that BaseLogger interface works correctly with mock implementation."""
        logger = MockLogger()

        # Test log_metrics
        metrics = {"loss": 0.5}
        logger.log_metrics(metrics, step=1, group="train")

        assert len(logger.logged_metrics) == 1
        assert logger.logged_metrics[0]["metrics"] == {"loss": 0.5}
        assert logger.logged_metrics[0]["step"] == 1
        assert logger.logged_metrics[0]["group"] == "train"

    def test_base_logger_setup_and_close(self):
        """Test that setup and close methods execute without errors."""
        logger = MockLogger()

        # Should not raise
        logger.setup("test_experiment", {"config": "value"})
        logger.close()
