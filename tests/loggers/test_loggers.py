from unittest.mock import (
    Mock,
    patch,
)

from optimus_dl.modules.loggers.base import BaseMetricsLogger
from optimus_dl.modules.loggers.jsonl import (
    JsonlLogger,
    JsonlLoggerConfig,
)
from optimus_dl.modules.loggers.wandb import (
    WandbLogger,
    WandbLoggerConfig,
)


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
