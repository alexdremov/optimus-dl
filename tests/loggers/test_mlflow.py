from unittest.mock import (
    MagicMock,
    patch,
)

from optimus_dl.modules.loggers.mlflow import (
    MlflowLogger,
    MlflowLoggerConfig,
)


class TestMlflowLogger:
    """Tests for MLflow Logger"""

    def test_mlflow_logger_config(self):
        """Test MlflowLoggerConfig initialization with custom parameters."""
        config = MlflowLoggerConfig(
            tracking_uri="http://localhost:5000",
            project="test_experiment",
            name="test_run",
            async_logging=True,
            tags=["tag1", "tag2"],
            log_system_metrics=True,
        )

        assert config.tracking_uri == "http://localhost:5000"
        assert config.project == "test_experiment"
        assert config.name == "test_run"
        assert config.async_logging is True
        assert config.tags == ["tag1", "tag2"]
        assert config.log_system_metrics is True

    @patch("optimus_dl.modules.loggers.mlflow.MLFLOW_AVAILABLE", True)
    @patch("mlflow.start_run")
    def test_mlflow_logger_init(self, mock_start_run):
        """Test MlflowLogger initialization when MLflow is available."""
        config = MlflowLoggerConfig()
        logger = MlflowLogger(config)

        assert logger.enabled is True
        assert logger.run_id is None

    @patch("optimus_dl.modules.loggers.mlflow.MLFLOW_AVAILABLE", True)
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.config.enable_async_logging")
    @patch("mlflow.enable_system_metrics_logging")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.create_experiment")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.set_tags")
    @patch("mlflow.log_params")
    @patch("mlflow.log_artifact")
    @patch("mlflow.utils.validation.MAX_PARAM_VAL_LENGTH", 6000)
    def test_mlflow_logger_setup(
        self,
        mock_log_artifact,
        mock_log_params,
        mock_set_tags,
        mock_start_run,
        mock_set_experiment,
        mock_create_exp,
        mock_get_exp,
        mock_enable_sys,
        mock_enable_async,
        mock_set_uri,
    ):
        """Test MlflowLogger setup method with experiment initialization."""
        config = MlflowLoggerConfig(
            tracking_uri="http://localhost:5000",
            project="test_exp",
            async_logging=True,
            log_system_metrics=True,
        )
        logger = MlflowLogger(config)

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run
        # Mocking getting an existing experiment
        mock_get_exp.return_value = MagicMock(experiment_id="exp_id")

        experiment_name = "overridden_exp"
        config_dict = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "long_param": "a" * 7000,
        }

        logger.setup(experiment_name, config_dict)

        mock_set_uri.assert_called_with("http://localhost:5000")
        mock_enable_async.assert_called_with(True)
        mock_enable_sys.assert_called_once()
        mock_set_experiment.assert_called_with(experiment_id="exp_id")
        mock_start_run.assert_called_once()
        mock_set_tags.assert_called_once()
        mock_log_artifact.assert_called()

        # Verify long param was truncated in log_params
        args, _ = mock_log_params.call_args
        logged_params = args[0]
        assert len(logged_params["long_param"]) == 6000

        assert logger.run_id == "test_run_id"

    @patch("optimus_dl.modules.loggers.mlflow.MLFLOW_AVAILABLE", True)
    @patch("mlflow.log_metrics")
    def test_mlflow_logger_log_metrics(self, mock_log_metrics):
        """Test MlflowLogger metric logging with flattening and group prefixing."""
        config = MlflowLoggerConfig(async_logging=True)
        logger = MlflowLogger(config)
        logger.active_run = MagicMock()

        metrics = {"loss": 0.5, "accuracy": {"top1": 0.9, "top5": 0.99}}
        logger.log_metrics(metrics, step=100, group="train")

        expected_metrics = {
            "train/loss": 0.5,
            "train/accuracy/top1": 0.9,
            "train/accuracy/top5": 0.99,
        }
        mock_log_metrics.assert_called_once_with(
            expected_metrics, step=100, synchronous=False
        )

    @patch("optimus_dl.modules.loggers.mlflow.MLFLOW_AVAILABLE", True)
    @patch("mlflow.log_artifacts")
    @patch("mlflow.end_run")
    @patch("mlflow.flush_async_logging")
    @patch("mlflow.flush_artifact_async_logging")
    @patch("mlflow.flush_trace_async_logging")
    def test_mlflow_logger_close(
        self,
        mock_flush_trace,
        mock_flush_art,
        mock_flush,
        mock_end_run,
        mock_log_artifacts,
    ):
        """Test MlflowLogger close method."""
        config = MlflowLoggerConfig()
        logger = MlflowLogger(config)
        logger.active_run = MagicMock()
        logger.logs_parent_path = "/tmp/logs"

        with patch("os.path.exists", return_value=True):
            logger.close()

        mock_log_artifacts.assert_called_with("/tmp/logs", artifact_path="logs")
        mock_end_run.assert_called_once()
        mock_flush.assert_called_once()
        assert logger.active_run is None

    @patch("optimus_dl.modules.loggers.mlflow.MLFLOW_AVAILABLE", False)
    def test_mlflow_logger_disabled_when_unavailable(self):
        """Test that MlflowLogger is disabled when mlflow library is not available."""
        config = MlflowLoggerConfig()
        logger = MlflowLogger(config)

        assert logger.enabled is False
