"""MLflow metrics logger implementation.

This logger integrates with MLflow for experiment tracking, supporting
local and remote tracking servers with asynchronous logging capabilities.
"""

import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

import yaml
from omegaconf import OmegaConf

from optimus_dl.modules.loggers import register_metrics_logger
from optimus_dl.modules.loggers.base import BaseMetricsLogger
from optimus_dl.modules.loggers.config import MetricsLoggerConfig

logger = logging.getLogger(__name__)

if importlib.util.find_spec("mlflow") is not None:
    MLFLOW_AVAILABLE = True
else:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not available - install with 'pip install mlflow'")


@dataclass
class MlflowLoggerConfig(MetricsLoggerConfig):
    """Configuration for MLflow logger.

    Attributes:
        project: Name of the MLflow experiment (maps to wandb project).
        name: Name of the MLflow run.
        workspace: Workspace or group for the MLflow experiment.
        tracking_uri: URI of the MLflow tracking server.
        async_logging: If True, enables asynchronous logging for better performance.
        artifact_location: Optional custom artifact location for the experiment.
        log_system_metrics: If True, enables system metrics logging (CPU, GPU, Memory, etc.)
        max_retries: The maximum number of retries for HTTP requests to the tracking server.
        backoff_factor: The backoff factor for HTTP request retries.
        timeout: The timeout for HTTP requests in seconds.
    """

    # MLflow specific settings
    project: str | None = None
    name: str | None = None
    workspace: str | None = None
    tracking_uri: str | None = None
    async_logging: bool = True
    artifact_location: str | None = None
    log_system_metrics: bool = True

    # Reliability settings
    max_retries: int = 10
    backoff_factor: int = 2
    timeout: int = 15


def get_git_commit_hash() -> str | None:
    """Get the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            return result.stdout.strip()

    except Exception as e:
        logger.warning(f"Failed to get git commit hash: {e}")
    return None


def get_git_remote() -> str | None:
    """Get the git remote URL if available."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
            cwd=os.getcwd(),
        )
        if result.returncode == 0:
            return result.stdout.strip()

    except Exception as e:
        logger.warning(f"Failed to get git remote URL: {e}")

    return None


@register_metrics_logger("mlflow", MlflowLoggerConfig)
class MlflowLogger(BaseMetricsLogger):
    """MLflow metrics logger.

    Logs training metrics, configuration, and artifacts to MLflow.
    Supports asynchronous logging to minimize impact on training throughput.
    """

    def __init__(self, cfg: MlflowLoggerConfig, state_dict=None, **kwargs):
        """Initialize MLflow logger.

        Args:
            cfg: MLflow logger configuration.
            state_dict: Optional state containing 'run_id' for resuming.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(cfg, **kwargs)

        if not MLFLOW_AVAILABLE:
            self.enabled = False
            logger.error("MLflow logger disabled - mlflow package not available")
            return

        self.run_id = (state_dict or {}).get("run_id")
        self.active_run = None
        self.experiment_id = None

    def setup(
        self,
        experiment_name: str,
        config: dict[str, Any],
        logs_parent_path: str | None = None,
        start_iteration: int | None = None,
    ) -> None:
        """Initialize MLflow session, experiment, and run."""
        if not self.enabled:
            return

        import mlflow
        import mlflow.config
        import mlflow.environment_variables

        # Configure reliability settings via environment variables
        # MLflow's Python client respects these for its internal requests session

        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_MAX_RETRIES.set(
            self.cfg.max_retries
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR.set(
            self.cfg.backoff_factor
        )
        mlflow.environment_variables.MLFLOW_HTTP_REQUEST_TIMEOUT.set(self.cfg.timeout)
        mlflow.environment_variables.MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER.set(True)

        # Set tracking URI if provided
        tracking_uri = self.cfg.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")

        workspace = self.cfg.workspace or os.getenv("MLFLOW_WORKSPACE")
        if workspace is not None:
            try:
                mlflow.set_workspace(workspace)
            except Exception as e:
                logger.error(f"Failed to set MLflow workspace: {e}")

        # Enable async logging if requested
        if self.cfg.async_logging:
            try:
                # In newer MLflow versions (like 3.10.1), this is the preferred way
                mlflow.config.enable_async_logging(True)
                logger.info("MLflow asynchronous logging enabled")
            except AttributeError:
                logger.warning(
                    "mlflow.config.enable_async_logging not found, will use synchronous=False in log calls"
                )

        # Convert config to container if it's an OmegaConf object
        if OmegaConf.is_config(config):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        # Set experiment
        exp_name = self.cfg.project or os.getenv("MLFLOW_EXPERIMENT_NAME")
        try:
            # Check if experiment exists or create it
            if exp_name is not None:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(
                        exp_name, artifact_location=self.cfg.artifact_location
                    )
                else:
                    self.experiment_id = experiment.experiment_id

                mlflow.set_experiment(experiment_id=self.experiment_id)

            # Enable system metrics logging if requested
            if self.cfg.log_system_metrics:
                try:
                    mlflow.enable_system_metrics_logging()
                    logger.info("MLflow system metrics logging enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable system metrics logging: {e}")

            # Start or resume run

            run_name = (
                self.cfg.name or experiment_name or os.environ.get("MLFLOW_RUN_NAME")
            )
            self.active_run = mlflow.start_run(
                run_id=self.run_id,
                run_name=run_name,
                description=self.cfg.notes,
            )
            self.run_id = self.active_run.info.run_id

            # Apply user-defined tags after run start to ensure correct format
            if self.cfg.tags:
                try:
                    mlflow.set_tags(dict.fromkeys(self.cfg.tags, "true"))
                except Exception as e:
                    logger.warning(f"Failed to set user-defined tags: {e}")

            # Set environment tags for better traceability
            git_hash = get_git_commit_hash()
            git_remote = get_git_remote()
            try:
                mlflow.set_tags(
                    {
                        "env.python_version": sys.version.split()[0],
                        "env.platform": platform.platform(),
                        "env.hostname": platform.node(),
                        "git.commit_hash": git_hash or "unknown",
                        "git.remote": git_remote or "unknown",
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to set environment tags: {e}")

            # Log configuration as parameters
            # MLflow params have a limit on value length, so we handle large configs
            self._log_params_recursively(config_dict)

            # Also log full config as an artifact to ensure no loss of data
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = os.path.join(tmp_dir, "config.json")
                    with open(tmp_path, "w") as f:
                        json.dump(config_dict, f, indent=2, default=str)

                    mlflow.log_artifact(tmp_path, artifact_path="config")

                    tmp_path = os.path.join(tmp_dir, "config.yaml")
                    with open(tmp_path, "w") as f:
                        yaml.dump(config_dict, f, default_flow_style=False)

                    mlflow.log_artifact(tmp_path, artifact_path="config")
            except Exception as e:
                logger.warning(f"Failed to log config artifact: {e}")

            logger.info(
                f"MLflow run initialized: {self.active_run.info.run_name} (ID: {self.run_id}) in experiment '{exp_name or 'default'}'"
            )

            # Store logs path if provided
            if logs_parent_path:
                self.logs_parent_path = logs_parent_path

        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}", exc_info=True)
            self.enabled = False

    def _log_params_recursively(self, config: dict, prefix: str = "") -> None:
        """Helper to log nested configuration as parameters."""
        import mlflow
        from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

        params = {}
        for k, v in config.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                self._log_params_recursively(v, prefix=f"{key}.")
            else:
                val = str(v)
                if len(val) > MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f"Parameter '{key}' exceeds MAX_PARAM_VAL_LENGTH ({MAX_PARAM_VAL_LENGTH}). Truncating for parameter log. Full value is available in the config artifact."
                    )
                    val = val[:MAX_PARAM_VAL_LENGTH]
                params[key] = val

        if params:
            # Use async if enabled
            if self.cfg.async_logging:
                mlflow.log_params(params, synchronous=False)
            else:
                mlflow.log_params(params)

    def log_metrics(
        self, metrics: dict[str, Any], step: int, group: str = "train"
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names to values.
            step: Current iteration/step.
            group: Metric group (e.g., 'train', 'eval').
        """
        if not self.enabled or self.active_run is None:
            return

        import mlflow

        try:
            flattened = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        # Using '/' as separator is standard in MLflow for grouping
                        flattened[f"{group}/{key}/{k}"] = float(v)
                else:
                    flattened[f"{group}/{key}"] = float(value)

            # Log to MLflow
            if self.cfg.async_logging:
                mlflow.log_metrics(flattened, step=step, synchronous=False)
            else:
                mlflow.log_metrics(flattened, step=step)

        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def close(self) -> None:
        """Close the MLflow run and log final artifacts."""
        if not self.enabled or self.active_run is None:
            return

        import mlflow

        try:
            # Log logs directory as artifact if it exists
            if hasattr(self, "logs_parent_path") and os.path.exists(
                self.logs_parent_path
            ):
                mlflow.log_artifacts(self.logs_parent_path, artifact_path="logs")
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")

        try:
            mlflow.flush_artifact_async_logging()
            mlflow.flush_async_logging()
            mlflow.flush_trace_async_logging()
        except Exception as e:
            logger.error(f"Error flushing MLflow logging: {e}")

        try:
            mlflow.end_run()
            logger.info("MLflow run ended successfully")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
        finally:
            self.active_run = None

    def state_dict(self) -> dict[str, Any]:
        """Return the run_id for resumption."""
        return {"run_id": self.run_id}
