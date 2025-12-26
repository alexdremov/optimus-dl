"""JSONL (JSON Lines) file metrics logger implementation.

This logger writes metrics to JSONL files, with one JSON object per line,
making it easy to process logs with standard tools or custom scripts.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml
from omegaconf import OmegaConf

from optimus_dl.modules.loggers import register_metrics_logger
from optimus_dl.modules.loggers.base import BaseMetricsLogger
from optimus_dl.modules.loggers.config import MetricsLoggerConfig

logger = logging.getLogger(__name__)


@dataclass
class JsonlLoggerConfig(MetricsLoggerConfig):
    """Configuration for JSONL file logger.

    Attributes:
        output_dir: Directory where log files will be saved.
        filename: Base name for the metrics file.
        append_mode: If True, appends to existing files. If False, overwrites.
        flush_every: Number of writes before forcing a disk sync.
        include_timestamp: Whether to add an ISO timestamp to each entry.
        include_group_in_filename: If True, creates separate files for each
            metric group (e.g., 'metrics_eval.jsonl').
        max_file_size_mb: Maximum size of a log file before it is rotated.
        max_rotated_files: Number of old log files to keep (-1 for unlimited).
    """

    # File settings
    output_dir: str = "logs"
    filename: str = "metrics.jsonl"

    # Logging behavior
    append_mode: bool = True  # Append to existing file vs overwrite
    flush_every: int = 10  # Flush to disk every N writes
    include_timestamp: bool = True
    include_group_in_filename: bool = True  # Separate files per group

    # File rotation settings
    max_file_size_mb: int = 200  # Maximum file size in MB before rotation
    max_rotated_files: int = (
        -1
    )  # Maximum number of rotated files to keep (-1 for unlimited)

    # Data formatting
    indent: int | None = None  # JSON indentation (None for compact)
    sort_keys: bool = False


@register_metrics_logger("jsonl", JsonlLoggerConfig)
class JsonlLogger(BaseMetricsLogger):
    """JSONL file metrics logger.

    Writes metrics to JSON Lines files, with one JSON object per line.
    Each line contains the step, group, timestamp (optional), and all metrics.
    Features automatic file rotation and separate file support for different
    metric groups.

    Example output:
    ```json
    {"step": 1, "group": "train", "timestamp": "2024-01-01T12:00:00", "loss": 2.5, "lr": 0.001}
    {"step": 2, "group": "train", "timestamp": "2024-01-01T12:00:01", "loss": 2.3, "lr": 0.001}
    ```
    """

    def __init__(self, cfg: JsonlLoggerConfig, **kwargs):
        """Initialize JSONL logger.

        Args:
            cfg: JSONL logger configuration
            **kwargs: Additional keyword arguments
        """
        super().__init__(cfg, **kwargs)

        self.output_dir = Path(cfg.output_dir)
        self.base_filename = cfg.filename
        self.file_handles: dict[str, TextIO] = {}
        self.file_paths: dict[str, Path] = {}  # Track current file paths
        self.write_count = 0
        self.max_file_size_bytes = (
            cfg.max_file_size_mb * 1024 * 1024
        )  # Convert MB to bytes

        if self.enabled:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"JSONL logger will write to: {self.output_dir}")

    def setup(self, experiment_name: str, config: dict[str, Any]) -> None:
        """Setup JSONL logger and export the experiment configuration.

        Writes the provided configuration to both `.json` and `.yaml` files in the
        output directory for reproducibility.
        """
        if not self.enabled:
            return

        try:
            # Write experiment config to separate file
            config_file = self.output_dir / f"{experiment_name}_config.json"
            with open(config_file, "w") as f:
                if OmegaConf.is_config(config):
                    config = OmegaConf.to_container(config, resolve=True)
                json.dump(config, f, indent=2, default=str)

            config_file = self.output_dir / f"{experiment_name}_config.yaml"
            with open(config_file, "w") as f:
                if OmegaConf.is_config(config):
                    config = OmegaConf.to_container(config, resolve=True)
                yaml.dump(config, f, indent=2)

            logger.info(f"Experiment config saved to: {config_file}")

        except Exception as e:
            logger.error(f"Failed to write config file: {e}")

    def _rotate_file(self, group: str) -> None:
        """Rotate log file when it gets too large.

        Renames the current file (e.g., `metrics.jsonl` -> `metrics.1.jsonl`) and
        opens a new one.
        """
        if group not in self.file_handles:
            return

        try:
            # Close current handle
            current_handle = self.file_handles[group]
            current_handle.flush()
            current_handle.close()

            current_path = self.file_paths[group]

            # Rotate existing files: file.log -> file.1.log -> file.2.log -> ...
            base_name = current_path.stem
            extension = current_path.suffix

            # Handle unlimited files case (-1)
            if self.cfg.max_rotated_files > 0:
                # Remove oldest rotated file if max limit reached
                oldest_file = (
                    current_path.parent
                    / f"{base_name}.{self.cfg.max_rotated_files}{extension}"
                )
                if oldest_file.exists():
                    oldest_file.unlink()

                # Rotate existing files (backwards to avoid overwrites)
                for i in range(self.cfg.max_rotated_files - 1, 0, -1):
                    old_file = current_path.parent / f"{base_name}.{i}{extension}"
                    new_file = current_path.parent / f"{base_name}.{i + 1}{extension}"
                    if old_file.exists():
                        old_file.rename(new_file)
            else:
                # Unlimited files (-1): find highest numbered file and increment
                max_index = 0
                for existing_file in current_path.parent.glob(
                    f"{base_name}.*{extension}"
                ):
                    try:
                        # Extract number from filename like "metrics.5.jsonl"
                        name_parts = existing_file.stem.split(".")
                        if len(name_parts) >= 2 and name_parts[-1].isdigit():
                            index = int(name_parts[-1])
                            max_index = max(max_index, index)
                    except (ValueError, IndexError):
                        continue

                # Rotate existing files starting from highest index
                for i in range(max_index, 0, -1):
                    old_file = current_path.parent / f"{base_name}.{i}{extension}"
                    new_file = current_path.parent / f"{base_name}.{i + 1}{extension}"
                    if old_file.exists():
                        old_file.rename(new_file)

            # Move current file to .1 position
            rotated_file = current_path.parent / f"{base_name}.1{extension}"
            current_path.rename(rotated_file)

            # Create new file handle
            new_handle = open(current_path, "a", encoding="utf-8")
            self.file_handles[group] = new_handle

            logger.info(f"Rotated JSONL file for group '{group}': {current_path}")

        except Exception as e:
            logger.error(f"Failed to rotate file for group '{group}': {e}")
            # Try to create a new handle anyway
            try:
                new_handle = open(self.file_paths[group], "a", encoding="utf-8")
                self.file_handles[group] = new_handle
            except Exception as create_error:
                logger.error(f"Failed to create new file handle: {create_error}")
                # Remove the group from file_handles to force recreation on next write
                self.file_handles.pop(group, None)

    def _should_rotate_file(self, group: str) -> bool:
        """Check if file should be rotated based on size."""
        if group not in self.file_paths:
            return False

        try:
            file_path = self.file_paths[group]
            if file_path.exists():
                file_size = file_path.stat().st_size
                return file_size >= self.max_file_size_bytes
        except Exception as e:
            logger.error(f"Error checking file size for group '{group}': {e}")

        return False

    def _get_file_handle(self, group: str) -> TextIO:
        """Get or create the file handle for a specific metrics group."""
        if group in self.file_handles:
            return self.file_handles[group]

        # Determine filename
        if self.cfg.include_group_in_filename and group != "train":
            # Use separate files for different groups
            base_name = self.base_filename.rsplit(".", 1)[0]
            extension = (
                self.base_filename.rsplit(".", 1)[1]
                if "." in self.base_filename
                else "jsonl"
            )
            group_safe = group.replace("/", "_").replace("\\", "_")
            filename = f"{base_name}_{group_safe}.{extension}"
        else:
            filename = self.base_filename

        filepath = self.output_dir / filename

        # Store file path for rotation tracking
        self.file_paths[group] = filepath

        # Open file handle
        mode = "a" if self.cfg.append_mode else "w"
        handle = open(filepath, mode, encoding="utf-8")

        self.file_handles[group] = handle
        logger.info(f"Opened JSONL file for group '{group}': {filepath}")

        return handle

    def log_metrics(
        self, metrics: dict[str, Any], step: int, group: str = "train"
    ) -> None:
        """Flatten and write metrics to the appropriate JSONL file."""
        if not self.enabled:
            return

        try:
            # Check if file needs rotation before writing
            if self._should_rotate_file(group):
                self._rotate_file(group)

            # Get file handle for this group
            file_handle = self._get_file_handle(group)

            # Prepare log entry
            log_entry = {
                "step": step,
                "group": group,
            }

            # Add timestamp if configured
            if self.cfg.include_timestamp:
                from datetime import datetime, timezone

                log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Flatten and add metrics
            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Handle nested metrics by flattening
                    for nested_key, nested_value in value.items():
                        full_key = f"{key}_{nested_key}"
                        log_entry[full_key] = nested_value
                else:
                    log_entry[key] = value

            # Write JSON line
            json_line = json.dumps(
                log_entry,
                indent=self.cfg.indent,
                sort_keys=self.cfg.sort_keys,
                default=str,  # Convert non-serializable objects to string
            )
            file_handle.write(json_line + "\n")

            # Flush periodically
            self.write_count += 1
            if self.write_count % self.cfg.flush_every == 0:
                file_handle.flush()
                os.fsync(file_handle.fileno())

        except Exception as e:
            logger.error(f"Failed to write metrics to JSONL: {e}")

    def close(self) -> None:
        """Close all open file handles and sync to disk."""
        for group, handle in self.file_handles.items():
            try:
                handle.flush()
                handle.close()
                logger.info(f"Closed JSONL file for group '{group}'")
            except Exception as e:
                logger.error(f"Error closing JSONL file for group '{group}': {e}")

        self.file_handles.clear()
