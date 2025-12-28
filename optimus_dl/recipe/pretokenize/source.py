"""Handles finding and reading data from various sources."""

import fnmatch
import json
import logging
import random
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
)
from tqdm import tqdm

from .config import (
    DatasetConfig,
    ProcessingConfig,
)

logger = logging.getLogger(__name__)


class FileFinder:
    """Discovers files from a Hugging Face Hub dataset repository.

    This class handles the logic for listing files in a dataset repo, filtering
    them by split/pattern, and optionally parsing `README.md` metadata to identify
    split-specific files (common in modern HF datasets).

    Args:
        config: Dataset configuration.
        seed: Random seed for shuffling file list order.
    """

    def __init__(self, config: DatasetConfig, seed: int):
        self.config = config
        self.seed = seed

    def get_files(self) -> list[str]:
        """Retrieve and filter the list of files to process.

        First attempts to use metadata from `README.md` to find files for the
        requested split/config. If that fails or is not applicable, falls back
        to file name pattern matching.

        Returns:
            List of file paths relative to the repository root.
        """
        logger.info(
            f"Listing files for {self.config.repo_id} split={self.config.split}"
        )
        all_files = list_repo_files(repo_id=self.config.repo_id, repo_type="dataset")
        logger.info(f"Found {len(all_files)} files before filtering.")

        if self.config.file_pattern is not None:
            logger.info(f"Filtering files based on pattern: {self.config.file_pattern}")
            files = self._filter_files(all_files, pattern=self.config.file_pattern)
        else:
            logger.info(
                f"Filtering files based on metadata for split '{self.config.split}' and config_name '{self.config.config_name}'"
            )
            files = self._get_files_from_metadata(all_files)

            if not files:
                logger.info(
                    "No metadata file found. Falling back to simple file name filtering."
                )
                assert (
                    self.config.config_name is None
                ), "config_name is not supported without metadata file"

                patterns = ["data/*"]
                if self.config.split and self.config.split != "all":
                    patterns = [
                        f"data/{self.config.split}-*",
                        f"data/{self.config.split}_*",
                        f"data/{self.config.split}/*",
                    ]
                files = []
                for pattern in patterns:
                    files += self._filter_files(all_files, pattern=pattern)

            if not files:
                logger.warning(
                    f"No files found after filtering for split '{self.config.split}'. {all_files = }"
                )
                return []

        # Shuffle the files for better distribution in the shuffle buffer
        random.seed(self.seed)
        random.shuffle(files)

        logger.info(f"Found {len(files)} files for processing.")
        return files

    def _get_files_from_metadata(self, all_files: list[str]) -> list[str] | None:
        """Parse dataset metadata from README.md to find relevant files."""
        if "README.md" not in all_files:
            return None

        try:
            readme_path = hf_hub_download(
                repo_id=self.config.repo_id,
                filename="README.md",
                repo_type="dataset",
                cache_dir=self.config.cache_dir,
            )
            with open(readme_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not download or read README.md: {e}")
            return None

        # Extract YAML front matter
        if not content.startswith("---"):
            logger.warning(
                "README.md does not contain YAML front matter (content.startswith('---'))."
            )
            return None

        parts = content.split("---")
        if len(parts) < 3:
            logger.warning(
                "README.md does not contain valid YAML front matter (content.split('---'))."
            )
            return None

        yaml_content = parts[1]
        try:
            metadata = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML from README.md: {e}")
            return None

        if not isinstance(metadata, dict) or "configs" not in metadata:
            logger.warning(f"Invalid metadata format in README.md: {metadata}")
            return None

        split_info = next(
            (
                s
                for s in metadata["configs"]
                if s.get("config_name") == self.config.config_name
            ),
            None,
        )

        if not split_info:
            logger.warning(
                f"No split info found in README.md {self.config.config_name = }, {metadata['configs'] = }"
            )
            return None

        patterns = [
            pattern["path"]
            for pattern in split_info["data_files"]
            if pattern["split"] == self.config.split or pattern["split"] == "all"
        ]

        matched_files = []
        for pattern in tqdm(patterns, desc="Matching patterns", leave=False):
            matched = fnmatch.filter(all_files, pattern)
            matched_files.extend(matched)

        return matched_files

    def _filter_files(self, all_files: list[str], pattern=None) -> list[str]:
        """Filters files based on extension, split, and pattern."""
        filtered = []

        for f in tqdm(all_files, desc="Filtering files", unit="file", leave=False):
            if not f.endswith((".parquet", ".jsonl", ".json")):
                continue
            if pattern and not fnmatch.fnmatch(f, pattern):
                continue
            filtered.append(f)

        filtered.sort()  # Sort for deterministic order before shuffling
        return filtered


class FileReader:
    """Reads raw text documents from different file formats.

    Supports reading text columns from:

    - Parquet files (`.parquet`)
    - JSON Lines files (`.jsonl`)
    - JSON files (`.json`)

    Handles automatic downloading from the Hub if files are remote.

    Args:
        config: Processing configuration (defines text column name).
        dataset_config: Dataset configuration (defines cache dir, repo ID).
    """

    def __init__(self, config: ProcessingConfig, dataset_config: DatasetConfig):
        self.text_column = config.text_column
        self.dataset_config = dataset_config

    def read_texts(self, file_path: str) -> Generator[str, None, None]:
        """Download and read a file, yielding text documents one by one.

        Args:
            file_path: Path to the file in the repo.

        Yields:
            String content of each document found in the file.
        """
        local_path = hf_hub_download(
            repo_id=self.dataset_config.repo_id,
            filename=file_path,
            repo_type="dataset",
            cache_dir=self.dataset_config.cache_dir,
        )
        local_path = Path(local_path)
        assert local_path.exists(), f"File not found: {local_path}"

        if file_path.endswith(".parquet"):
            yield from self._read_parquet(local_path)
        elif file_path.endswith((".jsonl", ".json")):
            yield from self._read_jsonl(local_path)

    def _read_parquet(self, local_path: Path) -> Generator[str, None, None]:
        """Reads texts from a Parquet file using streaming."""
        try:
            import pyarrow.parquet as pq

            # Use iter_batches to stream the file instead of loading it entirely
            parquet_file = pq.ParquetFile(local_path)
            total_rows = parquet_file.metadata.num_rows

            with tqdm(
                total=total_rows,
                desc=f"Reading {local_path.name} (streaming)",
                unit="row",
                leave=False,
            ) as pbar:
                for batch in parquet_file.iter_batches(columns=[self.text_column], batch_size=100):
                    # batch is a RecordBatch, convert to dict or pandas
                    # We can access columns directly as arrays
                    column_data = batch[self.text_column]
                    # Iterate over the PyArrow array efficiently
                    for item in column_data:
                        # item is a pyarrow scalar, convert to python string
                        text = item.as_py()
                        if isinstance(text, str) and text:
                            yield text
                    pbar.update(batch.num_rows)

        except ImportError:
            logger.warning(
                "PyArrow not available, falling back to non-streaming pandas read."
            )
            df = pd.read_parquet(local_path)
            if self.text_column in df.columns:
                for text in tqdm(
                    df[self.text_column],
                    desc=f"Reading {local_path.name} (inefficient)",
                    unit="row",
                    leave=False,
                ):
                    if isinstance(text, str) and text:
                        yield text

    def _read_jsonl(self, local_path: Path) -> Generator[str, None, None]:
        """Reads texts from a JSONL file."""
        file_size = local_path.stat().st_size
        with tqdm(
            total=file_size,
            desc=f"Reading {local_path.name}",
            unit="B",
            unit_scale=True,
            leave=False,
        ) as pbar:
            with open(local_path, encoding="utf-8") as f:
                for line in f:
                    pbar.update(len(line))
                    item = json.loads(line)

                    if isinstance(item, dict):
                        text = item.get(self.text_column, "")
                        if isinstance(text, str) and text:
                            yield text
                    elif isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, dict):
                                text = sub_item.get(self.text_column, "")
                                if isinstance(text, str) and text:
                                    yield text
