"""Configuration for data preparation recipe."""

from dataclasses import (
    dataclass,
    field,
)
from typing import Any

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    repo_id: str = MISSING
    split: str = "train"
    config_name: str | None = None
    cache_dir: str | None = None
    file_pattern: str | None = None  # To filter files if needed


@dataclass
class ProcessingConfig:
    shard_size_mb: int = 512
    shuffle_buffer_size: int = 10000
    text_column: str = "text"
    seed: int = 42
    dtype: str = "uint16"  # uint16 or uint32
    num_proc: int = 1


@dataclass
class OutputConfig:
    dir: str = MISSING
    name: str = "dataset"  # Prefix for shards?


@dataclass
class DataPrepConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tokenizer: Any = MISSING
