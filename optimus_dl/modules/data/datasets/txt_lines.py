import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field

import requests
from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfigStrict

from . import register_dataset
from .base import BaseDataset

logger = logging.getLogger(__name__)


@dataclass
class TxtLinesDatasetConfig(RegistryConfigStrict):
    file_link: str = MISSING
    cache_dir: str = field(default_factory=tempfile.gettempdir)
    skip_empty_lines: bool = True


@register_dataset("txt_lines", TxtLinesDatasetConfig)
class TxtLinesDataset(BaseDataset):
    def __init__(
        self, cfg: TxtLinesDatasetConfig, rank: int, world_size: int, **kwargs
    ):
        super().__init__(cfg)
        self.file_link = cfg.file_link
        self.cache_dir = cfg.cache_dir
        self.skip_empty_lines = cfg.skip_empty_lines
        self.rank = rank
        self.world_size = world_size

        self.lines = []
        self.index = 0
        self.limit = 0

    def _prepare_data(self):
        # 1. Resolve path / download
        local_path = self.file_link
        if self.file_link.startswith("http://") or self.file_link.startswith(
            "https://"
        ):
            # Encode URL to filename to avoid collisions
            url_hash = hashlib.sha256(self.file_link.encode("utf-8")).hexdigest()
            # Try to keep the extension if present
            _, ext = os.path.splitext(self.file_link)
            if not ext:
                ext = ".txt"
            # Sanitize extension (just in case)
            if len(ext) > 10:
                ext = ".txt"

            filename = f"{url_hash}{ext}"
            local_path = os.path.join(self.cache_dir, filename)

            if not os.path.exists(local_path):
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Downloading {self.file_link} to {local_path}")
                try:
                    response = requests.get(self.file_link, stream=True)
                    response.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    logger.error(f"Failed to download {self.file_link}: {e}")
                    raise e

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        # 2. Read and filter
        logger.info(f"Loading data from {local_path}")
        with open(local_path, encoding="utf-8") as f:
            raw_lines = f.readlines()

        # Strip trailing newlines
        self.lines = [i.rstrip("\n") for i in raw_lines]

        if self.skip_empty_lines:
            self.lines = [i for i in self.lines if i.strip()]

        # 3. Shard
        total_lines = len(self.lines)
        if total_lines == 0:
            logger.warning(f"Dataset at {local_path} is empty after filtering.")
            self.limit = 0
            self.index = 0
            return

        lines_per_rank = total_lines // self.world_size
        self.index = lines_per_rank * self.rank
        self.limit = lines_per_rank * (self.rank + 1)

        # Ensure index is within bounds (just in case)
        self.index = max(0, min(self.index, total_lines))
        self.limit = max(0, min(self.limit, total_lines))

    def next(self):
        if self.index >= self.limit:
            raise StopIteration

        line = self.lines[self.index]
        self.index += 1
        return {"text": line}

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)
        initial_state = initial_state or {}

        if "file_link" in initial_state:
            self.file_link = initial_state["file_link"]

        if "skip_empty_lines" in initial_state:
            self.skip_empty_lines = initial_state["skip_empty_lines"]

        self._prepare_data()

        self.index = initial_state.get("index", self.index)
        assert initial_state.get("rank", self.rank) == self.rank
        assert initial_state.get("world_size", self.world_size) == self.world_size

    def get_state(self):
        return {
            "index": self.index,
            "file_link": self.file_link,
            "skip_empty_lines": self.skip_empty_lines,
            "rank": self.rank,
            "world_size": self.world_size,
        }
