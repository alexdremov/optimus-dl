from dataclasses import dataclass
from typing import Any

from omegaconf import II

from optimus_dl.core.registry import RegistryConfig


@dataclass
class MetricsLoggerConfig(RegistryConfig):
    """Base configuration for metrics loggers."""

    # Common fields that all loggers might use
    enabled: bool = True
    id: str = II("._name")

    # Optional experiment metadata
    tags: list[str] | None = None
    notes: str | None = None
