from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfig


@dataclass
class ModelTransformConfig(RegistryConfig):
    """Base configuration for model transforms."""

    pass
