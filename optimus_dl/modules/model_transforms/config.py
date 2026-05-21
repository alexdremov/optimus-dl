from dataclasses import dataclass

from optimus_dl.core.registry import RegistryConfigStrict


@dataclass
class ModelTransformConfig(RegistryConfigStrict):
    """Base configuration for model transforms."""

    pass
