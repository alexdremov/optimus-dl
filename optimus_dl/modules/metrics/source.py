import hashlib
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

from optimus_dl.core.registry import (
    RegistryConfigStrict,
    make_registry,
)


class StandardProtocols:
    """Standardized string constants for common metric data protocols."""

    LOGITS = "logits"
    LABELS = "labels"
    LOSS = "loss"
    GENERATED_IDS = "generated_ids"
    CLASSIFICATION = "classification"


@dataclass
class MetricSourceConfig(RegistryConfigStrict):
    """Base configuration for metric sources.

    Attributes:
        dependencies: Maps internal role requirements to source names within the group.
    """

    dependencies: dict[str, str] = field(default_factory=dict)


class MetricSource(ABC):
    """Base class for data producers that extract information from the model."""

    def __init__(self, cfg: MetricSourceConfig):
        self.cfg = cfg
        self._hash: str | None = None

    @property
    def config_hash(self) -> str:
        """Returns a deterministic hash of the source's configuration for cross-group caching."""
        if self._hash is None:
            import dataclasses

            if dataclasses.is_dataclass(self.cfg):
                cfg_dict = dataclasses.asdict(self.cfg)
            else:
                cfg_dict = (
                    self.cfg.__dict__
                    if hasattr(self.cfg, "__dict__")
                    else str(self.cfg)
                )

            def make_hashable(obj: Any) -> Any:
                if isinstance(obj, (tuple, list)):
                    return tuple(make_hashable(e) for e in obj)
                if isinstance(obj, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                return obj

            stable_repr = str(make_hashable(cfg_dict))
            # Include the class name so different source types with same config don't collide
            stable_repr = f"{self.__class__.__name__}:{stable_repr}"
            self._hash = hashlib.md5(stable_repr.encode()).hexdigest()
        return self._hash

    @property
    @abstractmethod
    def provides(self) -> set[str]:
        """Returns the set of protocol strings this source provides."""
        raise NotImplementedError

    @property
    def requires(self) -> set[str]:
        """Mapping from internal dependency role name to required protocol strings.

        Override this if your source depends on the output of other sources.
        """
        return set()

    @abstractmethod
    def __call__(
        self, dependencies: dict[str, dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Execute the source and return a dictionary mapping Protocol string to data.

        Args:
            dependencies: Data from required sources, mapped by protocol.
        """
        raise NotImplementedError


metric_source_registry, register_metric_source, build_metric_source = make_registry(
    "metric_source", MetricSource
)
