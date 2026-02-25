from __future__ import annotations

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


@dataclass
class MetricConfig(RegistryConfigStrict):
    nested_name: str | None = field(
        default=None,
        metadata={
            "description": "Optional name to nest this metric under in the metrics tree."
        },
    )


metric_registry, register_metric, build_metric = make_registry("metric")


class Metric(ABC):
    """Stateless definition for computing metrics from model/source data.

    A Metric implementation defines:
    - What data it requires via `requires` mapping (Role -> set of protocol strings).
    - How to calculate raw results for a batch, potentially emitting multiple sub-values.
    - How to finalize those values after they've been aggregated (e.g., F1 from counts).
    """

    def __init__(self, cfg: MetricConfig):
        self.cfg = cfg
        self.nested_name = cfg.nested_name

    @property
    @abstractmethod
    def requires(self) -> set[str]:
        """Mapping from source role name to a set of required protocol strings."""
        raise NotImplementedError

    @property
    def accumulators(self) -> dict[str, str]:
        """Define how each sub-metric should be aggregated across batches.

        Returns a mapping from sub-metric names to accumulator types
        (e.g., 'average', 'sum', 'gather', 'perplexity').
        """
        return {self.cfg._name: "average"}

    @abstractmethod
    def __call__(
        self, sources_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Compute raw metric values for the batch.

        Args:
            batch: The original batch dictionary.
            sources_data: Protocol string -> data.

        Returns:
            Dict mapping sub-metric names to log kwargs (e.g., {'value': ..., 'weight': ...})
            for accumulators.
        """
        raise NotImplementedError

    def finalize(self, aggregated_data: dict[str, Any]) -> dict[str, Any]:
        """Perform final calculations on aggregated data.

        Args:
            aggregated_data: Dict mapping sub-metric names to their
                computed/aggregated values from accumulators.

        Returns:
            Dict of final metrics to be logged/reported.
        """
        return {
            k: v for k, v in aggregated_data.items() if not k.startswith("_internal/")
        }
