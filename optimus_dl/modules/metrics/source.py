from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    Any,
)

import torch

from optimus_dl.core.registry import (
    RegistryConfigStrict,
    make_registry,
)


@dataclass
class MetricSourceConfig(RegistryConfigStrict):
    pass


source_registry, register_source, build_source = make_registry("metric_source")


class MetricSource(ABC):
    """Base class for data producers that extract information from the model.

    Sources are responsible for executing model logic (e.g., forward pass,
    generation) and returning a dictionary of data that metrics can use.
    The MetricEngine ensures that each source is executed at most once per batch.
    """

    @abstractmethod
    def __call__(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute the source and return a dictionary of results."""
        raise NotImplementedError


@dataclass
class ForwardSourceConfig(MetricSourceConfig):
    _name: str = "forward"


@register_source("forward", ForwardSourceConfig)
class ForwardSource(MetricSource):
    """Standard forward pass.

    Returns:
        Dict containing 'logits' and any other model outputs.
    """

    def __init__(self, cfg: ForwardSourceConfig):
        self.cfg = cfg

    def __call__(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
        with torch.no_grad():
            outputs = model(**batch)
            return outputs


@dataclass
class GenerationSourceConfig(MetricSourceConfig):
    _name: str = "generation"
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50


@register_source("generation", GenerationSourceConfig)
class GenerationSource(MetricSource):
    """Autoregressive generation pass.

    Returns:
        Dict containing 'generated_ids'.
    """

    def __init__(self, cfg: GenerationSourceConfig):
        self.cfg = cfg

    def __call__(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
        # Implementation depends on model having a .generate method or similar
        # For now, we assume a standard interface
        with torch.no_grad():
            # Usually we only generate for a subset of the batch or with prompt truncation
            # This is a simplified version
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            return {"generated_ids": generated_ids}
