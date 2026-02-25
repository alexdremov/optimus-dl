from __future__ import annotations

import hashlib
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass, field
from typing import (
    Any,
)

import torch

from optimus_dl.core.registry import (
    RegistryConfigStrict,
    make_registry,
)


class StandardProtocols:
    """Standardized string constants for common metric data protocols."""
    LOGITS = "logits"
    LOSS = "loss"
    GENERATED_IDS = "generated_ids"
    TARGETS = "targets"
    PREDICTIONS = "predictions"


@dataclass
class MetricSourceConfig(RegistryConfigStrict):
    """Base configuration for metric sources.
    
    Attributes:
        dependencies: Maps internal role requirements to source names within the group.
    """
    dependencies: dict[str, str] = field(default_factory=dict)


source_registry, register_source, build_source = make_registry("metric_source")


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
                cfg_dict = self.cfg.__dict__ if hasattr(self.cfg, '__dict__') else str(self.cfg)
            
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
    def requires(self) -> dict[str, set[str]]:
        """Mapping from internal dependency role name to required protocol strings.
        
        Override this if your source depends on the output of other sources.
        """
        return {}

    @abstractmethod
    def __call__(
        self, 
        model: torch.nn.Module, 
        batch: dict[str, Any], 
        dependencies: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute the source and return a dictionary mapping Protocol string to data.
        
        Args:
            model: The PyTorch model.
            batch: The current batch dictionary.
            dependencies: Data from required sources, mapped by role name.
        """
        raise NotImplementedError


@dataclass
class ForwardSourceConfig(MetricSourceConfig):
    _name: str = "forward"


@register_source("forward", ForwardSourceConfig)
class ForwardSource(MetricSource):
    """Standard forward pass.

    Provides whatever the model natively provides (defaulting to LOGITS).
    """
    
    def __init__(self, cfg: ForwardSourceConfig):
        super().__init__(cfg)
        self._provides_cache: set[str] | None = None

    @property
    def provides(self) -> set[str]:
        # We don't know the exact protocols until runtime unless we assume LOGITS.
        # However, for handshake purposes before runtime, we can assume it provides
        # at least LOGITS, or we might need to defer validation.
        # For simplicity in the static handshake, we assume it provides LOGITS.
        # A more robust approach would query the model class if available.
        return {StandardProtocols.LOGITS}

    def __call__(
        self, model: torch.nn.Module, batch: dict[str, Any], dependencies: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        with torch.no_grad():
            outputs = model(**batch)
            
            # If the model explicitly tells us what it provides, we could map it.
            # But the standard contract is returning outputs.
            result = {
                StandardProtocols.LOGITS: outputs.get("logits", outputs)
            }
            # Also merge anything else the model explicitly returned that matches a protocol
            if hasattr(model, "provides"):
                for p in model.provides:
                    if p in outputs:
                        result[p] = outputs[p]
            
            return result


@dataclass
class CausalLMSourceConfig(MetricSourceConfig):
    _name: str = "causal_lm"


@register_source("causal_lm", CausalLMSourceConfig)
class CausalLMSource(MetricSource):
    """Causal LM forward pass. Assumes causal shifting of labels.
    
    Provides: LOGITS, TARGETS
    """
    @property
    def provides(self) -> set[str]:
        return {StandardProtocols.LOGITS, StandardProtocols.TARGETS}

    def __call__(
        self, model: torch.nn.Module, batch: dict[str, Any], dependencies: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.get("logits", outputs)
            
        targets = batch["input_ids"][:, 1:]
        logits = logits[:, :-1, :]
        
        return {
            StandardProtocols.LOGITS: logits,
            StandardProtocols.TARGETS: targets
        }


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

    Provides: GENERATED_IDS
    """
    @property
    def provides(self) -> set[str]:
        return {StandardProtocols.GENERATED_IDS}

    def __call__(
        self, model: torch.nn.Module, batch: dict[str, Any], dependencies: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            return {StandardProtocols.GENERATED_IDS: generated_ids}
