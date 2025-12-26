import logging
from dataclasses import dataclass, field
from typing import Any

import torchdata.nodes
from omegaconf import MISSING, OmegaConf
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    MapperConfig,
    register_transform,
)
from optimus_dl.modules.tokenizer import build_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenizeTransformConfig(RegistryConfigStrict):
    """Configuration for text tokenization.

    Attributes:
        tokenizer_config: Registry configuration for the tokenizer to use.
        debug_samples: Number of samples to print to the log for debugging.
        worker_cfg: Configuration for parallel processing workers.
    """

    tokenizer_config: Any = MISSING
    debug_samples: int = 0
    worker_cfg: MapperConfig = field(
        default_factory=MapperConfig,
    )


@register_transform("tokenize", TokenizeTransformConfig)
class TokenizeTransform(BaseTransform):
    """Transform that converts raw text strings into sequences of token IDs.

    Uses the registry system to instantiate a tokenizer and applies it to the
    input data. Supports parallel mapping via `torchdata.nodes.ParallelMapper`.

    Args:
        cfg: Tokenization configuration.
    """

    def __init__(self, cfg: TokenizeTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = build_tokenizer(cfg.tokenizer_config)

        # Convert MapperConfig dataclass to dict for torchdata
        from dataclasses import asdict

        if hasattr(cfg.worker_cfg, "__dataclass_fields__"):
            # It's a dataclass
            self.mapper_cfg = asdict(cfg.worker_cfg)
        else:
            # Try OmegaConf conversion for backwards compatibility
            self.mapper_cfg = OmegaConf.to_container(cfg.worker_cfg)

        self.debug_counter = 0
        self.debug_samples = cfg.debug_samples

    def _map(self, sample):
        """Map raw text sample to input_ids using the internal tokenizer."""
        text = sample["text"]
        ids = self.tokenizer.encode(text)
        if self.debug_counter < self.debug_samples:
            self.debug_counter += 1
            tokens_debug = []
            for (
                token_id
            ) in (
                ids
            ):  # Renamed 'token' to 'token_id' to avoid confusion with token strings
                token_decoded = self.tokenizer.decode([token_id])
                tokens_debug.append(f"{token_id}({token_decoded})")

            tokens_debug = ", ".join(tokens_debug)
            logger.info(f"Debugging tokenizer sample: \n{tokens_debug}\n=======")

        return {
            "input_ids": ids,
        }

    def build(self, source: BaseNode) -> BaseNode:
        """Wrap the source node with a parallel mapper using the tokenizer function."""
        return torchdata.nodes.ParallelMapper(
            source=source,
            map_fn=self._map,
            **self.mapper_cfg,
        )
