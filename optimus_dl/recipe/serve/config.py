from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from optimus_dl.modules.tokenizer import BaseTokenizerConfig


@dataclass
class ServeCommonConfig:
    checkpoint_path: str | None = field(
        default=None, metadata={"description": "Path to model checkpoint"}
    )
    model: Any = field(
        default=None,
        metadata={
            "description": "Model to build (if you want to load model not from checkpoint)"
        },
    )
    tokenizer: BaseTokenizerConfig = field(default=MISSING)
    device: str = field(
        default="auto", metadata={"description": "Device to use (cuda, cpu, auto)"}
    )


@dataclass
class ServeRecipeConfig:
    port: int = field(default=8000, metadata={"description": "Port to serve on"})
    host: str = field(default="0.0.0.0", metadata={"description": "Host to serve on"})


@dataclass
class ServeConfig:
    serve: ServeRecipeConfig = field(default_factory=ServeRecipeConfig)
    common: ServeCommonConfig = field(default_factory=ServeCommonConfig)
