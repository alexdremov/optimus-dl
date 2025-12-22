from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from optimus_dl.modules.tokenizer import BaseTokenizerConfig


@dataclass
class ServeCommonConfig:
    checkpoint_path: str | None = field(
        default=None, metadata={"help": "Path to model checkpoint"}
    )
    model: Any = field(
        default=None,
        metadata={
            "help": "Model to build (if you want to load model not from checkpoint)"
        },
    )
    tokenizer: BaseTokenizerConfig = field(default=MISSING)
    device: str = field(
        default="auto", metadata={"help": "Device to use (cuda, cpu, auto)"}
    )


@dataclass
class ServeRecipeConfig:
    port: int = field(default=8000, metadata={"help": "Port to serve on"})
    host: str = field(default="0.0.0.0", metadata={"help": "Host to serve on"})


@dataclass
class ServeConfig:
    serve: ServeRecipeConfig = field(default_factory=ServeRecipeConfig)
    common: ServeCommonConfig = field(default_factory=ServeCommonConfig)
