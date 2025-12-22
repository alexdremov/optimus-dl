"""Configuration for evaluation recipe."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING


@dataclass
class EvalCommonConfig:
    """Common evaluation configuration."""

    checkpoint_path: str | None = field(
        default=MISSING,
        metadata={"help": "Path to checkpoint directory or metadata file"},
    )
    model: Any = field(
        default=None,
        metadata={
            "help": "Model to build (if you want to load model not from checkpoint)"
        },
    )
    use_gpu: bool = field(
        default=True,
        metadata={"help": "Use gpu if available"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for evaluation"})
    tokenizer: Any = MISSING


@dataclass
class LMEvalConfig:
    """Configuration for lm_eval harness evaluation."""

    tasks: list[str] = field(
        default_factory=lambda: ["hellaswag"],
        metadata={"help": "List of lm_eval tasks to evaluate on"},
    )
    num_fewshot: int = field(
        default=0, metadata={"help": "Number of few-shot examples"}
    )
    batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    limit: int | None = field(
        default=None, metadata={"help": "Limit number of examples per task"}
    )
    output_path: str | None = field(
        default=None, metadata={"help": "Path to save evaluation results"}
    )


@dataclass
class EvalConfig:
    """Main evaluation configuration."""

    common: EvalCommonConfig = field(default_factory=EvalCommonConfig)
    lm_eval: LMEvalConfig = field(default_factory=LMEvalConfig)

    def __post_init__(self):
        """Validate configuration."""
        if self.common.checkpoint_path == MISSING:
            raise ValueError("checkpoint_path is required")

        # Convert checkpoint_path to Path for validation
        checkpoint_path = Path(self.common.checkpoint_path)
        if not (checkpoint_path.exists() or checkpoint_path.parent.exists()):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
