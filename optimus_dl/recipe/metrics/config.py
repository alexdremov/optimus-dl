"""Configuration for metrics evaluation recipe."""

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

from omegaconf import MISSING

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.checkpoint import (
    CheckpointManagerConfig,
)
from optimus_dl.modules.criterion import CriterionConfig
from optimus_dl.modules.data import DataConfig
from optimus_dl.modules.distributed.config import DistributedConfig
from optimus_dl.modules.loggers import MetricsLoggerConfig
from optimus_dl.modules.model import ModelConfig
from optimus_dl.modules.model_transforms import ModelTransformConfig
from optimus_dl.recipe.mixins.model_builder import ModelBuilderConfig
from optimus_dl.recipe.train.builders.criterion_builder import CriterionBuilderConfig
from optimus_dl.recipe.train.builders.data_builder import DataBuilderConfig
from optimus_dl.recipe.train.mixins.managers.evaluation_manager import (
    EvaluatorConfig,
)
from optimus_dl.recipe.train.mixins.managers.logger_manager import LoggerManagerConfig


@dataclass
class MetricsRecipeConfig:
    """Configuration for metrics evaluation recipe common settings."""

    # Experiment name
    name: str = field(
        default="metrics-eval",
        metadata={"description": "Experiment name for loggers"},
    )

    # Reproducibility
    seed: int = field(default=42)
    data_seed: int = field(default=42)

    # Output
    output_path: str = field(
        default="outputs/metrics",
        metadata={"description": "Base directory for outputs (logs, etc.)"},
    )

    # Checkpointing
    checkpoint_path: str | None = field(
        default=None,
        metadata={"description": "Path to checkpoint to load from"},
    )

    # Distributed
    use_gpu: bool = True
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Evaluation limit
    max_iterations: int | None = field(
        default=None,
        metadata={"description": "Max number of batches to process per dataset"},
    )

    # Tokenizer
    tokenizer: Any = field(
        default=None, metadata={"description": "Tokenizer configuration"}
    )


@dataclass
class MetricsConfig(RegistryConfigStrict):
    """Complete metrics evaluation configuration."""

    args: dict = field(default_factory=dict)
    common: MetricsRecipeConfig = field(default_factory=MetricsRecipeConfig)

    model: ModelConfig | None = field(default=None)
    data: DataConfig = field(default=MISSING)
    criterion: CriterionConfig = field(default=MISSING)

    # Metrics configuration for MetricEngine, mapped by dataset name
    metrics: dict[str, list[dict]] = field(
        default_factory=dict,
        metadata={"description": "Metric configurations mapped by dataset name"},
    )

    # Model transforms configuration
    model_transforms: list[ModelTransformConfig] = field(
        default_factory=list,
        metadata={"description": "List of model transforms to apply"},
    )

    # Logging
    loggers: list[MetricsLoggerConfig] | None = field(default=None)

    # Dependency Injection Configs
    model_builder: Any = field(default_factory=lambda: ModelBuilderConfig(_name="base"))
    criterion_builder: Any = field(
        default_factory=lambda: CriterionBuilderConfig(_name="base")
    )
    data_builder: Any = field(default_factory=lambda: DataBuilderConfig(_name="base"))
    checkpoint_manager: Any = field(
        default_factory=lambda: CheckpointManagerConfig(_name="base")
    )
    logger_manager: Any = field(
        default_factory=lambda: LoggerManagerConfig(_name="base")
    )
    evaluator: Any = field(default_factory=lambda: EvaluatorConfig(_name="base"))
