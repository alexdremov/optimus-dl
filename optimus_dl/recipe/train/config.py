"""Training recipe configuration.

This module defines the configuration classes for the training recipe, including
all hyperparameters, component configurations, and training settings.
"""

from dataclasses import (
    dataclass,
    field,
)

from omegaconf import (
    II,
    MISSING,
)

from optimus_dl.core.registry import (
    RegistryConfig,
    RegistryConfigStrict,
)
from optimus_dl.modules.checkpoint import (
    CheckpointManagerConfig,
    LoadStrategy,
)
from optimus_dl.modules.criterion import CriterionConfig
from optimus_dl.modules.data import DataConfig
from optimus_dl.modules.distributed.config import DistributedConfig
from optimus_dl.modules.loggers import MetricsLoggerConfig
from optimus_dl.modules.model import ModelConfig
from optimus_dl.modules.model_transforms import ModelTransformConfig
from optimus_dl.modules.optim import OptimizationConfig
from optimus_dl.recipe.mixins.model_builder import ModelBuilderConfig
from optimus_dl.recipe.train.builders.criterion_builder import CriterionBuilderConfig
from optimus_dl.recipe.train.builders.data_builder import DataBuilderConfig
from optimus_dl.recipe.train.builders.optimizer_builder import OptimizerBuilderConfig
from optimus_dl.recipe.train.builders.scheduler_builder import SchedulerBuilderConfig
from optimus_dl.recipe.train.mixins.managers.evaluation_manager import EvaluatorConfig
from optimus_dl.recipe.train.mixins.managers.logger_manager import LoggerManagerConfig


@dataclass
class TrainRecipeConfig:
    """Configuration for training recipe common settings.

    This class contains all the common settings shared across training runs,
    including experiment metadata, logging frequency, checkpointing, evaluation,
    and distributed training settings.
    """

    # Exp metadata
    exp_name: str = field(default=MISSING, metadata={"description": "Experiment name"})
    exp_description: str | None = field(
        default=None, metadata={"description": "Experiment description"}
    )
    exp_tags: list[str] = field(
        default_factory=list, metadata={"description": "Experiment tags"}
    )
    log_freq: int = field(
        default=16, metadata={"description": "Frequency of train metrics logging"}
    )

    # Reproducibility
    seed: int = field(
        default=42, metadata={"description": "Seed to seed everything that's possible"}
    )
    data_seed: int = field(
        default=42,
        metadata={
            "description": "Seed to seed everything data-related. Will be different on each rank."
        },
    )
    deterministic: bool = field(
        default=True,
        metadata={"description": "If True, force deterministic algorithms in PyTorch."},
    )

    # Evaluation
    eval_iterations: int | None = field(
        default=None,
        metadata={
            "description": "Max number of iterations of validation data for every subset"
        },
    )
    eval_freq: int = field(
        default=100, metadata={"description": "Frequency of evaluations. Zero disables"}
    )
    eval_train_metrics: bool = field(
        default=False,
        metadata={
            "description": "Whether to compute MetricEngine metrics on training batches. This can be computationally expensive."
        },
    )
    # Checkpointing
    save_freq: int = field(
        default=II(".eval_freq"),
        metadata={
            "description": "Frequency of checkpoint savings. As eval_freq by default"
        },
    )
    output_path: str = field(
        default="${oc.env:PERSISTENT_PATH,'./outputs'}/${.exp_name}",
        metadata={"description": "Directory to dump checkpoints to"},
    )

    load_checkpoint: str | None = field(
        default=None,
        metadata={
            "description": "Path to checkpoint to load from, what to load from it is controlled by load_checkpoint_strategy"
        },
    )
    load_checkpoint_strategy: LoadStrategy = field(
        default_factory=LoadStrategy,
        metadata={"description": "Strategy what to load from the checkpoint"},
    )

    # Distributed
    use_gpu: bool = True
    distributed: DistributedConfig = field(
        default_factory=DistributedConfig,
        metadata={"description": "Distributed training configuration (GPU, TP, etc.)"},
    )


@dataclass
class TrainConfig(RegistryConfigStrict):
    """Complete training configuration.

    This is the root configuration class for training. It contains all component
    configurations (model, data, optimizer, etc.) and uses the registry system
    for flexible component selection.

    The configuration is hierarchical and supports OmegaConf interpolation for
    sharing values across components. The `args` field serves as a "scratch space"
    for high-level variables that can be referenced throughout the config.

    Example:
        ```python
        config = TrainConfig(
            _name="base",
            args={"batch_size": 64, "seq_len": 1024},
            model=ModelConfig(_name="llama", n_embd=512),
            optimization=OptimizationConfig(
                batch_size="${args.batch_size}",
                lr=1e-4,
            ),
        )

        ```"""

    args: dict = field(default_factory=dict)
    common: TrainRecipeConfig = field(default_factory=TrainRecipeConfig)

    model: ModelConfig = field(default=MISSING)
    data: DataConfig = field(default=MISSING)
    criterion: CriterionConfig = field(default=MISSING)
    optimization: OptimizationConfig = field(default=MISSING)
    lr_scheduler: RegistryConfig | None = field(default=None)

    # Metrics logging configuration
    loggers: list[MetricsLoggerConfig] | None = field(
        default=None, metadata={"description": "List of metrics logger configurations"}
    )
    metrics: list[dict] = field(
        default_factory=list,
        metadata={"description": "List of metric configurations for the MetricEngine"},
    )

    # Model transforms configuration
    model_transforms: list[ModelTransformConfig] = field(
        default_factory=list,
        metadata={
            "description": "List of model transforms to apply after model building"
        },
    )

    # Dependency Injection Configs
    model_builder: RegistryConfig = field(
        default_factory=lambda: ModelBuilderConfig(_name="base")
    )
    optimizer_builder: RegistryConfig = field(
        default_factory=lambda: OptimizerBuilderConfig(_name="base")
    )
    criterion_builder: RegistryConfig = field(
        default_factory=lambda: CriterionBuilderConfig(_name="base")
    )
    data_builder: RegistryConfig = field(
        default_factory=lambda: DataBuilderConfig(_name="base")
    )
    scheduler_builder: RegistryConfig = field(
        default_factory=lambda: SchedulerBuilderConfig(_name="base")
    )
    logger_manager: RegistryConfig = field(
        default_factory=lambda: LoggerManagerConfig(_name="base")
    )
    checkpoint_manager: RegistryConfig = field(
        default_factory=lambda: CheckpointManagerConfig(_name="base")
    )
    evaluator: RegistryConfig = field(
        default_factory=lambda: EvaluatorConfig(_name="base")
    )
