from dataclasses import dataclass, field

from omegaconf import II, MISSING

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.checkpoint import CheckpointManagerConfig, LoadStrategy
from optimus_dl.modules.criterion import CriterionConfig
from optimus_dl.modules.data import DataConfig
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
    # Exp metadata
    exp_name: str = field(default=MISSING, metadata={"help": "Experiment name"})
    exp_description: str | None = field(
        default=None, metadata={"help": "Experiment description"}
    )
    exp_tags: list[str] = field(
        default_factory=list, metadata={"help": "Experiment tags"}
    )
    log_freq: int = field(
        default=16, metadata={"help": "Frequency of train metrics logging"}
    )

    # Reproducibility
    seed: int = field(
        default=42, metadata={"help": "Seed to seed everything that's possible"}
    )
    data_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to seed everything data-related. Will be different on each rank."
        },
    )

    # Evaluation
    eval_iterations: int | None = field(
        default=None,
        metadata={
            "help": "Max number of iterations of validation data for every subset"
        },
    )
    eval_freq: int = field(
        default=100, metadata={"help": "Frequency of evaluations. Zero disables"}
    )
    # Checkpointing
    save_freq: int = field(
        default=II(".eval_freq"),
        metadata={"help": "Frequency of checkpoint savings. As eval_freq by default"},
    )
    output_path: str = field(
        default="${oc.env:PERSISTENT_PATH,'./outputs'}/${.exp_name}",
        metadata={"help": "Directory to dump checkpoints to"},
    )

    load_checkpoint: str | None = field(
        default=None,
        metadata={
            "help": "Path to checkpoint to load from, what to load from it is controlled by load_checkpoint_strategy"
        },
    )
    load_checkpoint_strategy: LoadStrategy = field(
        default_factory=LoadStrategy,
        metadata={"help": "Strategy what to load from the checkpoint"},
    )

    use_gpu: bool = field(
        default=True, metadata={"help": "Use GPU (CUDA / MPS) if available"}
    )


@dataclass
class TrainConfig(RegistryConfig):
    args: dict = field(default_factory=dict)
    common: TrainRecipeConfig = field(default_factory=TrainRecipeConfig)

    model: ModelConfig = field(default=MISSING)
    data: DataConfig = field(default=MISSING)
    criterion: CriterionConfig = field(default=MISSING)
    optimization: OptimizationConfig = field(default=MISSING)
    lr_scheduler: RegistryConfig | None = field(default=None)

    # Metrics logging configuration
    loggers: list[MetricsLoggerConfig] | None = field(
        default=None, metadata={"help": "List of metrics logger configurations"}
    )

    # Model transforms configuration
    model_transforms: list[ModelTransformConfig] = field(
        default_factory=list,
        metadata={"help": "List of model transforms to apply after model building"},
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
