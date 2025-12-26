"""Training recipe configuration.

This module defines the configuration classes for the training recipe, including
all hyperparameters, component configurations, and training settings.
"""

from dataclasses import dataclass, field

from omegaconf import II, MISSING

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.checkpoint import CheckpointManagerConfig, LoadStrategy
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

    Attributes:
        exp_name: Unique name for this experiment. Used for organizing outputs
            and logs. If MISSING, will be auto-generated.
        exp_description: Optional description of the experiment.
        exp_tags: List of tags for organizing experiments.
        log_freq: How often (in iterations) to log training metrics.
        seed: Random seed for reproducibility. Seeds PyTorch, NumPy, Python RNG.
        data_seed: Separate seed for data-related randomness. Will be different
            on each rank for data diversity in distributed training.
        eval_iterations: Maximum number of validation iterations per evaluation
            run. If None, evaluates on entire validation set.
        eval_freq: How often (in iterations) to run evaluation. Set to 0 to disable.
        save_freq: How often (in iterations) to save checkpoints. Defaults to
            eval_freq if not specified.
        output_path: Directory where checkpoints and logs are saved. Supports
            environment variable interpolation (e.g., ${oc.env:OUTPUT_DIR}).
        load_checkpoint: Path to a checkpoint to load at startup. If None, training
            starts from scratch or resumes from latest checkpoint in output_path.
        load_checkpoint_strategy: Strategy for what to load from the checkpoint
            (model weights, optimizer state, etc.). See LoadStrategy for details.
        use_gpu: Whether to use GPU if available. If False, uses CPU.
        distributed: Configuration for distributed training mesh.
    """

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

    # Distributed
    use_gpu: bool = True
    distributed: DistributedConfig = field(
        default_factory=DistributedConfig,
        metadata={"help": "Distributed training configuration (GPU, TP, etc.)"},
    )


@dataclass
class TrainConfig(RegistryConfig):
    """Complete training configuration.

    This is the root configuration class for training. It contains all component
    configurations (model, data, optimizer, etc.) and uses the registry system
    for flexible component selection.

    The configuration is hierarchical and supports OmegaConf interpolation for
    sharing values across components. The `args` field serves as a "scratch space"
    for high-level variables that can be referenced throughout the config.

    Attributes:
        args: Dictionary for high-level variables that can be referenced via
            interpolation (e.g., ${args.batch_size}). This ensures consistency
            across components.
        common: Common training settings (logging, checkpointing, etc.).
        model: Model configuration. Must specify `_name` to select model type.
        data: Data pipeline configuration (sources, transforms, etc.).
        criterion: Loss function configuration.
        optimization: Optimization settings (batch size, learning rate, etc.).
        lr_scheduler: Learning rate scheduler configuration. If None, no scheduler.
        loggers: List of metrics logger configurations (WandB, TensorBoard, etc.).
        model_transforms: List of model transforms to apply (DDP, FSDP, compile, etc.).
        model_builder: Configuration for the model builder component.
        optimizer_builder: Configuration for the optimizer builder component.
        criterion_builder: Configuration for the criterion builder component.
        data_builder: Configuration for the data builder component.
        scheduler_builder: Configuration for the scheduler builder component.
        logger_manager: Configuration for the logger manager component.
        checkpoint_manager: Configuration for the checkpoint manager component.
        evaluator: Configuration for the evaluator component.

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
