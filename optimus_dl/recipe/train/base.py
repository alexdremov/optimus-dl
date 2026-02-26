import logging
import time

import torch
from omegaconf import OmegaConf
from torch.optim import Optimizer
from tqdm.auto import trange

from optimus_dl.core.device import setup_device_and_collective
from optimus_dl.core.log import setup_logging
from optimus_dl.core.registry import build as build_component
from optimus_dl.core.seed import set_seed
from optimus_dl.modules.checkpoint import CheckpointManager
from optimus_dl.modules.criterion import BaseCriterion
from optimus_dl.modules.metrics import (
    compute_meters,
    log_event_end,
    log_event_start,
    meters_group,
    reset_meters,
    step_meters,
)
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.recipe.train.builders import (
    CriterionBuilder,
    DataBuilder,
    ModelBuilder,
    OptimizerBuilder,
    SchedulerBuilder,
)
from optimus_dl.recipe.train.config import TrainConfig
from optimus_dl.recipe.train.mixins.execution import (
    TrainingContextMixin,
    TrainingInterruptionMixin,
    TrainingIterationMixin,
)
from optimus_dl.recipe.train.mixins.managers import (
    Evaluator,
    LoggerManager,
)

from . import register_train_recipe

logger = logging.getLogger(__name__)


@register_train_recipe("base", TrainConfig)
class TrainRecipe(
    TrainingContextMixin,
    TrainingIterationMixin,
    TrainingInterruptionMixin,
):
    """Main training recipe that orchestrates all training components.

    This class uses composition to coordinate specialized builders and managers:

    - ModelBuilder: Builds models and applies transforms
    - OptimizerBuilder: Builds optimizers
    - CriterionBuilder: Builds loss criteria
    - DataBuilder: Builds train/eval data pipelines
    - SchedulerBuilder: Builds learning rate schedulers
    - LoggerManager: Handles metrics logging setup and operations
    - CheckpointManager: Manages checkpoint saving and loading
    - Evaluator: Handles evaluation runs and metrics

    It inherits from training logic mixins for the core loop execution:

    - TrainingContextMixin: Sets up training context (AMP, scaler, etc.)
    - TrainingIterationMixin: Orchestrates complete training iterations
    - TrainingInterruptionMixin: Handles training interruptions and errors
    """

    cfg: TrainConfig

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        # Initialize components via composition using registry for dependency injection
        self.model_builder = build_component(
            "model_builder",
            cfg.model_builder,
            cast_to=ModelBuilder,
            model_transforms=cfg.model_transforms,
        )
        assert self.model_builder is not None, "Model builder not initialized"
        self.optimizer_builder = build_component(
            "optimizer_builder",
            cfg.optimizer_builder,
            cast_to=OptimizerBuilder,
            optimization_config=cfg.optimization,
        )
        assert self.optimizer_builder is not None, "Optimizer builder not initialized"
        self.criterion_builder = build_component(
            "criterion_builder",
            cfg.criterion_builder,
            cast_to=CriterionBuilder,
            criterion_config=cfg.criterion,
        )
        assert self.criterion_builder is not None, "Criterion builder not initialized"
        self.data_builder = build_component(
            "data_builder",
            cfg.data_builder,
            cast_to=DataBuilder,
            data_config=cfg.data,
            data_seed=cfg.common.data_seed,
        )
        assert self.data_builder is not None, "Data builder not initialized"
        self.scheduler_builder = build_component(
            "scheduler_builder",
            cfg.scheduler_builder,
            cast_to=SchedulerBuilder,
            lr_scheduler_config=cfg.lr_scheduler,
            optimization_config=cfg.optimization,
        )
        assert self.scheduler_builder is not None, "Scheduler builder not initialized"
        self.logger_manager = build_component(
            "logger_manager",
            cfg.logger_manager,
            cast_to=LoggerManager,
            loggers_config=cfg.loggers,
        )
        assert self.logger_manager is not None, "Logger manager not initialized"
        self.checkpoint_manager = build_component(
            "checkpoint_manager",
            cfg.checkpoint_manager,
            cast_to=CheckpointManager,
        )
        assert self.checkpoint_manager is not None, "Checkpoint manager not initialized"
        self.evaluator = build_component(
            "evaluator",
            cfg.evaluator,
            cast_to=Evaluator,
            eval_freq=cfg.common.eval_freq,
            eval_iterations=cfg.common.eval_iterations,
        )
        assert self.evaluator is not None, "Evaluator not initialized"

        # Initialize training logic mixins
        TrainingContextMixin.__init__(self, cfg.optimization)
        TrainingIterationMixin.__init__(self, cfg.optimization, cfg.common.log_freq)
        TrainingInterruptionMixin.__init__(
            self,
            cfg.common.save_freq,
            cfg.common.output_path,
            self.save_checkpoint,  # Pass the checkpoint method as callback
        )

        self.set_exp_name()
        self.validate_config()

    # Delegate methods
    def build_model(self, *args, **kwargs) -> BaseModel:
        """Delegate to ModelBuilder."""
        return self.model_builder.build_model(*args, **kwargs)

    def build_optimizer(self, *args, **kwargs) -> Optimizer:
        """Delegate to OptimizerBuilder."""
        return self.optimizer_builder.build_optimizer(*args, **kwargs)

    def build_lr_scheduler(self, *args, **kwargs):
        """Delegate to SchedulerBuilder."""
        return self.scheduler_builder.build_lr_scheduler(*args, **kwargs)

    def build_criterion(self, *args, **kwargs) -> BaseCriterion:
        """Delegate to CriterionBuilder."""
        return self.criterion_builder.build_criterion(*args, **kwargs)

    def build_train_data(self, *args, **kwargs):
        """Delegate to DataBuilder for training data."""
        return self.data_builder.build_train_data(*args, **kwargs)

    def build_eval_data(self, *args, **kwargs):
        """Delegate to DataBuilder for evaluation data."""
        return self.data_builder.build_eval_data(*args, **kwargs)

    def build_loggers(self, *args, **kwargs):
        """Delegate to LoggerManager for building loggers."""
        return self.logger_manager.build_loggers(*args, **kwargs)

    def setup_loggers(self, experiment_name: str, full_config: dict | None = None):
        """Setup logging with experiment configuration.

        Args:
            experiment_name: Name of the current experiment.
            full_config: Full configuration dictionary. If None, uses self.cfg.
        """
        if full_config is None:
            full_config = self.cfg if hasattr(self.cfg, "__dict__") else dict(self.cfg)
        return self.logger_manager.setup_loggers(experiment_name, full_config)

    def log_metrics_to_loggers(self, *args, **kwargs):
        """Delegate metrics logging to LoggerManager."""
        return self.logger_manager.log_metrics_to_loggers(*args, **kwargs)

    def close_loggers(self, *args, **kwargs):
        """Delegate logger cleanup to LoggerManager."""
        return self.logger_manager.close_loggers(*args, **kwargs)

    def save_checkpoint_if_needed(self, *args, **kwargs):
        """Check save frequency and delegate to CheckpointManager."""
        config_dict = self.cfg if hasattr(self.cfg, "__dict__") else dict(self.cfg)
        kwargs["full_config"] = config_dict
        kwargs["checkpoint_path"] = self.cfg.common.output_path
        kwargs["save_freq"] = self.cfg.common.save_freq
        kwargs["logger_manager"] = self.logger_manager
        return self.checkpoint_manager.save_checkpoint_if_needed(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs):
        """Save a checkpoint via CheckpointManager."""
        config_dict = self.cfg if hasattr(self.cfg, "__dict__") else dict(self.cfg)
        kwargs["full_config"] = config_dict
        if "checkpoint_path" not in kwargs:
            kwargs["checkpoint_path"] = self.cfg.common.output_path
        if "logger_manager" not in kwargs:
            kwargs["logger_manager"] = self.logger_manager
        return self.checkpoint_manager.save_checkpoint(*args, **kwargs)

    def load_checkpoint_if_exists(self, *args, **kwargs):
        """Try to resume from latest checkpoint in output path."""
        if "checkpoint_path" not in kwargs:
            kwargs["checkpoint_path"] = self.cfg.common.output_path
        if "logger_manager" not in kwargs:
            kwargs["logger_manager"] = self.logger_manager
        return self.checkpoint_manager.load_checkpoint_if_exists(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        """Load a specific checkpoint."""
        if "logger_manager" not in kwargs:
            kwargs["logger_manager"] = self.logger_manager
        return self.checkpoint_manager.load_checkpoint(*args, **kwargs)

    def run_evaluation_if_needed(self, *args, **kwargs):
        """Check eval frequency and run evaluation via Evaluator."""
        return self.evaluator.run_evaluation_if_needed(*args, **kwargs)

    def set_exp_name(self):
        """Set experiment name based on config or environment variables."""
        if not OmegaConf.is_missing(self.cfg.common, "exp_name"):
            return
        exp_name = f"run-{self.cfg.model._name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        self.cfg.common.exp_name = exp_name
        logger.info(f"Experiment name set to: {self.cfg.common.exp_name}")

    def validate_config(self) -> None:
        """Validate configuration before training starts."""
        # Required fields
        assert self.cfg.model is not None, "Model configuration is required"
        assert self.cfg.data is not None, "Data configuration is required"
        assert self.cfg.criterion is not None, "Criterion configuration is required"
        assert (
            self.cfg.optimization is not None
        ), "Optimization configuration is required"

        # Training parameters
        assert self.cfg.optimization.iterations > 0, "iterations must be positive"
        assert self.cfg.optimization.acc_steps > 0, "acc_steps must be positive"
        assert self.cfg.common.log_freq > 0, "log_freq must be positive"

        if self.cfg.common.save_freq > 0:
            assert (
                self.cfg.common.output_path
            ), "output_path required when save_freq > 0"

        logger.info("Configuration validation passed")

    def setup_context(self):
        """Setup global training context (precision, etc.)."""
        set_seed(self.cfg.common.seed, deterministic=self.cfg.common.deterministic)
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            if hasattr(torch.backends.cudnn, "conv"):
                torch.backends.cudnn.conv.fp32_precision = "ieee"

    def run(self):
        """Run the complete training pipeline."""
        self.setup_context()
        is_restart = self.checkpoint_manager.is_restart(self.cfg.common.output_path)

        with meters_group("init"):
            log_event_start("perf/init")
            logger.info(f"Using output path : {self.cfg.common.output_path}")
            logger.info(self.cfg)

            # Setup device and distributed collective
            device, collective = setup_device_and_collective(
                use_gpu=self.cfg.common.use_gpu, config=self.cfg.common.distributed
            )

            logger.info(
                "Setting up console logging. Will log from master only from now."
            )
            if not collective.is_master:
                setup_logging(logging.WARNING)

            model: BaseModel = self.build_model(
                model_config=self.cfg.model,
                collective=collective,
                is_restart=is_restart,
                checkpoint_manager=self.checkpoint_manager,
            )

            optimizer: Optimizer = self.build_optimizer(model.make_parameter_groups())
            lr_scheduler = self.build_lr_scheduler(optimizer)
            criterion: BaseCriterion = self.build_criterion(collective=collective)

            # Setup training context (AMP, scaler, etc.) using recipe mixin
            training_context = self.setup_training_context(device)

            try:
                train_datapipeline = self.build_train_data(
                    device=device, collective=collective
                )
                assert (
                    train_datapipeline is not None
                ), "Train data pipeline not initialized"
                eval_datapipeline = self.build_eval_data(
                    device=device, collective=collective
                )
                data_loaders = {
                    "train": train_datapipeline.dataloader,
                    # eval dataloader may be not restored
                }
            except Exception as e:
                logger.error(f"Failed to build data pipelines: {e}")
                raise

            model = model.to(device)

            # cannot be after checkpoint load as may erase the start event
            log_event_end("perf/init")

            # Try to resume from checkpoint in output paths
            start_iteration, metadata = self.load_checkpoint_if_exists(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                data_loaders=data_loaders,
                collective=collective,
                data_sources=train_datapipeline.datasets,
            )
            if is_restart:
                # cases when training run but did not produce any artifacts is
                # indistinguishable from the case when training is not started at all
                assert metadata is not None, "Misaligned is_restart flag"

            logger.info(f"Considering this run as {is_restart = }")
            if not is_restart and self.cfg.common.load_checkpoint is not None:
                # if checkpoint from output path was not loaded, we are sure that this launch is not
                # re-scheduling / preemption re-start, so we can try loading model from load_checkpoint
                metadata = self.load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    data_loaders=data_loaders,
                    data_sources=train_datapipeline.datasets,
                    collective=collective,
                    load_strategy=self.cfg.common.load_checkpoint_strategy,
                    checkpoint_path=self.cfg.common.load_checkpoint,
                )
                start_iteration = metadata["iteration"] + 1
                logger.info(
                    "Loaded checkpoint from "
                    f"checkpoint_path = {self.cfg.common.load_checkpoint} path with "
                    f"load_strategy = {self.cfg.common.load_checkpoint_strategy} "
                    f"with {start_iteration = }"
                )

            if collective.is_master:
                self.build_loggers()
                self.setup_loggers(self.cfg.common.exp_name)

        init_metrics = compute_meters(
            group_name="init",
            aggregate=True,
            collective=collective,
        )
        if collective.is_local_master:
            self.log_metrics_to_loggers(init_metrics, start_iteration, "init")

        common_chkp_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "collective": collective,
            "lr_scheduler": lr_scheduler,
            "data_loaders": data_loaders,
            "data_sources": train_datapipeline.datasets,
            "grad_scaler": training_context["scaler"],
        }

        train_data_iter = iter(train_datapipeline.dataloader)

        # Setup training metric engine if config exists
        train_metric_engine = None
        if self.cfg.metrics and "train" in self.cfg.metrics:
            from optimus_dl.modules.metrics.engine import MetricEngine

            train_metric_engine = MetricEngine("train", self.cfg.metrics["train"])

        collective.barrier()
        logger.info("All ranks are ready")

        pbar = trange(
            start_iteration,
            self.cfg.optimization.iterations + 1,
            initial=start_iteration,
            total=self.cfg.optimization.iterations,
            miniters=self.cfg.common.log_freq,
            maxinterval=1000000,
            disable=not collective.is_local_master,
            smoothing=0,
        )
        for iteration in pbar:
            try:
                # Execute one training iteration using recipe mixin
                self.run_training_iteration(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_data_iter=train_data_iter,
                    training_context=training_context,
                    lr_scheduler=lr_scheduler,
                    metric_engine=train_metric_engine,
                )

                with meters_group("train") as should_log:
                    if should_log:
                        # Get aggregated metrics for progress bar
                        current_metrics = compute_meters(
                            "train",
                            aggregate=True,
                            collective=collective,
                        )
                        if train_metric_engine:
                            current_metrics = train_metric_engine.compute(
                                current_metrics
                            )

                        if collective.is_local_master:
                            pbar.set_postfix(current_metrics, refresh=False)

                        # Log metrics to all configured loggers
                        if collective.is_master:
                            self.log_metrics_to_loggers(
                                current_metrics, iteration, "train"
                            )

                step_meters("train")  # Step the metrics logging iteration counter
                reset_meters(
                    "train"
                )  # Reset metrics after logging (keep metrics with reset=False)
                with training_context["amp_ctx"]:
                    metrics = self.run_evaluation_if_needed(
                        iteration=iteration,
                        model=model,
                        criterion=criterion,
                        eval_data={
                            k: v.dataloader
                            for k, v in eval_datapipeline.items()
                            if v is not None
                        },
                        collective=collective,
                        all_metrics_configs=self.cfg.metrics,
                    )
                if metrics and collective.is_master:
                    for eval_name, eval_metrics in metrics.items():
                        self.log_metrics_to_loggers(eval_metrics, iteration, eval_name)

                self.save_checkpoint_if_needed(
                    iteration=iteration,
                    **common_chkp_kwargs,
                )

            except KeyboardInterrupt:
                self.handle_training_interruption(
                    iteration=iteration,
                    **common_chkp_kwargs,
                )
                break
            except Exception as e:
                logger.error(f"Training step failed at iteration {iteration}: {e}")
                raise

        # Close loggers at the end of training
        if collective.is_master:
            self.close_loggers()
