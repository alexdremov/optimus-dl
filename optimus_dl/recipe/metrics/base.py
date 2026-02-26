"""Metrics evaluation recipe using the internal MetricEngine."""

import logging
from typing import (
    Any,
)

from omegaconf import OmegaConf

from optimus_dl.core.device import setup_device_and_collective
from optimus_dl.core.registry import build as build_component
from optimus_dl.core.seed import set_seed
from optimus_dl.modules.checkpoint import CheckpointManager
from optimus_dl.modules.criterion import BaseCriterion
from optimus_dl.recipe.train.builders import (
    CriterionBuilder,
    DataBuilder,
    ModelBuilder,
)
from optimus_dl.recipe.train.mixins.managers.evaluation_manager import (
    Evaluator,
    build_evaluator,
)
from optimus_dl.recipe.train.mixins.managers.logger_manager import (
    LoggerManager,
    build_logger_manager,
)

from .config import MetricsConfig

logger = logging.getLogger(__name__)


class MetricsRecipe:
    """Recipe for evaluating models using the internal Metrics system.

    Handles building the model, data pipelines, and executing the evaluation loop
    for all provided datasets, reporting metrics via the MetricEngine.
    """

    def __init__(self, cfg: MetricsConfig):
        self.cfg = cfg

        # Initialize builders via composition
        self.model_builder = build_component(
            "model_builder",
            cfg.model_builder,
            cast_to=ModelBuilder,
            model_transforms=cfg.model_transforms,
        )
        self.data_builder = build_component(
            "data_builder",
            cfg.data_builder,
            cast_to=DataBuilder,
            data_config=cfg.data,
            data_seed=cfg.common.data_seed,
            tokenizer_config=cfg.common.tokenizer,
        )
        self.criterion_builder = build_component(
            "criterion_builder",
            cfg.criterion_builder,
            cast_to=CriterionBuilder,
            criterion_config=cfg.criterion,
        )
        self.checkpoint_manager = build_component(
            "checkpoint_manager",
            cfg.checkpoint_manager,
            cast_to=CheckpointManager,
        )
        self.logger_manager: LoggerManager = build_logger_manager(
            cfg.logger_manager, loggers_config=cfg.loggers
        )
        self.evaluator: Evaluator = build_evaluator(
            cfg.evaluator,
            eval_iterations=cfg.common.max_iterations,
        )
        self.tokenizer = None

    def run(self) -> dict[str, dict[str, Any]]:
        """Run the complete evaluation pipeline."""
        set_seed(self.cfg.common.seed)

        # Setup device and distributed collective
        device, collective = setup_device_and_collective(
            use_gpu=self.cfg.common.use_gpu, config=self.cfg.common.distributed
        )

        logger.info(f"Starting Metrics Evaluation on {device}")

        # 1. Build Model
        # Try loading from checkpoint if provided, else build from model config
        if self.cfg.common.checkpoint_path:
            logger.info(
                f"Loading model from checkpoint: {self.cfg.common.checkpoint_path}"
            )
            model, _ = self.checkpoint_manager.build_model_from_checkpoint(
                checkpoint_path=self.cfg.common.checkpoint_path, device=device
            )
        else:
            assert (
                self.cfg.model is not None
            ), "Model config required if no checkpoint path provided"
            model = self.model_builder.build_model(
                model_config=self.cfg.model,
                collective=collective,
            )

        model.eval()
        model.to(device)

        # 2. Build Criterion
        criterion: BaseCriterion = self.criterion_builder.build_criterion(
            collective=collective
        )

        # 3. Build Data
        eval_datapipeline = self.data_builder.build_eval_data(
            device=device, collective=collective
        )

        # 4. Setup Loggers
        if collective.is_master:
            self.logger_manager.build_loggers()
            self.logger_manager.setup_loggers(
                self.cfg.common.name, OmegaConf.to_container(self.cfg, resolve=True)
            )

        all_results = {}

        try:
            # 5. Run evaluation using Evaluator component
            all_results = self.evaluator.run_evaluation(
                model=model,
                criterion=criterion,
                eval_data_dict=eval_datapipeline,
                max_iterations=self.cfg.common.max_iterations,
                collective=collective,
                all_metrics_configs=self.cfg.metrics,
                metrics_prefix="metrics",
                show_progress=True,
            )

            # 6. Log results to loggers
            if collective.is_master:
                for eval_name, eval_metrics in all_results.items():
                    self.logger_manager.log_metrics_to_loggers(
                        eval_metrics, step=0, group=f"eval/{eval_name}"
                    )

        finally:
            if collective.is_master:
                self.logger_manager.close_loggers()

        return all_results
