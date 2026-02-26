"""Metrics evaluation recipe using the internal MetricEngine."""

import logging
from typing import (
    Any,
)

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from optimus_dl.core.device import setup_device_and_collective
from optimus_dl.core.profile import measured_next
from optimus_dl.core.registry import build as build_component
from optimus_dl.core.seed import set_seed
from optimus_dl.modules.checkpoint import CheckpointManager
from optimus_dl.modules.criterion import BaseCriterion
from optimus_dl.modules.metrics import (
    compute_metrics,
    log_averaged,
    log_event_end,
    log_event_occurence,
    log_event_start,
    log_summed,
    metrics_group,
    step_metrics,
)
from optimus_dl.recipe.train.builders import (
    CriterionBuilder,
    DataBuilder,
    ModelBuilder,
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
        self.tokenizer = None

    def run(self) -> dict[str, dict[str, Any]]:
        """Run the complete evaluation pipeline."""
        set_seed(self.cfg.common.seed)

        # Setup device and distributed collective
        device, collective = setup_device_and_collective(
            use_gpu=self.cfg.common.use_gpu, config=self.cfg.common.distributed
        )

        logger.info(f"Starting Metrics Evaluation on {device}")

        # 0. Build Tokenizer
        from optimus_dl.modules.tokenizer import build_tokenizer

        self.tokenizer = build_tokenizer(self.cfg.common.tokenizer)

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
        self.logger_manager.build_loggers()
        self.logger_manager.setup_loggers(
            self.cfg.common.name, OmegaConf.to_container(self.cfg, resolve=True)
        )

        all_results = {}

        try:
            # 5. Evaluation Loop
            for eval_name, eval_data in eval_datapipeline.items():
                logger.info(f"Evaluating dataset: {eval_name}")

                # Setup MetricEngine for this dataset
                engine = None
                requested_protocols = None
                dataset_metrics = self.cfg.metrics.get(eval_name)
                if dataset_metrics:
                    from optimus_dl.modules.metrics.engine import MetricEngine

                    engine = MetricEngine(f"metrics/{eval_name}", dataset_metrics)
                    requested_protocols = engine.required_external_protocols

                with (
                    torch.no_grad(),
                    metrics_group(
                        f"metrics/{eval_name}", log_freq=1, force_recreate=True
                    ),
                ):
                    log_event_start("perf/total_run")

                    eval_iter = iter(eval_data.dataloader)
                    iterations = 0
                    max_iterations = self.cfg.common.max_iterations

                    pbar = tqdm(
                        desc=f"Eval {eval_name}",
                        disable=not collective.is_local_master,
                        unit="batch",
                    )

                    try:
                        while max_iterations is None or iterations < max_iterations:
                            log_event_occurence("perf/full_iteration")

                            elapsed_batch_get, batch = measured_next(eval_iter)

                            # Forward through criterion
                            loss, exposed = criterion(
                                model, batch, requested_protocols=requested_protocols
                            )

                            if engine:
                                computed_data = exposed.copy()
                                computed_data["loss"] = loss
                                engine.update(
                                    data=dict(model=model, batch=batch),
                                    computed_data=computed_data,
                                )

                            log_summed("num_batches", lambda: 1)
                            log_averaged("perf/batch_get", elapsed_batch_get)

                            iterations += 1
                            pbar.update(1)
                            step_metrics(f"metrics/{eval_name}")

                    except StopIteration:
                        pass
                    finally:
                        pbar.close()

                    log_event_end("perf/total_run")

                # Finalize and Aggregate
                eval_metrics = compute_metrics(
                    f"metrics/{eval_name}",
                    aggregate=True,
                    collective=collective,
                )

                if engine:
                    eval_metrics = engine.compute(eval_metrics)

                logger.info(f"Results for {eval_name}: {eval_metrics}")
                all_results[eval_name] = eval_metrics

                # Log to loggers
                self.logger_manager.log_metrics_to_loggers(
                    eval_metrics, step=0, group=f"eval/{eval_name}"
                )

        finally:
            self.logger_manager.close_loggers()

        return all_results
