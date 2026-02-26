"""Evaluation mixin for evaluation functionality."""

import logging
import time
from dataclasses import (
    dataclass,
)
from typing import Any

import torch
from tqdm.auto import tqdm

from optimus_dl.core.profile import measured_next
from optimus_dl.core.registry import (
    RegistryConfig,
    make_registry,
)
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
from optimus_dl.modules.model.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig(RegistryConfig):
    """Configuration for the Evaluator."""

    pass


class Evaluator:
    """Manager for running periodic evaluations during training.

    Handles iterating over validation datasets, computing loss and other metrics,
    and aggregating results across distributed ranks.

    Args:
        cfg: Evaluator configuration.
        eval_freq: Frequency of evaluation runs (in iterations).
        eval_iterations: Max number of batches to process per evaluation dataset.
            If None, processes the entire dataset.
    """

    def __init__(
        self,
        cfg: EvaluatorConfig,
        eval_freq: int = 0,
        eval_iterations: int | None = None,
        **kwargs: Any,
    ):
        self.cfg = cfg
        self.eval_freq = eval_freq
        self.eval_iterations = eval_iterations

    def run_evaluation_if_needed(
        self,
        iteration: int,
        model: BaseModel,
        criterion: BaseCriterion,
        eval_data: dict[str, Any],
        collective: Any = None,
        all_metrics_configs: dict[str, list[dict]] | None = None,
    ) -> None | dict:
        """Run evaluation if the current iteration matches the frequency.

        Args:
            iteration: Current training step.
            model: The model to evaluate.
            criterion: The loss function.
            eval_data: Dictionary mapping dataset names to dataloaders.
            collective: Distributed collective for metric aggregation.
            all_metrics_configs: Root metrics configuration from TrainConfig.

        Returns:
            Dictionary of computed metrics if evaluation ran, else None.
        """
        if self.eval_freq <= 0 or iteration % self.eval_freq != 0:
            return None

        try:
            return self.run_evaluation(
                model=model,
                criterion=criterion,
                eval_data_dict=eval_data,
                max_iterations=self.eval_iterations,
                collective=collective,
                all_metrics_configs=all_metrics_configs,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None

    def run_evaluation(
        self,
        model: BaseModel,
        criterion: BaseCriterion,
        eval_data_dict: dict,
        max_iterations: int | None = None,
        collective: Any = None,
        all_metrics_configs: dict[str, list[dict]] | None = None,
        metrics_prefix: str = "eval",
        show_progress: bool = False,
    ):
        """Execute the evaluation loop for all provided datasets.

        Sets the model to eval mode, disables gradients, and runs the forward pass
        for each batch. Metrics are aggregated globally.

        Args:
            model: Model to evaluate.
            criterion: Loss function.
            eval_data_dict: Dictionary of {name: dataloader/DataPipeline}.
            max_iterations: Limit on number of batches.
            collective: Distributed collective.
            all_metrics_configs: Root metrics configuration mapping dataset names to configs.
            metrics_prefix: Prefix for metric groups (e.g., "eval" or "metrics").
            show_progress: Whether to show a progress bar.

        Returns:
            Nested dictionary of results: {dataset_name: {metric_name: value}}.
        """
        model.eval()
        total_metrics = {}
        all_metrics_configs = all_metrics_configs or {}

        for eval_name, eval_data in eval_data_dict.items():
            logger.info(f"Running evaluation {eval_name}")

            # Handle both raw dataloader and DataPipeline object
            dataloader = getattr(eval_data, "dataloader", eval_data)

            engine = None
            requested_protocols = None
            dataset_metrics = all_metrics_configs.get(eval_name)
            if dataset_metrics:
                from optimus_dl.modules.metrics.engine import MetricEngine

                engine = MetricEngine(f"{metrics_prefix}/{eval_name}", dataset_metrics)
                requested_protocols = engine.required_external_protocols

            with (
                torch.no_grad(),
                metrics_group(
                    f"{metrics_prefix}/{eval_name}", log_freq=1, force_recreate=True
                ),
            ):
                log_event_start("perf/total_run")
                start_time = time.perf_counter()

                eval_iter = iter(dataloader)
                iterations = 0

                pbar = None
                if show_progress:
                    pbar = tqdm(
                        desc=f"Eval {eval_name}",
                        disable=collective is not None
                        and not collective.is_local_master,
                        unit="batch",
                        total=max_iterations,
                    )

                try:
                    while max_iterations is None or iterations < max_iterations:
                        log_event_occurence("perf/full_iteration")

                        elapsed_batch_get, batch = measured_next(eval_iter)
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
                        log_averaged(
                            "perf/batch_get",
                            elapsed_batch_get,
                        )

                        iterations += 1
                        if pbar:
                            pbar.update(1)

                        # Step metrics for each evaluation iteration
                        step_metrics(f"{metrics_prefix}/{eval_name}")

                except StopIteration:
                    pass
                finally:
                    if pbar:
                        pbar.close()

                total_time = time.perf_counter() - start_time
                log_event_end("perf/total_run")

            eval_metrics = compute_metrics(
                f"{metrics_prefix}/{eval_name}",
                aggregate=True,
                collective=collective,
            )

            if engine:
                eval_metrics = engine.compute(eval_metrics)

            # Add basic performance stats
            eval_metrics["perf/total_run_ms"] = total_time * 1000
            if iterations > 0:
                eval_metrics["perf/ms_per_batch"] = (total_time / iterations) * 1000

            logger.info(f"Finished eval {eval_name}: {eval_metrics}")
            total_metrics[eval_name] = eval_metrics
        return total_metrics


_, register_evaluator, build_evaluator = make_registry("evaluator", Evaluator)
register_evaluator("base", EvaluatorConfig)(Evaluator)
