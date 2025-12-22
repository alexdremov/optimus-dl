"""Evaluation mixin for evaluation functionality."""

import logging
from dataclasses import dataclass
from typing import Any

import torch

from optimus_dl.core.profile import measured_next
from optimus_dl.core.registry import RegistryConfig, make_registry
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
    pass


class Evaluator:
    """Mixin for evaluation functionality."""

    def __init__(
        self,
        cfg: EvaluatorConfig,
        eval_freq: int = 0,
        eval_iterations: int | None = None,
        **kwargs,
    ):
        self.eval_freq = eval_freq
        self.eval_iterations = eval_iterations

    def run_evaluation_if_needed(
        self,
        iteration: int,
        model: BaseModel,
        criterion: BaseCriterion,
        eval_data: dict[str, Any],
        collective: Any = None,
    ) -> None | dict:
        """Run evaluation if iteration matches eval_freq."""
        if self.eval_freq <= 0 or iteration % self.eval_freq != 0:
            return None

        try:
            return self.run_evaluation(
                model=model,
                criterion=criterion,
                eval_data_dict=eval_data,
                max_iterations=self.eval_iterations,
                collective=collective,
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
    ):
        """Run evaluation on all eval datasets and optionally log metrics.

        Args:
            model: Model to evaluate
            criterion: Loss criterion for evaluation
            eval_data_dict: Dictionary of evaluation datasets
            max_iterations: Maximum iterations per dataset
            loggers: List of loggers to send metrics to (optional)
            iteration: Current training iteration for logging
            collective: Collective for distributed aggregation (optional)
        """
        model.eval()
        total_metrics = {}
        for eval_name, eval_data in eval_data_dict.items():
            logger.info(f"Running evaluation {eval_name}")
            with (
                torch.no_grad(),
                metrics_group(f"eval/{eval_name}", log_freq=1, force_recreate=True),
            ):
                log_event_start("perf/total_run")

                eval_iter = iter(eval_data)
                iterations = 0
                try:
                    # Note: We assume consistent number of batches across workers.
                    # If workers have different number of batches, they might hang waiting for each other
                    # during distributed metric aggregation if not handled correctly.
                    while max_iterations is None or iterations < max_iterations:
                        log_event_occurence("perf/full_iteration")

                        elapsed_batch_get, batch = measured_next(eval_iter)
                        criterion(model, batch)
                        log_summed("num_batches", lambda: 1)
                        log_averaged(
                            "perf/batch_get",
                            elapsed_batch_get,
                        )

                        iterations += 1

                        # Step metrics for each evaluation iteration
                        step_metrics(f"eval/{eval_name}")

                except StopIteration:
                    pass

                log_event_end("perf/total_run")

            eval_metrics = compute_metrics(
                f"eval/{eval_name}",
                aggregate=True,
                collective=collective,
            )
            logger.info(f"Finished eval: {eval_metrics}")
            total_metrics[f"eval/{eval_name}"] = eval_metrics
        return total_metrics


_, register_evaluator, build_evaluator = make_registry("evaluator", Evaluator)
register_evaluator("base", EvaluatorConfig)(Evaluator)
