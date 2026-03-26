"""Evaluation mixin for evaluation functionality."""

import logging
import time
from dataclasses import (
    dataclass,
)
from typing import Any

import torch

from optimus_dl.core.log import tqdm
from optimus_dl.core.profile import measured_next
from optimus_dl.core.registry import (
    RegistryConfig,
    make_registry,
)
from optimus_dl.modules.criterion import BaseCriterion
from optimus_dl.modules.data import EvalDataPipeline
from optimus_dl.modules.distributed.base import Collective
from optimus_dl.modules.metrics import (
    compute_meters,
    log_averaged,
    log_event_end,
    log_event_occurence,
    log_event_start,
    log_summed,
    meters_group,
    step_meters,
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
            If None or negative, processes the entire dataset (negative values are
            treated as unlimited).
        eval_guaranteed_same_batches:  If True, assumes all ranks will see the same number of batches, allowing for simpler stopping logic.
            If False, uses collective communication to determine when to stop if any rank exhausts its dataloader.
    """

    def __init__(
        self,
        cfg: EvaluatorConfig,
        eval_freq: int = 0,
        eval_iterations: int | None = None,
        eval_guaranteed_same_batches: bool = False,
        **kwargs: Any,
    ):
        self.cfg = cfg
        self.eval_freq = eval_freq
        self.eval_iterations = eval_iterations
        self.eval_guaranteed_same_batches = eval_guaranteed_same_batches

    def run_evaluation_if_needed(
        self,
        iteration: int,
        model: BaseModel,
        criterion: BaseCriterion,
        eval_data_dict: dict[str, EvalDataPipeline],
        collective: Collective | None = None,
        all_metrics_configs: dict[str, list[dict]] | None = None,
    ) -> None | dict:
        """Run evaluation if the current iteration matches the frequency.

        Args:
            iteration: Current training step.
            model: The model to evaluate.
            criterion: The loss function.
            eval_data_dict: Dictionary mapping dataset names to dataloaders.
            collective: Distributed collective for metric aggregation.
            all_metrics_configs: Root metrics configuration from TrainConfig.

        Returns:
            Dictionary of computed metrics if evaluation ran, else None.
        """
        result = {}

        # deterministic order
        eval_data_dict_keys = sorted(eval_data_dict.keys())
        for eval_name in eval_data_dict_keys:
            eval_data = eval_data_dict[eval_name]

            max_iterations = (
                eval_data.eval_iterations
                if eval_data.eval_iterations is not None
                else self.eval_iterations
            )
            if max_iterations is not None and max_iterations < 0:
                max_iterations = None

            eval_freq = (
                eval_data.eval_freq
                if eval_data.eval_freq is not None
                else self.eval_freq
            )
            if eval_freq <= 0 or iteration % eval_freq != 0:
                continue

            try:
                result |= self.run_evaluation(
                    model=model,
                    criterion=criterion,
                    eval_data_dict={eval_name: eval_data},
                    max_iterations=max_iterations,
                    collective=collective,
                    all_metrics_configs=all_metrics_configs,
                    show_progress=True,
                )
            except Exception:
                logger.exception(f"Evaluation for {eval_name} failed.")

        if len(result) == 0:
            return None
        return result

    def run_evaluation(
        self,
        model: BaseModel,
        criterion: BaseCriterion,
        eval_data_dict: dict,
        max_iterations: int | None = None,
        collective: Collective | None = None,
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

        eval_data_dict_keys = sorted(eval_data_dict.keys())
        for eval_name in eval_data_dict_keys:
            eval_data = eval_data_dict[eval_name]
            max_iterations_local = (
                eval_data.eval_iterations
                if eval_data.eval_iterations is not None
                else max_iterations
            )
            guaranteed_same_batches_local = (
                eval_data.eval_guaranteed_same_batches
                if eval_data.eval_guaranteed_same_batches is not None
                else self.eval_guaranteed_same_batches
            )
            if max_iterations_local is not None and max_iterations_local < 0:
                max_iterations_local = None

            logger.info(
                f"Running evaluation {eval_name} for {max_iterations_local if max_iterations_local is not None else 'unlimited'} iterations (on each rank)"
            )

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
                meters_group(
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
                        disable=(
                            collective is not None and not collective.is_local_master
                        ),
                        unit="batch",
                        total=(
                            max_iterations_local
                            if max_iterations_local is not None
                            else None
                        ),
                    )

                try:
                    while (
                        max_iterations_local is None
                        or max_iterations_local < 0
                        or iterations < max_iterations_local
                    ):
                        log_event_occurence("perf/full_iteration")
                        exhausted = False
                        try:
                            elapsed_batch_get, batch = measured_next(eval_iter)
                        except StopIteration:
                            exhausted = True

                        if collective is not None and not guaranteed_same_batches_local:
                            flag = torch.tensor(
                                [exhausted],
                                device=collective.default_device,
                                dtype=torch.int32,
                            )
                            collective.all_reduce(
                                flag,
                                op=Collective.ReduceOp.MAX,
                            )
                            if flag.item() == 1:
                                # at least one rank is exhausted, stop evaluation
                                raise StopIteration
                        else:
                            # If we are guaranteed that all ranks see the same number of batches,
                            # we can just stop when this rank is exhausted
                            if exhausted:
                                raise StopIteration

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
                        if pbar is not None:
                            pbar.update(1)

                        # Step metrics for each evaluation iteration
                        step_meters(f"{metrics_prefix}/{eval_name}")

                except StopIteration:
                    pass
                finally:
                    if pbar is not None:
                        pbar.refresh()
                        pbar.close()

                total_time = time.perf_counter() - start_time
                log_event_end("perf/total_run")

            eval_metrics = compute_meters(
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

            logger.info(
                f"Finished eval {eval_name}: {eval_metrics} in {total_time:.1f}s"
            )
            total_metrics[f"{metrics_prefix}/{eval_name}"] = eval_metrics
        return total_metrics


_, register_evaluator, build_evaluator = make_registry("evaluator", Evaluator)
register_evaluator("base", EvaluatorConfig)(Evaluator)
