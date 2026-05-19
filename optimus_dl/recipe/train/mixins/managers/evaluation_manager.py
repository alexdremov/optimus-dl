"""Evaluation mixin for evaluation functionality."""

import contextlib
import logging
import pathlib
import time
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

import torch

from optimus_dl.core.dtype import str_to_dtype
from optimus_dl.core.log import (
    info_once,
    tqdm,
)
from optimus_dl.core.profile import measured_next
from optimus_dl.core.registry import (
    RegistryConfig,
    make_registry,
)
from optimus_dl.modules.checkpoint.eval_checkpoint_manager import (
    EvaluationCheckpointManager,
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
from optimus_dl.modules.optim import AmpConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig(RegistryConfig):
    """Configuration for the Evaluator."""

    amp: AmpConfig = field(default_factory=AmpConfig)


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
        eval_guaranteed_same_batches: If True, assumes all ranks will see the same
            number of batches, allowing for simpler stopping logic. If False, uses
            collective communication to determine when to stop if any rank exhausts
            its dataloader.
    """

    def __init__(
        self,
        cfg: EvaluatorConfig,
        eval_freq: int = 0,
        eval_iterations: int | None = None,
        eval_guaranteed_same_batches: bool = False,
        eval_checkpointing: int | None = None,
        output_path: str | pathlib.Path | None = None,
        **kwargs: Any,
    ):
        self.cfg = cfg
        self.eval_freq = eval_freq
        self.eval_iterations = eval_iterations
        self.eval_guaranteed_same_batches = eval_guaranteed_same_batches
        self.eval_checkpointing = eval_checkpointing
        self.output_path = output_path
        self.eval_checkpoint_manager = None
        if output_path:
            self.eval_checkpoint_manager = EvaluationCheckpointManager(output_path)

    @contextlib.contextmanager
    def forward_context(self, device: torch.device):
        """Context manager for evaluation forward pass.

        Can be used to set up any necessary context (e.g., mixed precision) for the
        forward pass during evaluation.

        Returns:
            A context manager (e.g., from contextlib) that sets up the desired context.
        """
        if self.cfg.amp.enabled:
            amp_cfg = self.cfg.amp
            dtype = str_to_dtype(amp_cfg.dtype)
            amp_ctx = torch.autocast(device.type, dtype=dtype, enabled=amp_cfg.enabled)
            with amp_ctx:
                yield
        else:
            yield

    def should_run_evaluation(
        self,
        iteration: int,
        eval_data_dict: dict[str, EvalDataPipeline],
    ) -> bool:
        """Check if any of the evaluation datasets match the current iteration frequency.

        Args:
            iteration: Current training step.
            eval_data_dict: Dictionary mapping dataset names to eval data pipelines.

        Returns:
            True if at least one evaluation should run, False otherwise.
        """
        for eval_data in eval_data_dict.values():
            eval_freq = (
                eval_data.eval_freq
                if eval_data.eval_freq is not None
                else self.eval_freq
            )
            if eval_freq > 0 and iteration % eval_freq == 0:
                return True
        return False

    def run_evaluation_if_needed(
        self,
        iteration: int,
        model: BaseModel,
        criterion: BaseCriterion,
        eval_data_dict: dict[str, EvalDataPipeline],
        device: torch.device,
        collective: Collective | None = None,
        all_metrics_configs: dict[str, list[dict]] | None = None,
    ) -> None | dict:
        """Run evaluation if the current iteration matches the frequency for any dataset.

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
                    iteration=iteration,
                    model=model,
                    criterion=criterion,
                    eval_data_dict={eval_name: eval_data},
                    max_iterations=max_iterations,
                    collective=collective,
                    all_metrics_configs=all_metrics_configs,
                    show_progress=True,
                    device=device,
                )
            except Exception as e:
                logger.error(
                    f"Error during evaluation of {eval_name} at iteration {iteration}: {e}"
                )
                raise

        if len(result) == 0:
            return None
        return result

    def run_evaluation(
        self,
        model: BaseModel,
        criterion: BaseCriterion,
        eval_data_dict: dict[str, EvalDataPipeline],
        device: torch.device,
        max_iterations: int | None = None,
        collective: Collective | None = None,
        all_metrics_configs: dict[str, list[dict]] | None = None,
        metrics_prefix: str = "eval",
        show_progress: bool = False,
        iteration: int | None = None,
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
            iteration: Current training iteration, used for naming checkpoints.

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

            eval_checkpointing = (
                eval_data.eval_checkpointing
                if eval_data.eval_checkpointing is not None
                else self.eval_checkpointing
            )

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

            group_name = f"{metrics_prefix}/{eval_name}"
            with (
                torch.no_grad(),
                meters_group(group_name, log_freq=1, force_recreate=True),
            ):
                iterations = 0
                if self.eval_checkpoint_manager is not None and iteration is not None:
                    iterations = self.eval_checkpoint_manager.load_iteration_state(
                        iteration=iteration,
                        eval_name=eval_name,
                        group_name=group_name,
                        dataloader=dataloader,
                        collective=collective,
                    )

                log_event_start("perf/total_run")
                start_time = time.perf_counter()

                eval_iter = iter(dataloader)

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
                        initial=iterations,
                    )

                check_exhaustion = (
                    collective is not None and not guaranteed_same_batches_local
                )

                def should_stop(_iterations, max_iterations_local=max_iterations_local):
                    return (
                        max_iterations_local is not None
                        and max_iterations_local > 0
                        and _iterations >= max_iterations_local
                    )

                try:
                    # Consider we loaded state where one rank is already exhausted.
                    # We should check it before starting the loop to have the same collectives set.

                    stop_flag = None
                    if check_exhaustion:
                        assert collective is not None
                        stop_flag = torch.tensor(
                            [int(should_stop(iterations))],
                            device=collective.default_device,
                            dtype=torch.int32,
                        )
                        collective.all_reduce(
                            stop_flag,
                            op=Collective.ReduceOp.MAX,
                        )
                        if stop_flag.item() == 1:
                            logger.info(
                                "Some ranks have already finished evaluation before starting the loop, stopping immediately."
                            )
                            raise StopIteration

                    while not should_stop(iterations):
                        logger.debug(
                            f"Eval {eval_name}: Starting iteration {iterations}"
                        )
                        log_event_occurence("perf/full_iteration")
                        exhausted = False
                        try:
                            logger.debug(
                                f"Eval {eval_name}: Fetching batch for iteration {iterations}"
                            )
                            elapsed_batch_get, batch = measured_next(eval_iter)
                            info_once(logger, f"Batch has keys {batch.keys()}")
                        except StopIteration:
                            logger.debug(
                                f"Eval {eval_name}: Dataloader exhausted on this rank"
                            )
                            exhausted = True

                        if check_exhaustion:
                            assert collective is not None
                            assert stop_flag is not None
                            logger.debug(
                                f"Eval {eval_name}: Synchronizing exhaustion state (all_reduce MAX)"
                            )
                            stop_flag[0] = int(exhausted)
                            collective.all_reduce(
                                stop_flag,
                                op=Collective.ReduceOp.MAX,
                            )
                            if stop_flag.item() == 1:
                                # at least one rank is exhausted, stop evaluation
                                logger.info(
                                    f"Eval {eval_name}: At least one rank exhausted its dataloader, stopping evaluation."
                                )
                                raise StopIteration
                        else:
                            # If we are guaranteed that all ranks see the same number of batches,
                            # we can just stop when this rank is exhausted
                            if exhausted:
                                logger.info(
                                    f"Eval {eval_name}: Dataloader exhausted, stopping evaluation."
                                )
                                raise StopIteration

                        logger.debug(
                            f"Eval {eval_name}: Running forward pass for iteration {iterations}"
                        )
                        with self.forward_context(device=device):
                            loss, exposed = criterion(
                                model, batch, requested_protocols=requested_protocols
                            )

                        if engine:
                            logger.debug(f"Eval {eval_name}: Updating metric engine")
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
                        logger.debug(
                            f"Eval {eval_name}: Finished iteration {iterations-1}"
                        )

                        if (
                            self.eval_checkpoint_manager is not None
                            and eval_checkpointing is not None
                            and eval_checkpointing > 0
                            and iterations % eval_checkpointing == 0
                        ):
                            assert (
                                iteration is not None
                            ), "Iteration must be provided for checkpointing"
                            self.eval_checkpoint_manager.save_iteration_state(
                                iteration=iteration,
                                eval_name=eval_name,
                                dataloader_state=dataloader.state_dict(),
                                group_name=group_name,
                                collective=collective,
                                eval_iterations_processed=iterations,
                            )
                            logger.info(
                                f"Saved evaluation metrics checkpoint at iteration {iterations}"
                            )

                except StopIteration:
                    pass
                finally:
                    if pbar is not None:
                        pbar.refresh()
                        pbar.close()

                total_time = time.perf_counter() - start_time
                log_event_end("perf/total_run")

            logger.debug(f"Eval {eval_name}: Computing aggregated meters")
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

    def cleanup_all_eval_checkpoints(
        self, iteration: int | None = None, exclude_iteration: int | None = None
    ) -> None:
        """Cleanup evaluation checkpoints.

        If iteration is provided, cleans up checkpoints for that iteration only.
        If iteration is None, cleans up ALL evaluation checkpoints in the output path,
        optionally excluding one specific iteration.
        """
        if self.eval_checkpoint_manager is not None:
            self.eval_checkpoint_manager.cleanup(
                iteration=iteration, exclude_iteration=exclude_iteration
            )
        else:
            logger.debug(
                "No evaluation checkpoint manager initialized, skipping cleanup."
            )


_, register_evaluator, build_evaluator = make_registry("evaluator", Evaluator)
register_evaluator("base", EvaluatorConfig)(Evaluator)
