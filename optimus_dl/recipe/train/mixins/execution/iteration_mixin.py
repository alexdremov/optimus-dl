"""Training iteration mixin for orchestrating complete training iterations."""

import logging
from collections.abc import Iterator
from contextlib import nullcontext
from typing import (
    Any,
    NamedTuple,
)

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer

from optimus_dl.core.log import warn_once
from optimus_dl.core.profile import (
    measured_lambda,
    measured_next,
)
from optimus_dl.modules.criterion import BaseCriterion
from optimus_dl.modules.metrics import (
    log_averaged,
    metrics_group,
)
from optimus_dl.modules.metrics.common import log_summed
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.recipe.train.config import OptimizationConfig

logger = logging.getLogger(__name__)


class ForwardPassResult(NamedTuple):
    loss: torch.Tensor
    exposed_protocols: dict[str, Any]
    elapsed_time: float


class OptimizerStepResult(NamedTuple):
    elapsed_time: float
    grad_norm: torch.Tensor | None


class TrainingIterationMixin:
    """Mixin for executing a complete training step with gradient accumulation.

    Encapsulates the core training logic:
    1.  **Forward Pass**: Runs the model and criterion, measuring time.
    2.  **Backward Pass**: Scales gradients and backpropagates, handling
        loss parallelism if applicable.
    3.  **Optimization**: Unscales gradients, clips norms, and steps the optimizer.
    4.  **Logging**: Records detailed performance metrics (forward/backward times,
        grad norms, etc.).

    Args:
        optimization_config: Configuration for optimization (accumulation steps, clipping).
        log_freq: Frequency of metric logging.
    """

    def __init__(self, optimization_config: OptimizationConfig, log_freq: int = 1):
        self.optimization_config = optimization_config
        self.log_freq = log_freq

    def log_memory_usage(self):
        """Log GPU memory usage statistics."""
        if torch.cuda.is_available():
            log_summed("gpu_gb_allocated", torch.cuda.memory_allocated() / (1024**3))
            log_summed("gpu_gb_used", torch.cuda.max_memory_allocated() / (1024**3))

    def execute_forward_pass(
        self,
        model: BaseModel,
        criterion: BaseCriterion,
        batch: Any,
        amp_ctx: Any,
        requested_protocols: set[str] | None = None,
    ) -> ForwardPassResult:
        """Run the forward pass inside an AMP context.

        Args:
            model: The model to run.
            criterion: The loss function.
            batch: The input data.
            amp_ctx: The autocast context manager.
            requested_protocols: Protocols requested by the metrics system.

        Returns:
            ForwardPassResult with the computed loss, exposed protocols, and execution time.
        """
        with amp_ctx:
            elapsed_forward, (loss, exposed) = measured_lambda(
                lambda: criterion(model, batch, requested_protocols=requested_protocols)
            )
        return ForwardPassResult(
            loss=loss, exposed_protocols=exposed, elapsed_time=elapsed_forward
        )

    def execute_backward_pass(self, loss: torch.Tensor, scaler: Any) -> float:
        """Run the backward pass with gradient scaling.

        Handles `loss_parallel` context if the loss is a DTensor.

        Args:
            loss: The computed loss tensor.
            scaler: The gradient scaler.

        Returns:
            Execution time in milliseconds.
        """

        def backward():
            with loss_parallel() if isinstance(loss, DTensor) else nullcontext():
                scaler.scale(loss).backward()

        elapsed_backward, _ = measured_lambda(backward)
        return elapsed_backward

    def execute_optimizer_step(
        self,
        optimizer: Optimizer,
        model: BaseModel,
        scaler: Any,
        clip_grad_norm: float | None = None,
    ) -> OptimizerStepResult:
        """Perform the optimization step.

        Includes gradient unscaling, optional gradient clipping, and the
        optimizer step itself. Updates the scaler state afterwards.

        Args:
            optimizer: The optimizer.
            model: The model (needed for clipping gradients).
            scaler: The gradient scaler.
            clip_grad_norm: Maximum norm for gradient clipping.

        Returns:
            OptimizerStepResult with execution time and the computed gradient norm.
        """
        scaler.unscale_(optimizer)

        grad_norm = None
        if clip_grad_norm is not None:
            from torch.distributed.tensor.experimental import implicit_replication

            with implicit_replication():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm
                )

        elapsed, _ = measured_lambda(lambda: scaler.step(optimizer))
        scaler.update()

        if scaler.is_enabled():
            log_averaged("grad_scale", scaler.get_scale())

        return OptimizerStepResult(elapsed_time=elapsed, grad_norm=grad_norm)

    def log_batch_metrics(
        self,
        elapsed_batch_get: float,
        elapsed_forward: float,
        elapsed_backward: float,
        acc_steps: int,
    ) -> None:
        """Log timing metrics for data loading and forward/backward passes."""
        weight = 1 / acc_steps

        log_averaged(
            "perf/batch_get",
            value=elapsed_batch_get,
            weight=weight,
            priority=999,
        )
        log_averaged(
            "perf/forward",
            value=elapsed_forward,
            weight=weight,
            priority=1000,
        )
        log_averaged(
            "perf/backward",
            value=elapsed_backward,
            weight=weight,
            priority=1001,
        )

    def log_optimizer_metrics(
        self,
        elapsed_optimizer: float,
        grad_norm: torch.Tensor | None,
        lr_scheduler: Any | None,
        optimizer: Optimizer,
    ) -> None:
        """Log optimizer performance, gradient norms, and learning rates."""
        log_averaged("perf/optimizer", value=elapsed_optimizer, priority=1002)

        # Log gradient norm if clipping was performed
        if grad_norm is not None:
            log_averaged(
                "grad_norm",
                lambda: (float(grad_norm) if grad_norm is not None else 0.0),
            )

        # Learning rate (cheap but we only need it periodically)
        if lr_scheduler is not None:
            log_averaged("learning_rate", lambda: lr_scheduler.get_last_lr()[0])
        else:
            log_averaged("learning_rate", lambda: optimizer.param_groups[0]["lr"])

    def run_training_iteration(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        criterion: BaseCriterion,
        train_data_iter: Iterator,
        training_context: dict[str, Any],
        lr_scheduler: Any | None = None,
        metric_engine: Any | None = None,
    ) -> None:
        """Execute one full training iteration, including gradient accumulation.

        This is the main driver for a training step. It loops `acc_steps` times
        to accumulate gradients before performing a single optimizer update.

        Args:
            model: The model to train.
            optimizer: The optimizer.
            criterion: The loss function.
            train_data_iter: Iterator yielding training batches.
            training_context: Dict with scaler, amp_ctx, etc.
            lr_scheduler: Optional learning rate scheduler.
            metric_engine: Optional MetricEngine for training metrics.
        """
        with metrics_group("train", log_freq=self.log_freq) as should_log:
            optimizer.zero_grad()
            model.train()

            requested_protocols = None
            if metric_engine and should_log:
                requested_protocols = metric_engine.required_external_protocols

            # Gradient accumulation loop
            for microbatch_idx in range(self.optimization_config.acc_steps):
                is_last_microbatch = (
                    microbatch_idx == self.optimization_config.acc_steps - 1
                )

                try:
                    elapsed_batch_get, batch = measured_next(train_data_iter)
                except StopIteration:
                    logger.error("Training data iterator exhausted unexpectedly")
                    break
                except Exception as e:
                    logger.error(f"Error getting batch: {e}")
                    continue

                with self.accumulation_context(model, is_last_microbatch):
                    forward_result = self.execute_forward_pass(
                        model,
                        criterion,
                        batch,
                        training_context["amp_ctx"],
                        requested_protocols=requested_protocols,
                    )
                    loss = forward_result.loss / self.optimization_config.acc_steps

                    if metric_engine and should_log:
                        # Pass computed data (loss, logits, etc.) to avoid redundant work in engine
                        computed_data = forward_result.exposed_protocols.copy()
                        computed_data["loss"] = forward_result.loss
                        metric_engine.update(
                            data=dict(model=model, batch=batch),
                            computed_data=computed_data,
                        )

                    elapsed_backward = self.execute_backward_pass(
                        loss, training_context["scaler"]
                    )

                # Log performance metrics using the training metrics mixin
                self.log_batch_metrics(
                    elapsed_batch_get,
                    forward_result.elapsed_time,
                    elapsed_backward,
                    self.optimization_config.acc_steps,
                )

            # Optimizer step
            optimizer_result = self.execute_optimizer_step(
                optimizer,
                model,
                training_context["scaler"],
                self.optimization_config.clip_grad_norm,
            )

            # Log optimizer metrics
            self.log_optimizer_metrics(
                optimizer_result.elapsed_time,
                optimizer_result.grad_norm,
                lr_scheduler,
                optimizer,
            )
            self.log_memory_usage()
            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

    def accumulation_context(self, model, is_last_microbatch):
        """Get the appropriate context manager for gradient accumulation.

        For FSDP/DDP models, this handles synchronization (e.g., disabling
        all-reduce during accumulation steps).
        """
        if hasattr(model, "accumulation_context"):
            ctx = model.accumulation_context(is_last_microbatch=is_last_microbatch)
            if not is_last_microbatch:
                warn_once(logger, "Using accumulation context")
            return ctx
        else:
            warn_once(logger, "Model does not support accumulation context, skipping")
            return nullcontext()
