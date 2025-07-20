"""Training iteration mixin for orchestrating complete training iterations."""

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, NamedTuple

import torch
from torch.optim import Optimizer

from optimus_dl.core.log import warn_once
from optimus_dl.core.profile import measured_lambda, measured_next
from optimus_dl.modules.criterion import BaseCriterion
from optimus_dl.modules.metrics import log_averaged, metrics_group
from optimus_dl.modules.metrics.common import log_summed
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.recipe.train.config import OptimizationConfig

logger = logging.getLogger(__name__)


class ForwardPassResult(NamedTuple):
    loss: torch.Tensor
    elapsed_time: float


class OptimizerStepResult(NamedTuple):
    elapsed_time: float
    grad_norm: torch.Tensor | None


class TrainingIterationMixin:
    """Mixin for orchestrating complete training iterations with gradient accumulation."""

    def __init__(self, optimization_config: OptimizationConfig, log_freq: int = 1):
        self.optimization_config = optimization_config
        self.log_freq = log_freq

    def log_memory_usage(self):
        if torch.cuda.is_available():
            log_summed("gpu_gb_allocated", torch.cuda.memory_allocated() / (1024**3))
            log_summed("gpu_gb_used", torch.cuda.max_memory_allocated() / (1024**3))

    def execute_forward_pass(
        self, model: BaseModel, criterion: BaseCriterion, batch: Any, amp_ctx: Any
    ) -> ForwardPassResult:
        """Execute forward pass and return loss + timing.

        Args:
            model: Model to run forward pass on
            criterion: Loss criterion
            batch: Input batch
            amp_ctx: Automatic mixed precision context

        Returns:
            ForwardPassResult namedtuple with loss and elapsed_time
        """
        with amp_ctx:
            elapsed_forward, loss = measured_lambda(lambda: criterion(model, batch))
        return ForwardPassResult(loss=loss, elapsed_time=elapsed_forward)

    def execute_backward_pass(self, loss: torch.Tensor, scaler: Any) -> float:
        """Execute backward pass and return timing.

        Args:
            loss: Loss tensor to backpropagate
            scaler: Gradient scaler for mixed precision

        Returns:
            Elapsed time for backward pass
        """
        elapsed_backward, _ = measured_lambda(lambda: scaler.scale(loss).backward())
        return elapsed_backward

    def execute_optimizer_step(
        self,
        optimizer: Optimizer,
        model: BaseModel,
        scaler: Any,
        clip_grad_norm: float | None = None,
    ) -> OptimizerStepResult:
        """Execute optimizer step with optional gradient clipping.

        Args:
            optimizer: Optimizer to step
            model: Model whose parameters to update
            scaler: Gradient scaler for mixed precision
            clip_grad_norm: Maximum gradient norm (optional)

        Returns:
            OptimizerStepResult namedtuple with elapsed_time and grad_norm
        """
        scaler.unscale_(optimizer)

        grad_norm = None
        if clip_grad_norm is not None:
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
        """Log performance metrics for batch processing.

        Args:
            elapsed_batch_get: Time spent getting the batch
            elapsed_forward: Time spent in forward pass
            elapsed_backward: Time spent in backward pass
            acc_steps: Number of accumulation steps for weight calculation
        """
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
        """Log optimizer-related metrics.

        Args:
            elapsed_optimizer: Time spent in optimizer step
            grad_norm: Gradient norm if clipping was performed
            lr_scheduler: Learning rate scheduler (optional)
            optimizer: Optimizer for learning rate extraction
        """
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
    ) -> None:
        """Execute one complete training iteration with gradient accumulation.

        Uses gradient accumulation with proper context management for DDP/FSDP/FSDP2.
        Logs metrics directly using the training metrics mixin methods.

        Args:
            model: Model to train
            optimizer: Optimizer for parameter updates
            criterion: Loss criterion
            train_data_iter: Iterator over training data
            training_context: Training context with scaler, amp_ctx, etc.
            lr_scheduler: Learning rate scheduler (optional)
        """
        with metrics_group("train", log_freq=self.log_freq):
            optimizer.zero_grad()
            model.train()

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
                        model, criterion, batch, training_context["amp_ctx"]
                    )
                    loss = forward_result.loss / self.optimization_config.acc_steps

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
        if hasattr(model, "accumulation_context"):
            ctx = model.accumulation_context(is_last_microbatch=is_last_microbatch)
            if not is_last_microbatch:
                warn_once(logger, f"Using accumulation context: {ctx}")
            return ctx
        else:
            warn_once(logger, "Model does not support accumulation context, skipping")
            return nullcontext()
