import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.distributed.tensor import (
    DTensor,
    Shard,
)
from torch.distributed.tensor.parallel import loss_parallel

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.criterion import (
    BaseCriterion,
    register_criterion,
)
from optimus_dl.modules.distributed.base import Collective
from optimus_dl.modules.distributed.mesh import MeshCollective
from optimus_dl.modules.metrics import (
    cached_lambda,
    log_averaged,
    log_averaged_exponent,
    log_summed,
)
from optimus_dl.modules.metrics.source import StandardProtocols

logger = logging.getLogger(__name__)


@dataclass
class CrossEntropyCriterionConfig(RegistryConfigStrict):
    label_smoothing: float = 0.0
    use_liger_kernel: bool | None = None
    padding_token_id: int = -100


@register_criterion("cross_entropy", CrossEntropyCriterionConfig)
class CrossEntropyCriterion(BaseCriterion):
    """Standard Cross Entropy loss with distributed and kernel optimizations.

    This criterion implements standard Cross Entropy but adds support for:

    - **Loss Parallelism**: Computes loss directly on sharded logits (DTensors) to
      save memory and communication.
    - **Liger Kernel**: Optional high-performance kernel for faster computation
      and lower memory usage on GPUs.
    - **Metrics**: Automatically logs accuracy, perplexity, and token counts.

    Args:
        cfg: Configuration for cross entropy.
        collective: Collective object for distributed operations.
    """

    def __init__(
        self, cfg: CrossEntropyCriterionConfig, collective: Collective, **kwargs
    ):
        self.cfg = cfg
        self.collective = collective
        self.padding_token_id = cfg.padding_token_id
        self._liger_available = False
        if self.cfg.use_liger_kernel or self.cfg.use_liger_kernel is None:
            try:
                from liger_kernel.transformers.functional import liger_cross_entropy

                self._liger_cross_entropy = liger_cross_entropy
                self._liger_available = True
                if self.cfg.use_liger_kernel is None:
                    logger.info("Using liger-kernel for cross-entropy.")
            except ImportError:
                if self.cfg.use_liger_kernel is not None:
                    logger.warning(
                        "use_liger_kernel=True but liger-kernel is not installed. Falling back to PyTorch."
                    )

    def __call__(self, model, batch, requested_protocols: set[str] | None = None):
        """Compute the cross entropy loss.

        Automatically handles target shifting (labels = inputs[1:]) and manages
        distributed loss computation if the model output is a DTensor.

        Args:
            model: The language model.
            batch: Dictionary containing 'input_ids'.
            requested_protocols: Optional set of requested protocols.

        Returns:
            Tuple of (loss tensor, exposed_protocols dictionary).
        """
        requested_protocols = requested_protocols or set()
        batch = copy.copy(batch)
        input_ids = batch.pop("input_ids")

        batch["input_ids"] = input_ids[:, :-1]

        log_averaged(
            "input_max_seq_len",
            input_ids.shape[1],
            round=2,
        )
        seq_lens = batch.get("seq_lens")
        if seq_lens is not None:
            log_averaged(
                "input_mean_seq_len",
                lambda: seq_lens.float().mean().item(),
                weight=seq_lens.shape[0],
                round=2,
            )

        targets = input_ids[:, 1:]
        model_out = model(**batch)
        logits = model_out["logits"]
        is_dtensor = isinstance(logits, DTensor)

        valid_tokens = cached_lambda(
            lambda: ((targets >= 0) & (targets != self.padding_token_id)).sum().item()
            / self.collective.tp_world_size
        )
        predictions = cached_lambda(lambda: self.gather_predictions(logits))

        log_averaged(
            "accuracy",
            lambda: self.accuracy_metric(predictions(), targets),
            weight=valid_tokens,
            round=2,
        )
        log_summed(
            "batch_tokens",
            valid_tokens,
        )
        log_summed(
            "total_tokens",
            valid_tokens,
            reset=False,
        )

        targets_flat = targets.reshape(-1)
        enable_loss_parallel = False
        if is_dtensor:
            from torch.distributed.tensor.placement_types import Replicate

            if not isinstance(targets_flat, DTensor):
                targets_parallel = DTensor.from_local(
                    targets_flat, logits.device_mesh, (Replicate(),)
                )
            else:
                targets_parallel = targets_flat

            # Only enable loss_parallel if logits are actually sharded
            for placement in logits.placements:
                if isinstance(placement, Shard):
                    enable_loss_parallel = True
                    break
        else:
            targets_parallel = targets_flat

        if (
            self._liger_available
            and targets_parallel.device.type != "cpu"
            and not is_dtensor
        ):
            # Liger kernel handles mixed precision internally, no need to cast to float
            loss = self._liger_cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=targets_parallel,
                label_smoothing=self.cfg.label_smoothing,
                ignore_index=self.padding_token_id,
            )
        else:
            with (
                torch.autocast(targets_parallel.device.type, enabled=False),
                loss_parallel() if enable_loss_parallel else nullcontext(),
            ):
                loss = torch.nn.functional.cross_entropy(
                    input=logits.view(-1, logits.size(-1)).float(),
                    target=targets_parallel,
                    label_smoothing=self.cfg.label_smoothing,
                    ignore_index=self.padding_token_id,
                )

        log_averaged(
            "loss",
            value=lambda: loss.item(),
            weight=valid_tokens,
        )
        log_averaged_exponent(
            "perplexity",
            value=lambda: loss.item(),
            weight=valid_tokens,
        )

        exposed = {}
        if StandardProtocols.LOGITS in requested_protocols:
            exposed[StandardProtocols.LOGITS] = logits

        if StandardProtocols.CLASSIFICATION in requested_protocols:
            # Build classification protocol data for reuse
            mask = targets != self.padding_token_id
            if seq_lens is not None:
                mask = mask & (
                    torch.arange(mask.shape[1], device=mask.device) < seq_lens[:, None]
                )

            classification = dict(
                predictions=predictions(),
                targets=targets,
                mask=mask,
            )
            exposed[StandardProtocols.CLASSIFICATION] = classification

        return loss, exposed

    @torch.no_grad()
    def gather_predictions(self, logits):
        """
        Get predictions from logits.
        """
        is_dtensor = isinstance(logits, DTensor)
        if is_dtensor:
            assert isinstance(self.collective, MeshCollective)
            local_logits = logits.to_local()
            maxes = torch.max(local_logits, -1)

            maxes_values_distr = DTensor.from_local(
                maxes.values,
                device_mesh=self.collective.tp_mesh,
                placements=(Shard(1),),
            ).full_tensor()
            tok_shift = self.collective.tp_rank * local_logits.size(-1)
            maxes_index_distr = DTensor.from_local(
                maxes.indices + tok_shift,
                device_mesh=self.collective.tp_mesh,
                placements=(Shard(1),),
            ).full_tensor()

            max_total = torch.max(maxes_values_distr, -1, keepdim=True)
            predictions = torch.gather(
                maxes_index_distr,
                dim=1,
                index=max_total.indices,
            )
        else:
            predictions = torch.argmax(logits, dim=-1)
        return predictions

    @torch.no_grad()
    def accuracy_metric(self, predictions, targets):
        """Compute top-1 accuracy.

        Handles both standard Tensors and distributed DTensors. For DTensors, it
        performs a distributed max across tensor-parallel ranks.
        """

        correct = predictions == targets
        valid = (targets >= 0) & (targets != self.padding_token_id)
        correct = (correct & valid).float()
        return (correct.sum() / valid.sum()).item()
