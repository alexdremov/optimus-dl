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
            batch: Dictionary containing 'input_ids' and optional 'labels'.
            requested_protocols: Optional set of requested protocols.

        Returns:
            Tuple of (loss tensor, exposed_protocols dictionary).
        """
        requested_protocols = requested_protocols or set()
        batch = copy.copy(batch)
        input_ids = batch.pop("input_ids")
        labels = batch.pop("labels", None)

        B, T = input_ids.shape

        if labels is not None:
            # Batcher already performed causal shifting (input_ids and labels are aligned)
            targets = labels
            batch["input_ids"] = input_ids
        else:
            assert (
                "cu_seqlens" not in batch
            ), "If input is flat, we cannot generate labels and inputs efficiently"
            # Perform standard causal shifting: targets = inputs[1:], inputs = inputs[:-1]
            targets = input_ids[:, 1:]
            batch["input_ids"] = input_ids[:, :-1]

            # Metadata tensors that match the sequence length must also be sliced
            for k in list(batch.keys()):
                v = batch[k]
                if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == T:
                    batch[k] = v[:, :-1]

        # Log sequence statistics accurately for all schemes
        if "cu_seqlens" in batch:
            # Packed/Flat batch: metadata already adjusted for shifting
            cu = batch["cu_seqlens"]
            doc_lens = (cu[1:] - cu[:-1]).float()
            log_averaged("input_max_seq_len", doc_lens.max().item(), round=2)
            log_averaged(
                "input_mean_seq_len",
                lambda: doc_lens.mean().item(),
                weight=len(doc_lens),
                round=2,
            )
        elif "seq_lens" in batch:
            # Padded batch: lengths represent un-shifted sequences, we adjust here
            sl = (batch["seq_lens"] - 1).float()
            log_averaged("input_max_seq_len", sl.max().item(), round=2)
            log_averaged(
                "input_mean_seq_len",
                lambda: sl.mean().item(),
                weight=sl.shape[0],
                round=2,
            )
        else:
            # Fixed-size batch
            current_T = batch["input_ids"].shape[1]
            log_averaged("input_max_seq_len", current_T, round=2)
            log_averaged("input_mean_seq_len", current_T, weight=B, round=2)

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
        if (
            StandardProtocols.LOGITS in requested_protocols
            or StandardProtocols.CLASSIFICATION in requested_protocols
        ):
            with torch.no_grad():
                is_flat = B == 1 and "cu_seqlens" in batch
                current_seq_lens = batch.get("seq_lens")

                if StandardProtocols.LOGITS in requested_protocols:
                    res_logits = logits
                    if isinstance(res_logits, DTensor):
                        res_logits = res_logits.full_tensor()

                    if is_flat:
                        res_logits = self._unflatten_flat(
                            res_logits, batch["cu_seqlens"], batch["max_seqlen"]
                        )
                    exposed[StandardProtocols.LOGITS] = res_logits

                if StandardProtocols.CLASSIFICATION in requested_protocols:
                    res_preds = predictions()  # Already gathered by cached_lambda
                    res_targets = targets

                    # Base mask for valid tokens
                    res_mask = res_targets != self.padding_token_id
                    # Refine mask for padded batches if current_seq_lens is available
                    if not is_flat and current_seq_lens is not None:
                        res_mask = res_mask & (
                            torch.arange(res_mask.shape[1], device=res_mask.device)
                            < current_seq_lens[:, None]
                        )

                    if is_flat:
                        cu = batch["cu_seqlens"]
                        ms = batch["max_seqlen"]
                        res_preds = self._unflatten_flat(res_preds, cu, ms)
                        res_targets = self._unflatten_flat(
                            res_targets, cu, ms, pad_val=self.padding_token_id
                        )
                        res_mask = self._unflatten_flat(res_mask, cu, ms, pad_val=False)

                    exposed[StandardProtocols.CLASSIFICATION] = dict(
                        predictions=res_preds,
                        targets=res_targets,
                        mask=res_mask,
                    )

        return loss, exposed

    @staticmethod
    def _unflatten_flat(t, cu_seqlens, max_seqlen, pad_val=0):
        """Helper to reconstruct (batch, time) layout from a flat (1, sum_T) batch."""
        # t is (1, T_sum, ...)
        device = t.device
        dtype = t.dtype
        num_docs = len(cu_seqlens) - 1
        total_tokens = int(cu_seqlens[-1].item())

        # seqlens of each document
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        # Batch index for each token: [0,0,0, 1,1, 2,2,2,2, ...]
        batch_idx = torch.repeat_interleave(
            torch.arange(num_docs, device=device), seqlens.to(torch.long)
        )

        # Local index for each token: [0,1,2, 0,1, 0,1,2,3, ...]
        # Global index minus sequence start index
        local_idx = torch.arange(total_tokens, device=device) - torch.repeat_interleave(
            cu_seqlens[:-1].to(torch.long), seqlens.to(torch.long)
        )

        # Prepare output buffer (batch, max_time, ...)
        out_shape = (num_docs, max_seqlen, *t.shape[2:])
        out = torch.full(out_shape, pad_val, device=device, dtype=dtype)

        # Vectorized assignment: out[batch_idx, local_idx] = t[0, :total_tokens]
        out[batch_idx, local_idx] = t[0]

        return out

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
