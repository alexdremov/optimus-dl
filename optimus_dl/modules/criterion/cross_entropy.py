from dataclasses import dataclass

import torch

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.criterion import BaseCriterion, register_criterion
from optimus_dl.modules.metrics import (
    cached_lambda,
    log_averaged,
    log_averaged_exponent,
    log_summed,
)


@dataclass
class CrossEntropyCriterionConfig(RegistryConfig):
    label_smoothing: float = 0.0


@register_criterion("cross_entropy", CrossEntropyCriterionConfig)
class CrossEntropyCriterion(BaseCriterion):
    def __init__(self, cfg: CrossEntropyCriterionConfig, **kwargs):
        self.cfg = cfg

    def __call__(self, model, batch):
        input_ids = batch.pop("input_ids")

        batch["input_ids"] = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        model_out = model(**batch)
        logits = model_out["logits"]

        valid_tokens = cached_lambda(lambda: (targets >= 0).sum().item())

        log_averaged(
            "accuracy",
            lambda: self.accuracy_metric(logits, targets),
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

        targets = targets.reshape(-1)
        with torch.autocast(targets.device.type, enabled=False):
            loss = torch.nn.functional.cross_entropy(
                input=logits.view(-1, logits.size(-1)).float(),
                target=targets,
                label_smoothing=self.cfg.label_smoothing,
                ignore_index=-100,
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

        return loss

    @torch.no_grad()
    def accuracy_metric(self, logits, targets):
        """Compute accuracy for the given logits and targets."""
        predictions = torch.argmax(logits, dim=-1)
        correct = predictions == targets
        valid = targets >= 0
        correct = (correct & valid).float()
        return (correct.sum() / valid.sum()).item()
