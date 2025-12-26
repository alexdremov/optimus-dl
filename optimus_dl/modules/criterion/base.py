"""Base criterion (loss function) class.

This module defines the BaseCriterion class that all loss functions must inherit from.
Criteria compute the loss given a model and a batch of data.
"""

import torch

from optimus_dl.modules.model.base import BaseModel


class BaseCriterion:
    """Base class for all loss criteria (loss functions).

    All loss functions in Optimus-DL should inherit from this class. The criterion
    is responsible for computing the loss given a model's output and the target data.

    Subclasses should implement:

    - `__call__()`: Compute the loss given model and batch

    Example:
        ```python
        @register_criterion("cross_entropy", CrossEntropyConfig)
        class CrossEntropyCriterion(BaseCriterion):
            def __init__(self, cfg: CrossEntropyConfig):
                self.cfg = cfg

            def __call__(self, model: BaseModel, batch: dict) -> torch.Tensor:
                logits = model(batch["input_ids"])
                return F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      batch["labels"].view(-1))

        ```"""

    def __call__(self, model: BaseModel, batch: dict) -> torch.Tensor:
        """Compute the loss for a given model and batch.

        Args:
            model: The model to compute loss for. Should be called with the batch
                to get model outputs.
            batch: Dictionary containing input data and targets. Typically includes:
                - "input_ids": Token IDs for the input sequence
                - "labels": Target token IDs for computing loss
                - Other model-specific fields

        Returns:
            Scalar tensor containing the loss value.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        Example:
            ```python
            criterion = CrossEntropyCriterion(cfg)
            loss = criterion(model, batch)
            loss.backward()

            ```"""
        raise NotImplementedError
