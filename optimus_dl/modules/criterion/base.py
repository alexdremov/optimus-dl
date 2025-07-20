import torch


class BaseCriterion:
    def __call__(self, model, batch) -> torch.Tensor:
        raise NotImplementedError
