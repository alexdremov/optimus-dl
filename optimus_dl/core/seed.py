"""Utilities for setting random seeds for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int, cuda_deterministic: bool = False) -> None:
    """Set random seeds for reproducibility across different libraries.

    Args:
        seed: The integer seed to set.
        cuda_deterministic: If True, makes CUDA operations deterministic.
            Note: This can sometimes come with a performance penalty.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Force determinism in PyTorch operations
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Default behavior for performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
