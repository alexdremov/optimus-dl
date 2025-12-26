"""Optimus-DL: A modular, high-performance framework for training Large Language Models.

Optimus-DL is a research framework built on PyTorch that provides:
- Modular "Recipe" architecture for clean separation of concerns
- Hydra-based configuration management
- Universal metrics system with distributed aggregation
- Modern PyTorch features (AMP, FSDP2, Tensor Parallelism, torch.compile)
- Efficient kernels via Liger-Kernel integration
- Registry system for easy component swapping

Example:
    Basic training::

        >>> from optimus_dl.core.registry import build
        >>> from optimus_dl.recipe.train.config import TrainConfig
        >>>
        >>> config = TrainConfig(...)
        >>> recipe = build("train_recipe", config)
        >>> recipe.run()
"""

import os

import torch

from optimus_dl.core.bootstrap import bootstrap_module

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for when the package is not installed or setuptools_scm hasn't run yet
    __version__ = "0.0.0+unknown"

# Set PyTorch to use single thread by default to avoid thread contention
# Users can override this if needed
torch.set_num_threads(1)

# Automatically import all submodules to register components
bootstrap_module(__name__)
