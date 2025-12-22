import os

import torch

from optimus_dl.core.bootstrap import bootstrap_module

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for when the package is not installed or setuptools_scm hasn't run yet
    __version__ = "0.0.0+unknown"

torch.set_num_threads(1)
bootstrap_module(__name__)
