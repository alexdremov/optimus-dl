import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

try:
    from liger_kernel.transformers.functional import liger_rms_norm

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    liger_rms_norm = None


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_liger: bool | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        if use_liger is None:
            self.use_liger = LIGER_AVAILABLE
        else:
            self.use_liger = use_liger

        if self.use_liger and not LIGER_AVAILABLE:
            logger.warning(
                "Liger Kernel requested for RMSNorm but not installed. Fallback to PyTorch."
            )
            self.use_liger = False

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_liger and x.device.type != "cpu":
            return liger_rms_norm(x, self.weight, self.eps)

        output = self._norm(x.float()).type_as(x)
        return output * self.weight
