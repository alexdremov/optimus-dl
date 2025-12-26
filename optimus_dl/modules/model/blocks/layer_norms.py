import logging

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)

try:
    from liger_kernel.transformers.functional import liger_rms_norm

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    liger_rms_norm = None

LIGER_AVAILABLE = False
liger_rms_norm = None


class LayerNorm(nn.Module):
    """LayerNorm with optional bias.

    PyTorch's standard LayerNorm always expects a bias if elementwise_affine
    is True. This implementation allows for a more flexible bias=False option
    as seen in some LLM architectures.

    Attributes:
        weight: Affine scale parameter.
        bias: Optional affine bias parameter.
    """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            input: Input tensor.

        Returns:
            Normalized tensor.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm is a simplification of LayerNorm that only scales the input by the
    root mean square of the activations, omitting the mean subtraction and
    bias.

    Args:
        dim: Input dimension.
        eps: Small value for numerical stability.
        use_liger: If True, uses the high-performance Liger kernel. If None,
            automatically enables if available.
    """

    def __init__(self, dim: int, eps: float = 1e-6, use_liger: bool | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        if use_liger is None:
            self.use_liger = LIGER_AVAILABLE
            if self.use_liger:
                logger.info("Using liger-kernel for RMSNorm.")
        else:
            self.use_liger = use_liger

        if self.use_liger and not LIGER_AVAILABLE:
            logger.warning(
                "Liger Kernel requested for RMSNorm but not installed. Fallback to PyTorch."
            )
            self.use_liger = False

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the RMS normalization of the input."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            x: Input tensor.

        Returns:
            RMS normalized tensor.
        """
        is_dtensor = isinstance(x, DTensor)

        if self.use_liger and x.device.type != "cpu" and not is_dtensor:
            return liger_rms_norm(x, self.weight, self.eps)

        output = self._norm(x.float()).type_as(x)

        weight = self.weight
        if is_dtensor and not isinstance(weight, DTensor):
            from torch.distributed.tensor.placement_types import Replicate

            # If x is DTensor, weight must be DTensor for multiplication.
            # We assume weight is replicated (available on all ranks).
            weight = DTensor.from_local(weight, x.device_mesh, (Replicate(),))

        return output * weight
