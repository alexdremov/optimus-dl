import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from liger_kernel.transformers.functional import liger_swiglu

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    liger_swiglu = None


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP variant used in Llama, Qwen, and Mistral.

    Consists of three linear layers (gate, up, down) and a SiLU (Swish)
    activation. Supports optional Liger kernel for performance.

    Attributes:
        w1: Gate projection layer.
        w2: Up projection layer.
        c_proj: Down projection layer.
        use_liger: Whether Liger kernel is enabled.
    """

    def __init__(
        self,
        n_embd: int,
        intermediate_size: int | None = None,
        multiple_of: int = 256,
        bias: bool = False,
        use_liger: bool | None = None,
    ):
        super().__init__()

        if intermediate_size is not None:
            hidden_dim = intermediate_size
        else:
            hidden_dim = n_embd * 4
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=bias)

        if use_liger is None:
            self.use_liger = LIGER_AVAILABLE
            if self.use_liger:
                logger.info("Using liger-kernel for SwiGLU.")
        else:
            self.use_liger = use_liger

        if self.use_liger and not LIGER_AVAILABLE:
            logger.warning(
                "Liger SwiGLU requested but not installed. Fallback to PyTorch."
            )
            self.use_liger = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        if self.use_liger and x.device.type != "cpu":
            x_swiglu = liger_swiglu(self.w1(x), self.w2(x))
        else:
            x_swiglu = nn.functional.silu(self.w1(x)) * self.w2(x)

        return self.c_proj(x_swiglu)


class GELUMLP(nn.Module):
    """Standard GPT-2 style MLP with GELU activation.

    Consists of an expansion layer, GELU activation, and a contraction layer.

    Attributes:
        c_fc: Expansion projection layer.
        gelu: GELU activation layer.
        c_proj: Contraction projection layer.
        dropout: Dropout layer.
    """

    def __init__(
        self,
        n_embd: int,
        intermediate_size: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Default GPT-2 expansion is 4x
        hidden_dim = intermediate_size if intermediate_size is not None else 4 * n_embd

        self.c_fc = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
