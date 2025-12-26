import logging

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.blocks.rope import apply_rotary_emb

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention layer as used in GPT-2.

    Includes support for dropout and causal masking.

    Attributes:
        c_attn: Combined Linear layer for query, key, and value projections.
        c_proj: Linear layer for output projection.
        n_head: Number of attention heads.
        n_embd: Embedding dimensionality.
        dropout: Dropout probability.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class RotarySelfAttention(nn.Module):
    """Generalized Rotary Self-Attention.

    Supports several modern features:

    - **Grouped Query Attention (GQA)**: For improved inference efficiency.
    - **Rotary Positional Embeddings (RoPE)**: For better positional encoding.
    - **Q/K Normalization**: Optional RMSNorm on Query/Key for training stability.

    Attributes:
        wq: Linear projection for Query.
        wk: Linear projection for Key.
        wv: Linear projection for Value.
        wo: Linear projection for Output.
        q_norm: Optional RMSNorm for Query.
        k_norm: Optional RMSNorm for Key.
        n_head: Number of Query heads.
        n_kv_head: Number of Key/Value heads.
        head_dim: Dimensionality of each head.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_qk_norm: bool = False,
        rmsnorm_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = head_dim or n_embd // n_head
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm

        assert (
            self.n_head % self.n_kv_head == 0
        ), "n_head must be divisible by n_kv_head"

        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_head * self.head_dim, n_embd, bias=bias)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rmsnorm_eps, use_liger=False)
            self.k_norm = RMSNorm(self.head_dim, eps=rmsnorm_eps, use_liger=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with RoPE and GQA.

        Args:
            x: Input tensor.
            freqs_cis: Precomputed frequencies for RoPE.

        Returns:
            Output tensor after attention and projection.
        """
        B, T, C = x.size()

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.n_rep > 1:
            xk = (
                xk[:, :, None, :, :]
                .expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                .reshape(B, self.n_head, T, self.head_dim)
            )
            xv = (
                xv[:, :, None, :, :]
                .expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                .reshape(B, self.n_head, T, self.head_dim)
            )

        force_make_dtensor = False
        force_make_dtensor_mesh = None
        if str(xq.device.type) == "cpu" and isinstance(xq, DTensor):
            force_make_dtensor_mesh = xq.device_mesh
            xq = xq.to_local()
            xk = xk.to_local()
            xv = xv.to_local()
            force_make_dtensor = True

        y = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None, dropout_p=self.dropout, is_causal=True
        )

        if force_make_dtensor and not isinstance(y, DTensor):
            y = DTensor.from_local(y, force_make_dtensor_mesh, (Shard(1),))

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.resid_dropout(self.wo(y))
        return y
