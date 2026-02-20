import logging

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    Shard,
)

from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.blocks.rope import apply_rotary_emb

logger = logging.getLogger(__name__)

# Try to import flex_attention
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None


def sliding_window_mask(_, __, q_idx, kv_idx, window_size):
    """Mask function for flex_attention sliding window."""
    return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)


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
    - **Sliding Window Attention**: Optional sliding window masking.

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
        qk_norm_per_head: bool = True,
        rmsnorm_eps: float = 1e-5,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = head_dim or n_embd // n_head
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm
        self.qk_norm_per_head = qk_norm_per_head
        self.sliding_window = sliding_window

        assert (
            self.n_head % self.n_kv_head == 0
        ), "n_head must be divisible by n_kv_head"

        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_head * self.head_dim, n_embd, bias=bias)

        if use_qk_norm:
            q_norm_dim = self.head_dim if qk_norm_per_head else n_head * self.head_dim
            k_norm_dim = (
                self.head_dim if qk_norm_per_head else self.n_kv_head * self.head_dim
            )
            self.q_norm = RMSNorm(q_norm_dim, eps=rmsnorm_eps, use_liger=False)
            self.k_norm = RMSNorm(k_norm_dim, eps=rmsnorm_eps, use_liger=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Flex attention block mask
        self._block_mask = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with RoPE and GQA.

        Args:
            x: Input tensor.
            freqs_cis: Precomputed frequencies for RoPE.

        Returns:
            Output tensor after attention and projection.
        """
        B, T, C = x.size()

        # Infer if we are in SP mode. It is expected for input to be correct sequence-sharded DTensor
        is_sp = isinstance(x, DTensor) and any(
            isinstance(p, Shard) and p.dim == 1 for p in x.placements
        )

        if is_sp:
            # If sequence parallel, attention is a global operation. We gather full sequence context before computing attention.
            sp_placements = x.placements
            sp_mesh = x.device_mesh
            x = x.redistribute(placements=[Replicate()])

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # If TP, then Q K V are sharded across heads

        if self.use_qk_norm and not self.qk_norm_per_head:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)

        if self.use_qk_norm and self.qk_norm_per_head:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        is_dtensor = isinstance(xq, DTensor)
        if is_dtensor:
            # heads are sharded, so we get local heads to compute attention
            input_mesh = xq.device_mesh
            input_placements = xq.placements
            xq = xq.to_local()
            xk = xk.to_local()
            xv = xv.to_local()

        enable_gqa = self.n_rep > 1
        if self.sliding_window is not None and FLEX_ATTENTION_AVAILABLE:
            if self._block_mask is None or self._block_mask.shape[-1] != T:
                from functools import partial

                mask_fn = partial(sliding_window_mask, window_size=self.sliding_window)
                self._block_mask = create_block_mask(
                    mask_fn, None, None, T, T, device=x.device
                )
            _flex_attention = flex_attention
            if xq.device.type == "cuda":
                _flex_attention = torch.compile(flex_attention)

            y = _flex_attention(
                xq, xk, xv, block_mask=self._block_mask, enable_gqa=enable_gqa
            )
        else:
            mask = None
            if self.sliding_window is not None:
                q_idx = torch.arange(T, device=x.device).view(-1, 1)
                kv_idx = torch.arange(T, device=x.device).view(1, -1)
                mask = (q_idx >= kv_idx) & (q_idx - kv_idx < self.sliding_window)

            y = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(self.sliding_window is None),
                enable_gqa=enable_gqa,
            )

        if is_dtensor:
            # if it was dtensor, then attention output has the same sharding scheme as input (head-sharded)
            y = DTensor.from_local(y, input_mesh, input_placements)

        y = y.transpose(1, 2).contiguous().view(B, -1, self.n_head * self.head_dim)
        y = self.resid_dropout(self.wo(y))

        # if it was SP, keep it SP
        if is_sp:
            y = y.redistribute(sp_mesh, sp_placements)

        return y
