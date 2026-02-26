import logging
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    Shard,
)

from optimus_dl.core.log import warn_once
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

# Try to import varlen_attn
try:
    from torch.nn.attention.varlen import varlen_attn

    VARLEN_ATTENTION_AVAILABLE = True
except ImportError:
    VARLEN_ATTENTION_AVAILABLE = False
    varlen_attn = None


def attention_mask_fn(
    b, _, q_idx, kv_idx, window_size=None, seq_lens=None, document_ids=None
):
    """Mask function for flex_attention supporting causal, sliding window, padding, and flat batching.

    Args:
        b: Batch index.
        _: Head index (unused).
        q_idx: Query index.
        kv_idx: Key/Value index.
        window_size: Optional sliding window size.
        seq_lens: Optional 1D tensor of sequence lengths (for padding).
        document_ids: Optional 2D tensor of document IDs (for flat/packed batching).
    """
    mask = q_idx >= kv_idx  # causal
    if window_size is not None:
        mask = mask & (q_idx - kv_idx < window_size)
    if seq_lens is not None:
        mask = mask & (q_idx < seq_lens[b]) & (kv_idx < seq_lens[b])
    if document_ids is not None:
        mask = mask & (document_ids[b, q_idx] == document_ids[b, kv_idx])
    return mask


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
    - **Dynamic Sequence Padding**: Support for `seq_lens` masking via flex_attention.
    - **Variable-length Attention**: Support for optimized Flash Attention on packed batches via `cu_seqlens`.

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

    def _varlen_attn_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """CPU fallback for varlen attention, used for testing and non-CUDA environments.

        Args:
            q: Flattened query (1, total_tokens, n_head, head_dim)
            k: Flattened key (1, total_tokens, n_kv_head, head_dim)
            v: Flattened value (1, total_tokens, n_kv_head, head_dim)
            cu_seqlens: Cumulative sequence lengths
            max_seqlen: Maximum sequence length

        Returns:
            Flattened attention output (1, total_tokens, n_head, head_dim)
        """
        device = q.device
        num_docs = len(cu_seqlens) - 1
        n_head = q.shape[2]
        n_kv_head = k.shape[2]
        head_dim = q.shape[3]

        # 1. Un-flatten into padded batch
        q_padded = torch.zeros(
            num_docs, max_seqlen, n_head, head_dim, device=device, dtype=q.dtype
        )
        k_padded = torch.zeros(
            num_docs, max_seqlen, n_kv_head, head_dim, device=device, dtype=k.dtype
        )
        v_padded = torch.zeros(
            num_docs, max_seqlen, n_kv_head, head_dim, device=device, dtype=v.dtype
        )

        for i in range(num_docs):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            length = end - start
            q_padded[i, :length] = q[0, start:end]
            k_padded[i, :length] = k[0, start:end]
            v_padded[i, :length] = v[0, start:end]

        # 2. Transpose for SDPA: (B, H, T, D)
        q_padded = q_padded.transpose(1, 2)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)

        # 3. Create mask
        # We need a mask of shape (B, 1, T, T)
        q_idx = torch.arange(max_seqlen, device=device).view(1, 1, -1, 1)
        kv_idx = torch.arange(max_seqlen, device=device).view(1, 1, 1, -1)

        doc_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).view(-1, 1, 1, 1)
        # Padding mask: doc attends only to valid tokens
        mask = (q_idx < doc_lens) & (kv_idx < doc_lens)

        # Causal mask
        mask &= q_idx >= kv_idx

        # Sliding window mask
        if self.sliding_window is not None:
            mask &= q_idx - kv_idx < self.sliding_window

        # 4. Compute attention
        y = torch.nn.functional.scaled_dot_product_attention(
            q_padded,
            k_padded,
            v_padded,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # Already handled in mask
            enable_gqa=(self.n_rep > 1),
        )

        # 5. Transpose back: (B, T, H, D)
        y = y.transpose(1, 2)

        # 6. Flatten back
        y_flat = torch.zeros_like(q)
        for i in range(num_docs):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            length = end - start
            y_flat[0, start:end] = y[i, :length]

        return y_flat

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Perform the forward pass with RoPE and GQA.

        Args:
            x: Input tensor of shape (B, T, C).
            freqs_cis: Precomputed frequencies for RoPE.
            seq_lens: Optional 1D tensor of sequence lengths to mask out padding.
            document_ids: Optional 2D tensor of document IDs for packed/flat batching.
            position_ids: Optional 2D tensor of position IDs for RoPE.
            cu_seqlens: Optional 1D tensor of cumulative sequence lengths for Flash Attention varlen.
            max_seqlen: Optional maximum sequence length in the packed batch.

        Returns:
            Output tensor after attention and projection.
        """
        B, T, C = x.size()

        # Input validation
        if cu_seqlens is not None:
            assert (
                B == 1
            ), f"cu_seqlens is only supported for flat batches (B=1), but got B={B}"
            assert (
                cu_seqlens.ndim == 1
            ), f"cu_seqlens must be a 1D tensor, got ndim={cu_seqlens.ndim}"
            assert (
                cu_seqlens[0] == 0
            ), f"cu_seqlens must start with 0, got {cu_seqlens[0]}"
            assert (
                cu_seqlens[-1] == T
            ), f"cu_seqlens[-1] ({cu_seqlens[-1]}) must match sequence length T ({T})"

        if document_ids is not None:
            assert document_ids.shape == (
                B,
                T,
            ), f"document_ids shape must be (B, T) = ({B}, {T}), got {document_ids.shape}"

        if position_ids is not None:
            assert position_ids.shape == (
                B,
                T,
            ), f"position_ids shape must be (B, T) = ({B}, {T}), got {position_ids.shape}"

        if seq_lens is not None:
            assert seq_lens.shape == (
                B,
            ), f"seq_lens shape must be (B,) = ({B},), got {seq_lens.shape}"

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

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis, position_ids=position_ids)

        is_dtensor = isinstance(xq, DTensor)
        if is_dtensor:
            # heads are sharded, so we get local heads to compute attention
            input_mesh = xq.device_mesh
            input_placements = xq.placements
            xq = xq.to_local()
            xk = xk.to_local()
            xv = xv.to_local()

        enable_gqa = self.n_rep > 1

        if cu_seqlens is not None:
            if VARLEN_ATTENTION_AVAILABLE and xq.is_cuda:
                # Flash Attention Varlen path
                # Reshape to (total_tokens, n_heads, head_dim)
                xq_varlen = xq.reshape(-1, self.n_head, self.head_dim)
                xk_varlen = xk.reshape(-1, self.n_kv_head, self.head_dim)
                xv_varlen = xv.reshape(-1, self.n_kv_head, self.head_dim)

                # Use provided max_seqlen or compute if missing (fallback)
                max_q = (
                    max_seqlen
                    if max_seqlen is not None
                    else int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
                )

                y = varlen_attn(
                    xq_varlen,
                    xk_varlen,
                    xv_varlen,
                    cu_seq_q=cu_seqlens,
                    cu_seq_k=cu_seqlens,
                    max_q=max_q,
                    max_k=max_q,
                    is_causal=True,
                )
                # Reshape back to (B, T, n_heads, head_dim)
                y = y.view(B, T, self.n_head, self.head_dim)
            else:
                # CPU fallback path for testing and non-CUDA devices
                # We need max_seqlen
                max_q = (
                    max_seqlen
                    if max_seqlen is not None
                    else int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
                )
                y = self._varlen_attn_fallback(
                    xq, xk, xv, cu_seqlens=cu_seqlens, max_seqlen=max_q
                )
        else:
            # SDPA or Flex Attention path
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # Decide if we can use flex_attention masks
            use_flex = (
                FLEX_ATTENTION_AVAILABLE
                and (
                    self.sliding_window is not None
                    or seq_lens is not None
                    or document_ids is not None
                )
                and (x.device.type in {"cuda", "cpu", "xpu", "hpu"})
            )

            if use_flex:
                mask_fn = partial(
                    attention_mask_fn,
                    window_size=self.sliding_window,
                    seq_lens=seq_lens,
                    document_ids=document_ids,
                )

                # Since seq_lens or document_ids relies on dynamic per-batch metadata, it needs the true batch dimension `B`
                if seq_lens is not None or document_ids is not None:
                    block_mask = create_block_mask(
                        mask_fn, B, None, T, T, device=x.device
                    )
                else:
                    if self._block_mask is None or self._block_mask.shape[-1] != T:
                        self._block_mask = create_block_mask(
                            mask_fn, None, None, T, T, device=x.device
                        )
                    block_mask = self._block_mask

                _flex_attention = flex_attention
                if xq.device.type == "cuda":
                    _flex_attention = torch.compile(flex_attention)

                if self.dropout > 1e-5:
                    warn_once(
                        logger=logger,
                        message="Dropout is not supported in flex attention. Ignoring dropout.",
                    )

                y = _flex_attention(
                    xq, xk, xv, block_mask=block_mask, enable_gqa=enable_gqa
                )
            else:
                mask = None
                if (
                    self.sliding_window is not None
                    or seq_lens is not None
                    or document_ids is not None
                ):
                    q_idx = torch.arange(T, device=x.device).view(1, 1, -1, 1)
                    kv_idx = torch.arange(T, device=x.device).view(1, 1, 1, -1)
                    mask = q_idx >= kv_idx

                    if self.sliding_window is not None:
                        mask &= q_idx - kv_idx < self.sliding_window

                    if seq_lens is not None:
                        seq_lens_view = seq_lens.view(-1, 1, 1, 1)
                        seq_lens_mask = (q_idx < seq_lens_view) & (
                            kv_idx < seq_lens_view
                        )
                        mask = torch.broadcast_to(mask, seq_lens_mask.shape)
                        mask = mask & seq_lens_mask

                    if document_ids is not None:
                        doc_ids_q = document_ids.view(B, 1, T, 1)
                        doc_ids_kv = document_ids.view(B, 1, 1, T)
                        doc_mask = doc_ids_q == doc_ids_kv
                        if mask is None:
                            mask = doc_mask
                        else:
                            mask = mask & doc_mask

                y = torch.nn.functional.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=(mask is None),
                    enable_gqa=enable_gqa,
                )
            y = y.transpose(1, 2)

        if is_dtensor:
            # if it was dtensor, then attention output has the same sharding scheme as input (head-sharded)
            y = DTensor.from_local(y, input_mesh, input_placements)

        y = y.contiguous().view(B, -1, self.n_head * self.head_dim)
        y = self.resid_dropout(self.wo(y))

        # if it was SP, keep it SP
        if is_sp:
            y = y.redistribute(sp_mesh, sp_placements)

        return y
