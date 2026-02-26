import torch
import torch.nn as nn

from optimus_dl.modules.model.blocks.attention import RotarySelfAttention
from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.blocks.mlp import SwiGLUMLP


class RotaryTransformerBlock(nn.Module):
    """Unified Transformer block with RMSNorm, Rotary Attention, and SwiGLU MLP.

    Used by Llama and Qwen models. Supports optional Q/K normalization.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        rmsnorm_eps: float = 1e-5,
        bias: bool = False,
        attention_bias: bool = False,
        use_qk_norm: bool = False,
        qk_norm_per_head: bool = True,
        intermediate_size: int | None = None,
        multiple_of: int = 256,
        use_liger_rmsnorm: bool | None = None,
        use_liger_swiglu: bool | None = None,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd, eps=rmsnorm_eps, use_liger=use_liger_rmsnorm)
        self.attn = RotarySelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            dropout=dropout,
            bias=attention_bias,
            use_qk_norm=use_qk_norm,
            qk_norm_per_head=qk_norm_per_head,
            rmsnorm_eps=rmsnorm_eps,
        )
        self.ln_2 = RMSNorm(n_embd, eps=rmsnorm_eps, use_liger=use_liger_rmsnorm)
        self.mlp = SwiGLUMLP(
            n_embd=n_embd,
            intermediate_size=intermediate_size,
            multiple_of=multiple_of,
            bias=bias,
            use_liger=use_liger_swiglu,
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Compute the forward pass for the transformer block (pre-norm residual)."""
        ln_1 = self.ln_1(x)
        attn_out = self.attn(
            ln_1,
            freqs_cis=freqs_cis,
            seq_lens=seq_lens,
            document_ids=document_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x
