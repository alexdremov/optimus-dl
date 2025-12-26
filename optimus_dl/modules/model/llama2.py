"""
Llama style Language Model.
References:
1) Llama inference code:
https://github.com/facebookresearch/llama/blob/main/llama/model.py
2) Mistral one file ref:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
3) Llama paper:
https://arxiv.org/pdf/2302.13971.pdf

Main differences from GPT2:
* Uses RMSNorm instead of LayerNorm
* Uses a slightly different MLP (SwiGLU)
* rotary embeddings (RoPE)
"""

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from optimus_dl.modules.model import register_model
from optimus_dl.modules.model.blocks.attention import CausalSelfAttention
from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.gpt2 import GPT, GPTConfig

logger = logging.getLogger(__name__)

try:
    from liger_kernel.transformers.functional import liger_swiglu

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    liger_swiglu = None


@dataclass
class LlamaConfig(GPTConfig):
    sequence_length: int = 16000
    rmsnorm_eps: float = 1e-5
    bias: bool = False
    tie_word_embeddings: bool = True
    n_kv_head: int | None = None
    intermediate_size: int | None = None
    multiple_of: int = field(
        default=256,
        metadata={"help": "make SwiGLU hidden layer size multiple of large power of 2"},
    )
    # Liger Kernel flags (None = auto-enable if available)
    use_liger_rmsnorm: bool | None = None
    use_liger_swiglu: bool | None = None


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    # Stack the cos and sin parts in the last dimension to simulate complex numbers
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (
        x.shape[1],
        x.shape[-2],
    ), f"{freqs_cis.shape = }, {x.shape = }"
    # New shape for broadcasting
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)

    # Handle DTensor (Tensor Parallelism)
    # We perform RoPE on local shards to avoid complex sharding propagation issues with reshape/select.
    is_q_dtensor = isinstance(q, DTensor)
    is_k_dtensor = isinstance(k, DTensor)
    is_freqs_cis_dtensor = isinstance(freqs_cis, DTensor)

    q_in = q.to_local() if is_q_dtensor else q
    k_in = k.to_local() if is_k_dtensor else k
    freqs_cis = freqs_cis.to_local() if is_freqs_cis_dtensor else freqs_cis

    q_in = q_in.float().reshape(*q_in.shape[:-1], -1, 2)
    k_in = k_in.float().reshape(*k_in.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q_in)

    # Perform manual "complex" multiplication
    q_cos = q_in[..., 0] * freqs_cis[..., 0] - q_in[..., 1] * freqs_cis[..., 1]
    q_sin = q_in[..., 0] * freqs_cis[..., 1] + q_in[..., 1] * freqs_cis[..., 0]
    k_cos = k_in[..., 0] * freqs_cis[..., 0] - k_in[..., 1] * freqs_cis[..., 1]
    k_sin = k_in[..., 0] * freqs_cis[..., 1] + k_in[..., 1] * freqs_cis[..., 0]

    # Combine the results back into the interleaved format expected by q and k
    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q_in.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k_in.shape).flatten(3)

    # Wrap back to DTensor if inputs were DTensor
    if is_q_dtensor:
        q_out = DTensor.from_local(q_out, q.device_mesh, q.placements)
    if is_k_dtensor:
        k_out = DTensor.from_local(k_out, k.device_mesh, k.placements)

    return q_out, k_out


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        if config.intermediate_size is not None:
            hidden_dim = config.intermediate_size
        else:
            hidden_dim = config.n_embd * 4
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = config.multiple_of * (
                (hidden_dim + config.multiple_of - 1) // config.multiple_of
            )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

        if config.use_liger_swiglu is None:
            self.use_liger = LIGER_AVAILABLE
            if self.use_liger:
                logger.info("Using liger-kernel for SwiGLU.")
        else:
            self.use_liger = config.use_liger_swiglu

        if self.use_liger and not LIGER_AVAILABLE:
            logger.warning(
                "Liger SwiGLU requested but not installed. Fallback to PyTorch."
            )
            self.use_liger = False

    def forward(self, x):
        if self.use_liger and x.device.type != "cpu":
            x_swiglu = liger_swiglu(self.w1(x), self.w2(x))
        else:
            x_swiglu = nn.functional.silu(self.w1(x)) * self.w2(x)

        return self.c_proj(x_swiglu)


class LlamaAttention(CausalSelfAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.n_head = config.n_head
        self.n_kv_head = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        )
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        # We don't use the parent's c_attn/c_proj for Llama with potential GQA
        # Delete them to avoid confusion/unused parameters
        del self.c_attn
        del self.c_proj

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.size()

        # (B, T, n_head * head_dim)
        xq = self.wq(x)
        # (B, T, n_kv_head * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, T, n_head, head_dim)
        xq = xq.view(B, T, -1, self.head_dim)
        xk = xk.view(B, T, -1, self.head_dim)
        xv = xv.view(B, T, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # (B, n_head, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Repeat K/V heads if n_kv_head < n_head (GQA)
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
            assert isinstance(xk, DTensor)
            assert isinstance(xv, DTensor)
            xq = xq.to_local()
            xk = xk.to_local()
            xv = xv.to_local()
            force_make_dtensor = True

        y = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None, dropout_p=self.dropout, is_causal=True
        )

        if force_make_dtensor and not isinstance(y, DTensor):
            y = DTensor.from_local(y, force_make_dtensor_mesh, (Shard(1),))

        # (B, T, n_head * head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.wo(y))
        return y


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )
        self.attn = LlamaAttention(config)
        self.ln_2 = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )
        self.mlp = LlamaMLP(config)

    def forward(self, x, freqs_cis):
        ln_1 = self.ln_1(x)
        attn_out = self.attn(ln_1, freqs_cis)

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


@register_model("llama2", LlamaConfig)
class Llama(GPT):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)]),
                "ln_f": RMSNorm(
                    config.n_embd,
                    eps=config.rmsnorm_eps,
                    use_liger=config.use_liger_rmsnorm,
                ),
            }
        )
        # Weight tying:
        # When using torch.compile(), PyTorch may emit a UserWarning about multiple values
        # for tied weights. This is a known behavior when tying weights for FSDP/compilation
        # compatibility and is generally safe to ignore.
        if config.tie_word_embeddings:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def apply_tp(
        self, mesh, loss_parallel: bool = False, sequence_parallel: bool = False
    ):
        tp_size = mesh.size(0)
        assert (
            self.config.n_head % tp_size == 0
        ), f"Number of heads ({self.config.n_head}) must be divisible by TP size ({tp_size})"
        n_kv_head = (
            self.config.n_kv_head
            if self.config.n_kv_head is not None
            else self.config.n_head
        )
        assert (
            n_kv_head % tp_size == 0
        ), f"Number of KV heads ({n_kv_head}) must be divisible by TP size ({tp_size})"

        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
            RowwiseParallel,
            SequenceParallel,
            parallelize_module,
        )

        layer_plan = {
            "transformer.wte": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "transformer.h.*.attn.wq": ColwiseParallel(use_local_output=False),
            "transformer.h.*.attn.wk": ColwiseParallel(use_local_output=False),
            "transformer.h.*.attn.wv": ColwiseParallel(use_local_output=False),
            "transformer.h.*.attn.wo": RowwiseParallel(),
            "transformer.h.*.mlp.w1": ColwiseParallel(use_local_output=False),
            "transformer.h.*.mlp.w2": ColwiseParallel(use_local_output=False),
            "transformer.h.*.mlp.c_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(use_local_output=False),
        }
        if sequence_parallel:
            layer_plan.update(
                {
                    "transformer.wte": RowwiseParallel(
                        input_layouts=Replicate(),
                        output_layouts=Shard(1),
                        use_local_output=True,
                    ),
                    "transformer.h.*.ln_1": SequenceParallel(),
                    "transformer.h.*.ln_2": SequenceParallel(),
                    "transformer.ln_f": SequenceParallel(),
                    "transformer.h.*.attn": PrepareModuleInput(
                        input_layouts=(Shard(1), Replicate()),
                        desired_input_layouts=(Shard(1), Replicate()),
                        use_local_output=False,
                    ),
                    "transformer.h.*.attn.wq": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.attn.wk": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.attn.wv": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.attn.wo": RowwiseParallel(output_layouts=Shard(1)),
                    "transformer.h.*.mlp.w1": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.mlp.w2": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.mlp.c_proj": RowwiseParallel(
                        output_layouts=Shard(1)
                    ),
                    "lm_head": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                }
            )

        parallelize_module(self, mesh, layer_plan)

        if self.config.tie_word_embeddings:
            # re-tie
            self.transformer.wte.weight = self.lm_head.weight

        if not loss_parallel:
            parallelize_module(
                self.lm_head,
                mesh,
                PrepareModuleOutput(
                    output_layouts=Shard(2),
                    desired_output_layouts=Replicate(),
                    use_local_output=False,
                ),
            )

    def forward(self, input_ids, **kwargs):
        idx = input_ids
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        for _block_idx, block in enumerate(self.transformer.h):
            x = block(x, freqs_cis)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return {
            "logits": logits,
        }


@Llama.register_arch("7b")
def llama_7b():
    return LlamaConfig(
        n_layer=32,
        n_head=32,
        n_embd=4096,
        multiple_of=256,
    )


@Llama.register_arch("1b")
def llama_1b():
    return LlamaConfig(
        n_layer=18,
        n_head=32,
        n_embd=2048,
        multiple_of=4,
    )


@Llama.register_arch("210M")
def llama_210M():
    return LlamaConfig(
        n_layer=24,
        n_head=12,
        n_embd=768,
        multiple_of=4,
    )


@Llama.register_arch("lite")
def llama_lite():
    return LlamaConfig(
        n_layer=6,
        n_head=8,
        n_embd=768,
        multiple_of=4,
    )


@Llama.register_arch("x-lite")
def llama_x_lite():
    return LlamaConfig(
        n_layer=6,
        n_head=4,
        n_embd=256,
        multiple_of=4,
    )
