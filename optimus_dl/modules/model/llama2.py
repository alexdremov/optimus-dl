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
from torch.distributed.tensor.placement_types import Replicate, Shard

from optimus_dl.modules.model import register_model
from optimus_dl.modules.model.blocks.attention import RotarySelfAttention
from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.blocks.mlp import SwiGLUMLP
from optimus_dl.modules.model.blocks.rope import precompute_freqs_cis
from optimus_dl.modules.model.gpt2 import GPT, GPTConfig

logger = logging.getLogger(__name__)


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
    rope_theta: float = 10000.0
    # Liger Kernel flags (None = auto-enable if available)
    use_liger_rmsnorm: bool | None = None
    use_liger_swiglu: bool | None = None


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.ln_1 = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )
        self.attn = RotarySelfAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            dropout=config.dropout,
            bias=False,  # Llama typically uses bias=False
            use_qk_norm=False,
        )
        self.ln_2 = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )
        self.mlp = SwiGLUMLP(
            n_embd=config.n_embd,
            intermediate_size=config.intermediate_size,
            multiple_of=config.multiple_of,
            bias=False,
            use_liger=config.use_liger_swiglu,
        )

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
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, config.sequence_length, theta=config.rope_theta
        )

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
        if config.tie_word_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
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
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        for block in self.transformer.h:
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
