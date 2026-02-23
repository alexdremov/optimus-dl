"""
Llama style Language Model.
References:

- Llama inference code:
https://github.com/facebookresearch/llama/blob/main/llama/model.py
- Mistral one file ref:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
- Llama paper:
https://arxiv.org/pdf/2302.13971.pdf

Main differences from GPT2:
- Uses RMSNorm instead of LayerNorm
- Uses a slightly different MLP (SwiGLU)
- rotary embeddings (RoPE)
"""

import logging
import math
from dataclasses import (
    dataclass,
    field,
)

import torch
import torch.nn as nn
from torch.distributed.tensor.placement_types import (
    Replicate,
    Shard,
)

from optimus_dl.modules.model import register_model
from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.blocks.rope import precompute_freqs_cis
from optimus_dl.modules.model.blocks.transformer import RotaryTransformerBlock
from optimus_dl.modules.model.gpt2 import (
    GPT,
    GPTConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class LlamaConfig(GPTConfig):
    """Configuration for Llama-style models."""

    sequence_length: int = field(
        default=16000,
        metadata={"description": "Maximum context length."},
    )
    rmsnorm_eps: float = field(
        default=1e-5,
        metadata={"description": "Epsilon for RMSNorm."},
    )
    bias: bool = field(
        default=False,
        metadata={"description": "Whether to use bias (usually False for Llama)."},
    )
    attention_bias: bool = field(
        default=False,
        metadata={"description": "Specific bias flag for attention projections."},
    )
    tie_word_embeddings: bool = field(
        default=True,
        metadata={"description": "Whether to tie input and output embeddings."},
    )
    n_kv_head: int | None = field(
        default=None,
        metadata={
            "description": "Number of Key/Value heads (for GQA). If None, will be set to num_attention_heads."
        },
    )
    intermediate_size: int | None = field(
        default=None,
        metadata={
            "description": "Dimension of SwiGLU hidden layer. If None, will be set based on multiple_of"
        },
    )
    multiple_of: int = field(
        default=256,
        metadata={
            "description": "Make SwiGLU hidden layer size multiple of large power of 2"
        },
    )
    rope_theta: float = field(
        default=10000.0,
        metadata={"description": "Base frequency for rotary embeddings."},
    )
    rope_scaling: dict | None = field(
        default=None,
        metadata={"description": "RoPE scaling configuration."},
    )
    use_liger_rmsnorm: bool | None = field(
        default=None,
        metadata={
            "description": "Enable Liger-kernel for RMSNorm. None = auto-enable if available."
        },
    )
    use_liger_swiglu: bool | None = field(
        default=None,
        metadata={
            "description": "Enable Liger-kernel for SwiGLU. None = auto-enable if available."
        },
    )


class LlamaBlock(RotaryTransformerBlock):
    """Llama Transformer block with RMSNorm, Rotary Attention, and SwiGLU MLP."""

    def __init__(self, config: LlamaConfig):
        super().__init__(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            dropout=config.dropout,
            rmsnorm_eps=config.rmsnorm_eps,
            bias=config.bias,
            attention_bias=config.attention_bias,
            use_qk_norm=False,
            intermediate_size=config.intermediate_size,
            multiple_of=config.multiple_of,
            use_liger_rmsnorm=config.use_liger_rmsnorm,
            use_liger_swiglu=config.use_liger_swiglu,
        )


@register_model("llama2", LlamaConfig)
class Llama(GPT):
    """Llama Language Model architecture.

    Based on the standard GPT class but incorporates modern architectural
    improvements:

    - **Rotary Embeddings (RoPE)**: Position encoding integrated into attention.
    - **RMSNorm**: More efficient normalization layer.
    - **SwiGLU MLP**: SiLU-gated MLP variant.
    - **Tensor Parallelism**: Comprehensive sharding plan for distributed training.

    Args:
        config: Llama model configuration.
    """

    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.sequence_length,
            theta=config.rope_theta,
            scaling_config=config.rope_scaling,
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
        """Apply a 1D Tensor Parallelism plan to the Llama model.

        Shards attention (Q/K/V/O) and MLP (w1/w2/c_proj) layers across the
        provided device mesh. Supports optional sequence parallelism for norms
        and communication-efficient sharded loss.

        Args:
            mesh: DeviceMesh for sharding.
            loss_parallel: If True, shards the LM head and uses loss_parallel.
            sequence_parallel: If True, enables sequence sharding and sharded norms.
        """
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
                    "transformer.h.*": PrepareModuleInput(
                        input_kwarg_layouts=dict(
                            x=Shard(1),
                            freqs_cis=Replicate(),
                            seq_lens=Replicate(),
                        ),
                        desired_input_kwarg_layouts=dict(
                            x=Shard(1),
                            freqs_cis=Replicate(),
                            seq_lens=Replicate(),
                        ),
                        use_local_output=False,
                    ),
                    "transformer.h.*.attn.wo": RowwiseParallel(
                        output_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.mlp.w1": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.mlp.w2": ColwiseParallel(
                        input_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.mlp.c_proj": RowwiseParallel(
                        output_layouts=Shard(1), use_local_output=False
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

    def forward(self, input_ids, seq_lens: torch.Tensor | None = None, **kwargs):
        """Perform the forward pass, handling rotary frequency lookup."""
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
            block_kwargs = dict(x=x, freqs_cis=freqs_cis)
            if seq_lens is not None:
                block_kwargs["seq_lens"] = seq_lens
            x = block(**block_kwargs)
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
