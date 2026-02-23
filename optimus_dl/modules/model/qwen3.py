"""
Qwen3 Language Model implementation.
Features Q/K normalization in attention, optional biases, and SwiGLU MLP.
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
class Qwen3Config(GPTConfig):
    """Configuration for Qwen3-style models."""

    sequence_length: int = field(
        default=32768,
        metadata={"description": "Maximum context length."},
    )
    rmsnorm_eps: float = field(
        default=1e-6,
        metadata={"description": "Epsilon for RMSNorm."},
    )
    rope_theta: float = field(
        default=1000000.0,
        metadata={"description": "Base frequency for rotary embeddings."},
    )
    rope_scaling: dict | None = field(
        default=None,
        metadata={"description": "RoPE scaling configuration."},
    )
    head_dim: int | None = field(
        default=None,
        metadata={
            "description": "Dimensionality of each attention head. If None, will be set to hidden_size // num_attention_heads."
        },
    )
    bias: bool = field(
        default=False,
        metadata={"description": "Global bias flag for linear layers."},
    )
    attention_bias: bool = field(
        default=True,
        metadata={"description": "Specific bias flag for attention projections."},
    )
    tie_word_embeddings: bool = field(
        default=True,
        metadata={"description": "Tie input and output embeddings."},
    )
    n_kv_head: int | None = field(
        default=None,
        metadata={
            "description": "Number of Key/Value heads. If None, will be set to num_attention_heads."
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


class Qwen3Block(RotaryTransformerBlock):
    """Qwen3 Transformer block with Q/K normalization."""

    def __init__(self, config: Qwen3Config):
        super().__init__(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            head_dim=config.head_dim,
            dropout=config.dropout,
            rmsnorm_eps=config.rmsnorm_eps,
            bias=config.bias,
            attention_bias=config.attention_bias,
            use_qk_norm=True,
            qk_norm_per_head=True,
            intermediate_size=config.intermediate_size,
            multiple_of=config.multiple_of,
            use_liger_rmsnorm=config.use_liger_rmsnorm,
            use_liger_swiglu=config.use_liger_swiglu,
        )


@register_model("qwen3", Qwen3Config)
class Qwen3(GPT):
    """Qwen3 Language Model architecture.

    Extends the framework's GPT base with Qwen-specific features:

    - **Q/K Normalization**: Applies RMSNorm to Query and Key tensors before
      attention computation to improve training stability.
    - **Configurable Biases**: Supports bias in attention and MLP layers.
    - **Large Context**: Optimized for very long sequence lengths.

    Args:
        config: Qwen3 model configuration.
    """

    def __init__(self, config: Qwen3Config, **kwargs):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config

        # create the token and position embeddings
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.n_embd // config.n_head
        )
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
                "h": nn.ModuleList([Qwen3Block(config) for _ in range(config.n_layer)]),
                "ln_f": RMSNorm(
                    config.n_embd,
                    eps=config.rmsnorm_eps,
                    use_liger=config.use_liger_rmsnorm,
                ),
            }
        )
        # Weight tying
        if config.tie_word_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

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
        """Apply Tensor Parallelism plan to the Qwen3 model.

        Similar to the Llama plan but handles Qwen3-specific parameter names
        and bias configurations.

        Args:
            mesh: DeviceMesh for sharding.
            loss_parallel: If True, shards the LM head.
            sequence_parallel: If True, enables sequence sharding.
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
        """Forward pass with rotary frequency selection."""
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
            block_kwargs = dict(x=x, freqs_cis=freqs_cis)
            if seq_lens is not None:
                block_kwargs["seq_lens"] = seq_lens
            x = block(**block_kwargs)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return {
            "logits": logits,
        }


@Qwen3.register_arch("0.6B")
def qwen3_0_6b():
    return Qwen3Config(
        n_layer=28,
        n_head=16,
        n_kv_head=8,
        n_embd=1024,
        head_dim=128,
        intermediate_size=3072,
        vocab_size=151936,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        sequence_length=40960,
        attention_bias=False,
    )


@Qwen3.register_arch("4B")
def qwen3_4b():
    return Qwen3Config(
        n_layer=36,
        n_head=32,
        n_kv_head=8,
        n_embd=2560,
        intermediate_size=9216,
        multiple_of=256,
        attention_bias=True,
    )
