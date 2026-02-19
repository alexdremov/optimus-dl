"""
Olmo3 Language Model implementation.
Features alternating sliding window and full attention, YaRN RoPE, and SwiGLU MLP.
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
from optimus_dl.modules.model.blocks.attention import RotarySelfAttention
from optimus_dl.modules.model.blocks.layer_norms import RMSNorm
from optimus_dl.modules.model.blocks.mlp import SwiGLUMLP
from optimus_dl.modules.model.blocks.rope import precompute_freqs_cis
from optimus_dl.modules.model.gpt2 import (
    GPT,
    GPTConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class Olmo3Config(GPTConfig):
    """Configuration for Olmo3-style models."""

    sequence_length: int = field(
        default=65536,
        metadata={"description": "Maximum context length."},
    )
    rmsnorm_eps: float = field(
        default=1e-6,
        metadata={"description": "Epsilon for RMSNorm."},
    )
    rope_theta: float = field(
        default=500000.0,
        metadata={"description": "Base frequency for rotary embeddings."},
    )
    rope_scaling: dict | None = field(
        default=None,
        metadata={"description": "RoPE scaling configuration (e.g., YaRN)."},
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
        default=False,
        metadata={"description": "Specific bias flag for attention projections."},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"description": "Tie input and output embeddings."},
    )
    n_kv_head: int | None = field(
        default=8,
        metadata={
            "description": "Number of Key/Value heads. If None, will be set to num_attention_heads."
        },
    )
    intermediate_size: int | None = field(
        default=27648,
        metadata={"description": "Dimension of SwiGLU hidden layer."},
    )
    multiple_of: int = field(
        default=256,
        metadata={
            "description": "Make SwiGLU hidden layer size multiple of large power of 2"
        },
    )
    sliding_window: int = field(
        default=4096,
        metadata={"description": "Window size for sliding window attention."},
    )
    layer_types: list[str] = field(
        default_factory=lambda: [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        * 16,
        metadata={"description": "List of attention types for each layer."},
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


class Olmo3Attention(RotarySelfAttention):
    """Olmo3 Attention supporting sliding window via flex_attention and Q/K normalization."""

    def __init__(self, config: Olmo3Config, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx]
        sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )
        super().__init__(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            head_dim=config.head_dim,
            dropout=config.dropout,
            bias=config.attention_bias,
            use_qk_norm=True,
            qk_norm_per_head=False,  # Olmo3 uses across-heads norm
            rmsnorm_eps=config.rmsnorm_eps,
            sliding_window=sliding_window,
        )


class Olmo3Block(nn.Module):
    """Olmo3 Transformer block."""

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.attn = Olmo3Attention(config, layer_idx)
        self.ln_1 = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )
        self.mlp = SwiGLUMLP(
            n_embd=config.n_embd,
            intermediate_size=config.intermediate_size,
            multiple_of=config.multiple_of,
            bias=False,
            use_liger=config.use_liger_swiglu,
        )
        self.ln_2 = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )

    def forward(self, x, freqs_cis):
        # x = x + Norm(attn(x))
        attn = self.attn(x, freqs_cis)
        x = x + self.ln_1(attn)
        # x = x + Norm(mlp(x))
        x = x + self.ln_2(self.mlp(x))
        return x


@register_model("olmo3", Olmo3Config)
class Olmo3(GPT):
    """Olmo3 Language Model architecture."""

    def __init__(self, config: Olmo3Config, **kwargs):
        super().__init__(config)
        self.config = config

        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.n_embd // config.n_head
        )
        # Precompute two sets of frequencies
        self.freqs_cis_sliding = precompute_freqs_cis(
            self.head_dim,
            config.sequence_length,
            theta=config.rope_theta,
            scaling_config=None,  # Standard RoPE for sliding attention
        )
        self.freqs_cis_full = precompute_freqs_cis(
            self.head_dim,
            config.sequence_length,
            theta=config.rope_theta,
            scaling_config=config.rope_scaling,  # YaRN for full attention
        )

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [Olmo3Block(config, i) for i in range(config.n_layer)]
                ),
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

    def forward(self, input_ids, **kwargs):
        idx = input_ids
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        freqs_cis_sliding = self.freqs_cis_sliding.to(x.device)[pos]
        freqs_cis_full = self.freqs_cis_full.to(x.device)[pos]

        for block in self.transformer.h:
            # Each block knows its type
            if block.attn.layer_type == "sliding_attention":
                x = block(x, freqs_cis_sliding)
            else:
                x = block(x, freqs_cis_full)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return {"logits": logits}

    def apply_tp(
        self, mesh, loss_parallel: bool = False, sequence_parallel: bool = False
    ):
        """Apply Tensor Parallelism plan to the Olmo3 model."""
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
            RowwiseParallel,
            SequenceParallel,
            parallelize_module,
        )

        tp_size = mesh.size(0)
        n_kv_head = (
            self.config.n_kv_head
            if self.config.n_kv_head is not None
            else self.config.n_head
        )

        assert self.config.n_head % tp_size == 0
        assert n_kv_head % tp_size == 0

        layer_plan = {
            "transformer.wte": RowwiseParallel(input_layouts=Replicate()),
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
                        use_local_output=False,
                    ),
                    "transformer.h.*.ln_1": SequenceParallel(),
                    "transformer.h.*.ln_2": SequenceParallel(),
                    "transformer.ln_f": SequenceParallel(),
                    "transformer.h.*": PrepareModuleInput(
                        input_layouts=(Shard(1), Replicate()),
                        desired_input_layouts=(Shard(1), Replicate()),
                        use_local_output=False,
                    ),
                    "transformer.h.*.attn.wo": RowwiseParallel(
                        output_layouts=Shard(1), use_local_output=False
                    ),
                    "transformer.h.*.mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Shard(1),),
                        use_local_output=False,
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
