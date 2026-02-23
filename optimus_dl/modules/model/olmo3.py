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
        default=4096,
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
    rope_parameters: dict = field(
        default_factory=lambda: {
            "rope_type": "default",
        },
        metadata={"description": "Full RoPE configuration dictionary."},
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
        default=4,
        metadata={
            "description": "Number of Key/Value heads. If None, will be set to num_attention_heads."
        },
    )
    intermediate_size: int | None = field(
        default=1024,
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
    n_layer: int = field(
        default=16, metadata={"description": "Number of transformer blocks"}
    )
    layer_types: list[str] = field(
        default_factory=lambda: [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        * 4,
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
    """Olmo3 Attention supporting sliding window and Q/K normalization."""

    def __init__(self, config: Olmo3Config, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx
        assert self.layer_type in ("sliding_attention", "full_attention")
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
    """Olmo3 Transformer block.

    Architecture:
    x = x + Norm(Attn(x))
    x = x + Norm(MLP(x))
    """

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.attn = Olmo3Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )
        self.mlp = SwiGLUMLP(
            n_embd=config.n_embd,
            intermediate_size=config.intermediate_size,
            multiple_of=config.multiple_of,
            bias=False,
            use_liger=config.use_liger_swiglu,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.n_embd, eps=config.rmsnorm_eps, use_liger=config.use_liger_rmsnorm
        )

    def forward(self, x, freqs_cis, seq_lens: torch.Tensor | None = None):
        # x = x + Norm(attn(x))
        x = x + self.post_attention_layernorm(
            self.attn(x, freqs_cis=freqs_cis, seq_lens=seq_lens)
        )
        # x = x + Norm(mlp(x))
        x = x + self.post_feedforward_layernorm(self.mlp(x))
        return x


@register_model("olmo3", Olmo3Config)
class Olmo3(GPT):
    """Olmo3 Language Model architecture."""

    def __init__(self, config: Olmo3Config, **kwargs):
        super().__init__(config)
        self.config = config

        assert config.n_layer == len(
            self.config.layer_types
        ), "Number of layers must match the length of layer_types"

        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.n_embd // config.n_head
        )

        # Olmo3 uses a single rotary embedding for the entire model
        rope_params = config.rope_parameters.copy()
        if "rope_theta" not in rope_params:
            rope_params["rope_theta"] = config.rope_theta

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.sequence_length,
            theta=rope_params["rope_theta"],
            scaling_config=rope_params,
        )

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    config.vocab_size,
                    config.n_embd,
                    padding_idx=config.padding_token_id,
                ),
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

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        if config.tie_word_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

    def forward(self, input_ids, seq_lens: torch.Tensor | None = None, **kwargs):
        idx = input_ids
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[pos]

        for block in self.transformer.h:
            block_kwargs = dict(x=x, freqs_cis=freqs_cis)
            if seq_lens is not None:
                block_kwargs["seq_lens"] = seq_lens
            x = block(**block_kwargs)
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
                        use_local_output=True,
                    ),
                    "transformer.h.*.post_attention_layernorm": SequenceParallel(),
                    "transformer.h.*.post_feedforward_layernorm": SequenceParallel(),
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
