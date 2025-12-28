"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import logging
import math
from dataclasses import (
    dataclass,
    field,
)

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.nn import functional as F

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.model import register_model
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model.blocks.attention import CausalSelfAttention
from optimus_dl.modules.model.blocks.layer_norms import (
    LayerNorm,
    RMSNorm,
)

logger = logging.getLogger(__name__)


@dataclass
class GPTConfig(RegistryConfigStrict):
    """Configuration for GPT-style language models."""

    block_size: int = field(
        default=1024,
        metadata={
            "description": "Maximum context length. Determines max pos embeddings"
        },
    )
    vocab_size: int = field(default=50304, metadata={"description": "Vocabulary size"})
    n_layer: int = field(
        default=12, metadata={"description": "Number of transformer blocks"}
    )
    n_head: int = field(
        default=12, metadata={"description": "Number of attention heads"}
    )
    n_embd: int = field(
        default=768, metadata={"description": "Embedding dimensionality"}
    )
    dropout: float = field(default=0.0, metadata={"description": "Dropout probability"})
    bias: bool = field(
        default=True,
        metadata={"description": "Whether to use bias in linear layers and norms"},
    )
    tie_word_embeddings: bool = field(
        default=True,
        metadata={"description": "Share weights between token embeddings and LM head"},
    )
    shard_every_ith_layer: int = field(
        default=1,
        metadata={
            "description": "Control FSDP sharding granularity. Shard every i-th layer, 1 means all layers are sharded (if global reshard_after_forward is True)"
        },
    )


class MLP(nn.Module):
    """Standard GPT-2 MLP with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """A single Transformer block with self-attention and MLP."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


@register_model("gpt2", GPTConfig)
class GPT(BaseModel):
    """GPT Language Model architecture.

    Implements a decoder-only transformer with causal self-attention, absolute
    position embeddings, and standard GPT-2 layer ordering (LayerNorm before
    attention/MLP).

    Args:
        config: GPT model configuration.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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

    def _init_weights(self, module):
        """Standard Gaussian initialization for weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, **kwargs):
        """Compute model output for the given input tokens."""
        idx = input_ids

        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return {"logits": logits}

    def make_parameter_groups(self):
        """Divide parameters into decayed and non-decayed groups.

        Excludes biases and 1D parameters (normalization weights, embeddings)
        from weight decay. Handles weight tying correctly.

        Returns:
            List of dictionaries for PyTorch optimizer.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)

        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # Note: because named_modules and named_parameters are recursive,
                # we will see the same tensors multiple times. We use the parent module
                # to determine the weight decay strategy.
                if pn.endswith("weight_clip_val"):
                    # quant params are not decayed
                    no_decay.add(fpn)
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")
        if self.lm_head.weight is not self.transformer.wte.weight:
            no_decay.add("lm_head.weight")

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        return [
            {"params": [(n, p) for n, p in self.named_parameters() if n in decay]},
            {
                "params": [(n, p) for n, p in self.named_parameters() if n in no_decay],
                "weight_decay": 0.0,
            },
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation of new tokens.

        Args:
            idx: Starting token sequence (LongTensor).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Optional top-k sampling threshold.

        Returns:
            LongTensor containing original and generated tokens.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            output = self(idx_cond)
            logits = output["logits"]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def fully_shard(self, **fsdp_kwargs):
        """Apply FSDP sharding to each transformer block."""
        for i, module in enumerate(self.transformer.h):
            reshard_after_forward = fsdp_kwargs.get("reshard_after_forward", False)
            if i % self.config.shard_every_ith_layer == 0:
                # Shard this layer
                reshard_after_forward &= True
            else:
                # Do not shard this layer
                reshard_after_forward &= False
            fully_shard(
                module,
                **(fsdp_kwargs | {"reshard_after_forward": reshard_after_forward}),
            )
