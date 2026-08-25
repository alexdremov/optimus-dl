"""KV-cache primitives for incremental (cached) generation.

A ``KVCacheLayer`` holds pre-allocated key/value tensors for a single
attention layer plus a write cursor. The generation engine allocates one
cache per layer, runs a prefill forward (which fills the cache for the whole
prompt), and then feeds tokens one at a time (decode steps) instead of
re-running the full sequence at every step.

Cache layout matches what ``scaled_dot_product_attention`` expects:
``(batch, n_kv_head, max_len, head_dim)``.

Rows are *right-packed*: row ``i`` owns slots ``[0, len_i)``; slots beyond a
row's current length contain garbage that is excluded from attention via a
key-padding mask derived from the per-row lengths.
"""

from dataclasses import dataclass

import torch


@dataclass
class KVCacheLayer:
    """Per-layer key/value cache with a scalar write cursor.

    Attributes:
        k: Key cache ``(B, n_kv_head, max_len, head_dim)``.
        v: Value cache ``(B, n_kv_head, max_len, head_dim)``.
        cursor: Next write position (number of tokens cached so far).
    """

    k: torch.Tensor
    v: torch.Tensor
    cursor: int = 0

    def reset(self) -> None:
        """Reset the write cursor (buffers are reused as-is; stale data is
        never read because attention is always masked by valid lengths)."""
        self.cursor = 0

    @property
    def max_len(self) -> int:
        return self.k.shape[2]
