"""transformer --- Transformer-based NQS architectures."""

from __future__ import annotations

from qvartools.nqs.transformer.attention import CausalSelfAttention, CrossAttention
from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer

__all__ = [
    "AutoregressiveTransformer",
    "CausalSelfAttention",
    "CrossAttention",
]
