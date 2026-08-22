"""Graph Transformer encoder (QTrail improvement).

Global self-attention over all nodes (no adjacency restriction) captures
long-range dependencies between logically distant but strongly coupled qubits.
Optional relative-distance bias injects grid geometry for the device graph.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qtrail.models.layers import init_linear


class TransformerLayer(nn.Module):
    """Pre-LN transformer block with optional attention bias."""

    def __init__(self, d: int, heads: int = 8, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None = None,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        att, _ = self.mha(h, h, h, key_padding_mask=key_pad_mask,
                          attn_mask=bias, need_weights=False)
        x = x + att
        x = x + self.ffn(self.norm2(x))
        return x


class GraphTransformerEncoder(nn.Module):
    """Global-attention encoder; same forward(x, adj, mask) signature as GAT.

    Args:
        dist_bias: if True, add -D/τ (or -log1p(adj)/τ for weighted graphs) as
            an attention bias — injects graph geometry into global attention.
        tau: temperature of the distance bias.
    """

    def __init__(self, d: int = 128, layers: int = 4, heads: int = 8,
                 ff: int = 512, dropout: float = 0.1, dist_bias: bool = False,
                 tau: float = 8.0, bias_from_adj: bool = False):
        super().__init__()
        self.dist_bias = dist_bias
        self.bias_from_adj = bias_from_adj
        self.tau = tau
        self.layers = nn.ModuleList([
            TransformerLayer(d, heads=heads, ff=ff, dropout=dropout)
            for _ in range(layers)
        ])
        self.apply(init_linear)

    def _bias(self, adj: torch.Tensor) -> torch.Tensor | None:
        if not self.dist_bias:
            return None
        B, n, _ = adj.shape
        if self.bias_from_adj:
            # interaction-weighted graph: bias by -log1p(w)/tau
            b = -torch.log1p(adj) / self.tau
        else:
            b = -adj / self.tau
        # broadcast to all heads: [B*heads, n, n]
        return b.unsqueeze(1).expand(B, self.heads, n, n).reshape(B * self.heads, n, n)

    @property
    def heads(self) -> int:
        return self.layers[0].mha.num_heads

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        key_pad = None if mask is None else ~mask   # [B, n] True = padded
        bias = self._bias(adj) if self.dist_bias else None
        for layer in self.layers:
            x = layer(x, key_pad_mask=key_pad, bias=bias)
        return x
