"""GAT encoder (CO-MAP reproduction): multi-head graph attention layers.

Paper: 4 GATConv layers, 8 heads, d=128, GraphNorm (BatchNorm/GraphNorm best).
Batched via padded adjacency; attention restricted to graph neighbors + self.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qtrail.models.layers import make_norm


class GATLayer(nn.Module):
    """Multi-head graph attention layer (additive attention, heads concat)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 8,
                 dropout: float = 0.1, norm: str = "graph",
                 edge_weight_bias: bool = False):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = dropout
        self.edge_weight_bias = edge_weight_bias
        head_dim = max(out_dim // heads, 1)
        self.head_dim = head_dim
        self.lin = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.att_l = nn.Parameter(torch.empty(heads, head_dim))
        self.att_r = nn.Parameter(torch.empty(heads, head_dim))
        self.out_lin = nn.Linear(heads * head_dim, out_dim)
        self.norm = make_norm(norm, out_dim)
        self.residual = in_dim == out_dim
        if not self.residual:
            self.res_lin = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.out_lin.weight)
        nn.init.zeros_(self.out_lin.bias)
        if not self.residual:
            nn.init.xavier_uniform_(self.res_lin.weight)
            nn.init.zeros_(self.res_lin.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, n, _ = x.shape
        H, dh = self.heads, self.head_dim

        Wh = self.lin(x).view(B, n, H, dh)              # [B, n, H, dh]
        e = (Wh * self.att_l).sum(-1)                   # [B, n, H]
        logits = e.unsqueeze(2) + e.unsqueeze(1)        # [B, n, n, H]
        logits = F.leaky_relu(logits, 0.2)

        att_mask = ((adj > 0).float()
                    + torch.eye(n, device=x.device, dtype=x.dtype)
                    .unsqueeze(0))                      # [B, n, n]
        logits = logits.masked_fill(att_mask.unsqueeze(-1) <= 0, -1e9)

        alpha = F.softmax(logits, dim=2)                # [B, n, n, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.einsum("bijh,bjhf->bihf", alpha, Wh)  # [B, n, H, dh]
        out = out.reshape(B, n, H * dh)
        out = self.out_lin(out)
        res = x if self.residual else self.res_lin(x)
        out = self.norm(out + res, mask) if mask is not None else out + res
        return F.relu(out)


class GATEncoder(nn.Module):
    """Stacked GAT layers producing node embeddings [B, n, d]."""

    def __init__(self, d: int = 128, layers: int = 4, heads: int = 8,
                 dropout: float = 0.1, norm: str = "graph",
                 edge_weight_bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(d, d, heads=heads, dropout=dropout, norm=norm,
                     edge_weight_bias=edge_weight_bias)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj, mask)
        return x
