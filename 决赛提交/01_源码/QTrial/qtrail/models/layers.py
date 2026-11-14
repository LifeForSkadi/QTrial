"""Shared building blocks: GraphNorm, activation, init helpers (pure torch)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GraphNorm1d(nn.Module):
    """Instance-wise normalization over REAL nodes only (CO-MAP 'GraphNorm').

    mean/var are computed per instance over non-padded nodes; padded node
    activations are then also normalized with those statistics (they are
    masked downstream, so this is harmless and keeps the op differentiable).
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, n, d]; mask: [B, n] bool (True = real node)
        m = mask.unsqueeze(-1).to(x.dtype)            # [B, n, 1]
        cnt = m.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        mean = (x * m).sum(dim=1, keepdim=True) / cnt
        var = ((x - mean) ** 2 * m).sum(dim=1, keepdim=True) / cnt
        out = (x - mean) / torch.sqrt(var + self.eps)
        return out * self.weight + self.bias


class LayerNormMasked(nn.Module):
    """LayerNorm variant for ablation (CO-MAP found it worst; kept for parity)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d, eps=eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class BatchNormMasked(nn.Module):
    """BatchNorm over the node dim (CO-MAP parity variant)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.BatchNorm1d(d, eps=eps, affine=True, track_running_stats=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [B, n, d] -> [B*n, d] -> back (padded nodes included; fine for parity)
        B, n, d = x.shape
        return self.norm(x.reshape(B * n, d)).reshape(B, n, d)


def make_norm(kind: str, d: int) -> nn.Module:
    if kind == "graph":
        return GraphNorm1d(d)
    if kind == "layer":
        return LayerNormMasked(d)
    if kind == "batch":
        return BatchNormMasked(d)
    raise ValueError(f"unknown norm kind: {kind}")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)


def init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
