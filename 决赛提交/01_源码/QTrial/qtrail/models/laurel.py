"""LAUREL (Learned Augmented Residual Layer) — arXiv:2411.07501.

把通用的残差连接 ``x_{i+1} = f(x_i) + x_i`` 推广为可学习形式，作为
encoder 残差的**原地替换**。三个变体（论文 §2）：

- ``rw``（LAUREL-RW）:  ``out = alpha * branch + beta * resid``，
  ``alpha, beta = softmax(logits)`` —— 可学习标量、softmax 归一防发散
  （论文 §2.1：α/β 不能无界增长，softmax 归一有帮助）。
- ``lr``（LAUREL-LR）:  ``out = branch + (AB + I) resid``，其中 ``A``
  为 [d, r]、``B`` 为 [r, d] 可学习低秩矩阵（r << d），新增参数 2rD。
  初始化近零使 ``AB ≈ 0``，因此初始化时 ≈ 原样残差。
- ``rw_lr``: 上述两者组合。

``none`` 为原样残差（``branch + resid``），默认值，保证旧 checkpoint
零改动加载、行为不变。

QTrail 用途：替换 ``qtrail/models/gat.py`` 的 ``GATLayer`` 与
``qtrail/models/graph_transformer.py`` 的 ``TransformerLayer`` 中的
普通残差加法。``forward(branch, resid)`` 输入均为 ``[B, n, d]``。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LaurelResidual(nn.Module):
    """Learned augmented residual (RW / LR / RW+LR), or identity (none)."""

    def __init__(self, d: int, mode: str = "none", rank: int = 4):
        super().__init__()
        if mode not in ("none", "rw", "lr", "rw_lr"):
            raise ValueError(f"unknown laurel mode: {mode}")
        self.mode = mode
        self.d = d
        self.rank = rank

        if mode in ("rw", "rw_lr"):
            # softmax over two logits -> alpha, beta convex weights
            self.logits = nn.Parameter(torch.zeros(2))
        if mode in ("lr", "rw_lr"):
            # low-rank augmentation W = A @ B + I; init near zero -> identity
            self.A = nn.Parameter(torch.empty(d, rank))
            self.B = nn.Parameter(torch.empty(rank, d))
            nn.init.normal_(self.A, mean=0.0, std=0.01)
            nn.init.normal_(self.B, mean=0.0, std=0.01)

    def forward(self, branch: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return branch + resid
        if self.mode == "rw":
            alpha, beta = torch.softmax(self.logits, dim=0)
            return alpha * branch + beta * resid
        if self.mode == "lr":
            # resid @ (A@B)^T = resid @ B^T @ A^T  (avoid materializing d*d)
            aug = resid @ self.B.t() @ self.A.t()
            return branch + resid + aug
        # rw_lr
        alpha, beta = torch.softmax(self.logits, dim=0)
        aug = resid @ self.B.t() @ self.A.t()
        return alpha * branch + beta * resid + aug
