# -*- coding: utf-8 -*-
"""学习式路由评分器（RouteScorer，2026-11-10）。

设计定位：不学端到端路由策略，只学"候选交换边的评分函数"——
SABRE 骨架保留（贪心执行、候选边集合、10n 逃逸、平局裁决），评分
从手写线性组合换成 MLP。动机：qiskit decay 启发式的"最近使用惩罚"
是历史特征的非线性函数，手写参数无法复现，学习式评分可以自动发现
这类依赖；边校准误差入特征实现噪声感知（手写硬约束实测掉陷阱，
交给网络学平衡）。

训练：REINFORCE + 自批评基线（终局奖励 = -swap 数），与布局训练
同范式（qtrail/training/reinforce.py）。先以手写评分做回归暖启动，
再策略梯度微调。验收：诊断集 swap 低于前瞻基线（gf2^16 1371 /
gf2^32 6477 / dtc_100 4488），不达标则弃用（非劣化开关兜底）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

FEATURE_DIM = 9
# 特征索引（与 router._route_features 对应）：
# 0 basic/|F|   1 ext/|E|    2 h2/|F|     3 unlocked/|F|
# 4 边步龄     5 比特步龄   6 边误差      7 度归一     8 窗口使用次数


class RouteScorer(nn.Module):
    """候选边特征 (C, 9) → 评分 (C,)。"""

    def __init__(self, in_dim: int = FEATURE_DIM, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1))

    def forward(self, F: torch.Tensor,
                lin: torch.Tensor | None = None) -> torch.Tensor:
        """lin = 手写线性评分（路由器传入，暖启动回归用；推理忽略）。"""
        return self.net(F).squeeze(-1)

    # ------------------------------------------------------ 训练辅助
    # 约定：评分 = 代价（越小越好，与手写线性评分一致）。
    def sample_pick(self, scores: torch.Tensor, temp: float = 1.0):
        """代价的 Boltzmann 采样：p ∝ exp(-score/temp)。
        返回 (pick, logp, entropy)——logp/entropy 为保留计算图的
        0 维张量（REINFORCE 反向传播用）。"""
        logp = torch.log_softmax(-scores / temp, dim=0)
        probs = torch.exp(logp)
        pick = torch.multinomial(probs, 1)[0]
        ent = -(logp * probs).sum()
        return int(pick), logp[int(pick)], ent

    @staticmethod
    def greedy_pick(scores: torch.Tensor) -> int:
        return int(torch.argmin(scores))
