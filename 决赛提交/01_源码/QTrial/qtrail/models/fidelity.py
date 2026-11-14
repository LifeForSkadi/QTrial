"""QuEst 风格的保真度预测器（arXiv:2210.16724）—— 零 qiskit 纯 PyTorch。

把「电路 → 门图 → 图 Transformer → 保真度标量」这条链路搬进交付管线，
替代/补充 ``qtrail/pure/metrics.py`` 的朴素乘积模型（该模型只用 ε_1q/ε_2q/
ε_ro，**完全忽略 T1/T2**；QuEst 预测器用 T1/T2 特征学到解相干效应）。

图构建（``build_gate_graph``）：
  - 节点 = 线路所有 op + measure；边 = ``circ.deps()`` 的依赖 DAG。
  - 节点特征（d_in=15，适配平台基，替代 QuEst 的 24 维）：
      门类型 one-hot（7: rz/sx/x/cz/swap/measure/other）
      + 第一/第二比特归一化索引（各 1，1Q 的第二为 0）
      + T1/T2（第一比特）归一化 + T1/T2（第二比特）归一化
      + 门误差（2Q 用 err_2q、1Q 用 err_1q、measure 用 0）
      + 读出误差（measure 用 err_ro、否则 0）
  - T1/T2 按设备中位数归一化，比特索引按 N 归一化。

预测器（``FidelityPredictor``）：in_proj → GraphTransformerEncoder（全局
注意力）→ 节点均值池化 → 3 层 FC 回归 → sigmoid 压到 [0,1]。

训练数据生成（PST 标签）见 ``scripts/train_fidelity.py``（实验侧，用
qiskit AerSimulator + 含 T1/T2 解相干的噪声模型）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qtrail.models.graph_transformer import GraphTransformerEncoder
from qtrail.models.layers import init_linear

# 门类型 one-hot 顺序（平台基 + measure + 兜底）
GATE_TYPES = ["rz", "sx", "x", "cz", "swap", "measure", "other"]
_GATE_IDX = {g: i for i, g in enumerate(GATE_TYPES)}
D_IN = len(GATE_TYPES) + 1 + 1 + 2 + 2 + 1 + 1  # = 15


def _clamp_norm(v: float, ref: float, hi: float = 5.0) -> float:
    if ref <= 0:
        return 0.0
    return float(np.clip(v / ref, 0.0, hi))


def build_gate_graph(circ, calib):
    """构建门图：返回 (x[M, D_IN] float32, adj[M, M] float32, mask[M] bool)。"""
    n = len(calib.t1) if calib is not None and len(calib.t1) else 1
    t1 = np.asarray(calib.t1, dtype=np.float64) if calib is not None else np.ones(n)
    t2 = np.asarray(calib.t2, dtype=np.float64) if calib is not None else np.ones(n)
    err_1q = np.asarray(calib.err_1q, dtype=np.float64) if calib is not None else np.full(n, 1e-3)
    err_ro = np.asarray(calib.err_ro, dtype=np.float64) if calib is not None else np.full(n, 8.2e-3)
    err_2q = getattr(calib, "err_2q", {}) if calib is not None else {}

    t1_med = float(np.median(t1)) if len(t1) else 1.0
    t2_med = float(np.median(t2)) if len(t2) else 1.0
    median2q = float(np.median(list(err_2q.values()))) if err_2q else 1e-3

    ops = circ.ops
    measures = circ.measures
    M = len(ops) + len(measures)
    x = np.zeros((M, D_IN), dtype=np.float32)
    adj = np.zeros((M, M), dtype=np.float32)

    last_touch = {}
    for i, inst in enumerate(ops):
        qs = inst.qubits
        name = inst.name
        gt = _GATE_IDX.get(name, _GATE_IDX["other"])
        q1 = int(qs[0])
        q2 = int(qs[1]) if inst.nq == 2 else -1
        x[i, gt] = 1.0
        x[i, len(GATE_TYPES)] = q1 / max(n, 1)
        x[i, len(GATE_TYPES) + 1] = (q2 / max(n, 1)) if q2 >= 0 else 0.0
        x[i, len(GATE_TYPES) + 2] = _clamp_norm(t1[q1], t1_med)
        x[i, len(GATE_TYPES) + 3] = _clamp_norm(t2[q1], t2_med)
        if q2 >= 0:
            x[i, len(GATE_TYPES) + 4] = _clamp_norm(t1[q2], t1_med)
            x[i, len(GATE_TYPES) + 5] = _clamp_norm(t2[q2], t2_med)
            e = err_2q.get((min(q1, q2), max(q1, q2)),
                           err_2q.get((max(q1, q2), min(q1, q2)), median2q))
            x[i, len(GATE_TYPES) + 6] = float(e)
        else:
            x[i, len(GATE_TYPES) + 6] = float(err_1q[q1])
        # 读出误差列（+7）对非 measure 门保持 0
        for q in qs:
            last_touch[q] = i

    # 依赖 DAG 边（无向化，供 transformer 备用 / 未来 dist_bias）
    for i, deps in enumerate(circ.deps()):
        for j in deps:
            adj[i, j] = adj[j, i] = 1.0

    # measure 节点：接在最后一个触碰该比特的 op 之后
    for k, inst in enumerate(measures):
        idx = len(ops) + k
        q1 = int(inst.qubits[0])
        x[idx, _GATE_IDX["measure"]] = 1.0
        x[idx, len(GATE_TYPES)] = q1 / max(n, 1)
        x[idx, len(GATE_TYPES) + 2] = _clamp_norm(t1[q1], t1_med)
        x[idx, len(GATE_TYPES) + 3] = _clamp_norm(t2[q1], t2_med)
        x[idx, len(GATE_TYPES) + 7] = float(err_ro[q1])
        if q1 in last_touch:
            adj[idx, last_touch[q1]] = adj[last_touch[q1], idx] = 1.0

    mask = np.ones(M, dtype=bool)
    return x, adj, mask


class FidelityPredictor(nn.Module):
    """图 Transformer 保真度预测器：门图 → 保真度标量。"""

    def __init__(self, d_in: int = D_IN, d: int = 64, layers: int = 2,
                 heads: int = 4, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.d = d
        self.n_layers = layers
        self.heads = heads
        self.ff = ff
        self.in_proj = nn.Linear(d_in, d)
        self.encoder = GraphTransformerEncoder(
            d=d, layers=layers, heads=heads, ff=ff, dropout=dropout,
            dist_bias=False, bias_from_adj=False)
        self.regressor = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.apply(init_linear)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """x [M, d_in] / adj [M, M] / mask [M] -> 标量 logit。"""
        if x.shape[0] == 0:
            # 空线路（无门无测量）：保真度 1 -> 返回大正 logit（sigmoid≈1）
            return torch.tensor(10.0, device=x.device, dtype=x.dtype)
        h = self.in_proj(x)                       # [M, d]
        h = h.unsqueeze(0)                        # [1, M, d]
        a = adj.unsqueeze(0)                      # [1, M, M]
        m = mask.unsqueeze(0)                     # [1, M]
        h = self.encoder(h, a, m)                 # [1, M, d]
        pooled = h.squeeze(0).mean(dim=0)         # [d]
        return self.regressor(pooled).squeeze(0)  # scalar []

    @torch.no_grad()
    def predict(self, circ, calib) -> float:
        x, adj, mask = build_gate_graph(circ, calib)
        x = torch.from_numpy(x)
        adj = torch.from_numpy(adj)
        mask = torch.from_numpy(mask)
        out = self.forward(x, adj, mask)
        return float(torch.sigmoid(out).item())

    def save_checkpoint(self, path, *, epoch: int = 0, val_loss: float = 0.0,
                        optimizer=None, extra: dict | None = None) -> None:
        ckpt = {
            "model": self.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "cfg": {"d_in": self.d_in, "d": self.d, "layers": self.n_layers,
                    "heads": self.heads, "ff": self.ff},
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt.get("cfg", {})
        model = cls(d_in=cfg.get("d_in", D_IN), d=cfg.get("d", 64),
                    layers=cfg.get("layers", 2), heads=cfg.get("heads", 4),
                    ff=cfg.get("ff", 128))
        model.load_state_dict(ckpt["model"])
        if isinstance(map_location, (str, torch.device)):
            model = model.to(map_location)
        return model, ckpt
