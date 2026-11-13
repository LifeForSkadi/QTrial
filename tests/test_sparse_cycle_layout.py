# -*- coding: utf-8 -*-
"""稀疏奇环布局先验测试（2026-11-13）。

预注册判据（实现前写定，见 scripts/ab_sparse_cycle_layout.py）：
  T1 5 节点奇环（3×3 网格二分）→ 返回布局：双射、非禁用、除被删的
     1 条边外全部映射后相邻。
  T2 4 节点路径（二分）→ 返回 None（不干预常规布局）。
  T3 最大度 >2 的图 → 返回 None（不干预稠密线路）。
  T4 与大布局的语义边界：返回 pi 长度 = circ.n，且不含禁用比特。
"""
from __future__ import annotations

import numpy as np

from qtrail.devices import build_grid3x3_spec
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.relabel_vf2 import sparse_cycle_layout


def _cycle(n):
    c = Circuit(n)
    for i in range(n):
        c.append(Inst("cx", (i, (i + 1) % n)))
    return c


def _path(n):
    c = Circuit(n)
    for i in range(n - 1):
        c.append(Inst("cx", (i, i + 1)))
    return c


def test_odd_cycle_returns_valid_layout():
    spec = build_grid3x3_spec(0)
    circ = _cycle(5)
    pi = sparse_cycle_layout(circ, spec, time_budget=3.0)
    assert pi is not None and len(pi) == 5
    assert len(set(int(x) for x in pi)) == 5
    assert all(int(x) not in set(spec.disabled_qubits) for x in pi)
    # 5 环删 1 条边 = 路径；至少 4 条边在设备上相邻
    adj = 0
    for i in range(5):
        a, b = int(pi[i]), int(pi[(i + 1) % 5])
        if spec.adj[a, b]:
            adj += 1
    assert adj >= 4


def test_bipartite_path_returns_none():
    spec = build_grid3x3_spec(0)
    assert sparse_cycle_layout(_path(4), spec, time_budget=2.0) is None


def test_dense_graph_returns_none():
    spec = build_grid3x3_spec(0)
    c = Circuit(4)
    for i in range(4):
        for j in range(i + 1, 4):
            c.append(Inst("cx", (i, j)))
    assert sparse_cycle_layout(c, spec, time_budget=2.0) is None
