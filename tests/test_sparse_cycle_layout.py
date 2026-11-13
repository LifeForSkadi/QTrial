# -*- coding: utf-8 -*-
"""稀疏奇环布局先验测试（2026-11-13）。

预注册判据（实现前写定，见 scripts/ab_sparse_cycle_layout.py）：
  T1 11 节点奇环（设备二分）→ 返回布局：双射、非禁用、除被删的
     1 条边外全部映射后相邻。
  T2 11 节点路径（二分）→ 返回 None（不干预常规布局）。
  T3 最大度 >2 的图 → 返回 None（不干预稠密线路）。
  T4 小线路（n<10）→ 返回 None（触发门槛，避免 toffoli_n3 回归）。
"""
from __future__ import annotations

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.relabel_vf2 import sparse_cycle_layout


def _spec():
    return build_tianyan287_spec(load_device_config())


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
    spec = _spec()
    circ = _cycle(11)
    pi = sparse_cycle_layout(circ, spec, time_budget=3.0)
    assert pi is not None and len(pi) == 11
    assert len(set(int(x) for x in pi)) == 11
    assert all(int(x) not in set(spec.disabled_qubits) for x in pi)
    adj = 0
    for i in range(11):
        a, b = int(pi[i]), int(pi[(i + 1) % 11])
        if spec.adj[a, b]:
            adj += 1
    assert adj >= 10


def test_bipartite_path_returns_none():
    spec = _spec()
    assert sparse_cycle_layout(_path(11), spec, time_budget=2.0) is None


def test_dense_graph_returns_none():
    spec = _spec()
    c = Circuit(11)
    for i in range(11):
        for j in range(i + 1, 11):
            c.append(Inst("cx", (i, j)))
    assert sparse_cycle_layout(c, spec, time_budget=2.0) is None


def test_small_circuit_returns_none():
    spec = _spec()
    assert sparse_cycle_layout(_cycle(5), spec, time_budget=2.0) is None
