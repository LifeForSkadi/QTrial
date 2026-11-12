# -*- coding: utf-8 -*-
"""LNN 链合成（A1 组件，2026-11-10）。

长程 CNOT 的近邻链分解：4(d-1) 门最优构造（自研推导，BFS 验证），
链表固化于 kms_chain_table.json（生成脚本 tests 内验证：d=1..25 门数
精确、GF(2) 矩阵精确）。

噪声感知链选路（qiskit 的链分解是噪声无关的——本模块的差异化）：
多条最短路径按逐边校准误差选最优。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_TABLE_PATH = Path(__file__).parent / "kms_chain_table.json"
_CHAINS: dict = {}


def _load():
    global _CHAINS
    if not _CHAINS:
        _CHAINS = {int(k): [tuple(g) for g in v]
                   for k, v in json.loads(
                       _TABLE_PATH.read_text(encoding="utf-8")).items()}
    return _CHAINS


def chain_for_distance(d: int) -> list:
    """距离 d 的 CNOT 链（LNN 序 0..d 上的相邻 cx 序列）。"""
    chains = _load()
    if d not in chains:
        raise ValueError(f"链表中无距离 {d}（支持 1..{max(chains)}）")
    return chains[d]


def cnot_chain_on_path(path: list, ctl_idx: int, tgt_idx: int) -> list:
    """路径 path（设备节点序）上 ctl_idx→tgt_idx 的 CNOT 相邻 cx 序列。

    路径段被归一化到 LNN 序（方向反转时链的控制/目标方向需翻转——
    CNOT 的链构造不对称）。返回 [(q0, q1), ...]（相邻设备节点对）。
    """
    d = abs(tgt_idx - ctl_idx)
    if d == 0:
        raise ValueError("同一节点")
    seg = path[min(ctl_idx, tgt_idx):max(ctl_idx, tgt_idx) + 1]
    if ctl_idx < tgt_idx:
        chain = chain_for_distance(d)
        return [(seg[c], seg[t]) for c, t in chain]
    # 反向：CNOT(seg[d], seg[0]) = swap 共轭镜像（每门 (c,t) → (d-c,d-t)，
    # 全链 seg 反转映射；矩阵验证由测试锁定）
    chain = chain_for_distance(d)
    return [(seg[d - c], seg[d - t]) for c, t in chain]


_GRAPH_CACHE: dict = {}


def _device_graph(spec):
    """设备图（按 spec 对象缓存——链化热循环避免重复构图）。"""
    import networkx as nx
    g = _GRAPH_CACHE.get(id(spec))
    if g is None:
        g = nx.Graph()
        g.add_nodes_from(range(spec.n))
        g.add_edges_from([(i, j) for i in range(spec.n)
                          for j in range(i + 1, spec.n) if spec.adj[i, j]])
        _GRAPH_CACHE[id(spec)] = g
    return g


def _shortest_paths(spec, a, b, max_d=8, max_paths: int = 32):
    """设备图上 a→b 的最短路径采样（噪声感知选路只需少量候选）。

    复杂度缺陷修复（2026-11-12，graphstate_50 900s 超时+段错误的
    根因）：原实现对 d ≤ max_d 调用 nx.all_shortest_paths 枚举
    【全部】最短路径——密集设备图上两点间最短路径数随 d 指数级，
    152 门线路链化即内存爆掉（SIGSEGV）。改为 islice 封顶采样
    （确定性前缀，O(max_paths·d)）。
    """
    from itertools import islice
    import networkx as nx
    g = _device_graph(spec)
    if not nx.has_path(g, a, b):
        return []
    d = nx.shortest_path_length(g, a, b)
    if d > max_d:
        return [nx.shortest_path(g, a, b)]  # 长距离只取一条（BFS 路径）
    return list(islice(nx.all_shortest_paths(g, a, b), max_paths))


def noise_aware_chain(spec, a, b):
    """长程 CNOT(a,b) 的噪声感知链：多最短路径按 Σ 校准误差选优。

    返回 (gates, cost)，gates = [(q0, q1)] 相邻 cx 序列（q0 控制）；
    cost = 链门数（4(d-1)）与路径误差的加权和（选路用）。
    """
    err2q = spec.calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3

    def edge_err(x, y):
        if x > y:
            x, y = y, x
        return err2q.get((x, y), err2q.get((y, x), median2q))

    paths = _shortest_paths(spec, a, b)
    if not paths:
        return None, float("inf")
    best = None
    for path in paths:
        idx_a = path.index(a)
        idx_b = path.index(b)
        gates = cnot_chain_on_path(path, idx_a, idx_b)
        cost = sum(edge_err(x, y) for x, y in gates)
        if best is None or cost < best[0]:
            best = (cost, gates)
    return best[1], best[0]
