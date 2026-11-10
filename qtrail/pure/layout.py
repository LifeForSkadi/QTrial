"""qiskit-free 谱序启发式布局（自主实现：Fiedler 序 + 设备哈密顿蛇形路径）。"""
from __future__ import annotations

import numpy as np

from qtrail.devices.spec import DeviceSpec
from qtrail.problems import ProgramGraph


def heuristic_layout(graph: ProgramGraph, spec: DeviceSpec,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Spectral heuristic: Fiedler-vector order of logical qubits mapped onto
    a device Hamiltonian path (grid snake). Never fails, works without a
    policy. 路径仅使用非禁用物理比特（2026-11-09 缺口 1 修复）。"""
    n = graph.n
    lap = np.diag(graph.adj.sum(axis=1)) - graph.adj
    evals, evecs = np.linalg.eigh(lap)
    fiedler = evecs[:, 1] if n > 1 else np.zeros(n)
    logical_order = np.argsort(fiedler)

    path = _device_snake(spec)

    pi = np.zeros(n, dtype=np.int64)
    for k, logical in enumerate(logical_order):
        pi[logical] = path[k % len(path)]
    return pi


def _device_snake(spec: DeviceSpec) -> list[int]:
    """设备哈密顿路径：完整网格用行向蛇形；其余拓扑贪心最近邻 + BFS 逃逸。
    禁用比特（spec.disabled_qubits）被排除在路径之外。"""
    import networkx as nx

    disabled = set(spec.disabled_qubits)
    usable = [q for q in range(spec.n) if q not in disabled]

    coords = spec.coords
    if np.allclose(coords, coords.astype(np.int64)):
        rows = sorted({int(c[0]) for c in coords})
        ncols = len({int(c[1]) for c in coords})
        if len(rows) * ncols == spec.n:
            path = []
            for ri, r in enumerate(rows):
                row_q = sorted([i for i in usable
                                if int(coords[i][0]) == r],
                               key=lambda i: int(coords[i][1]))
                path += row_q if ri % 2 == 0 else row_q[::-1]
            return path

    g = nx.Graph()
    g.add_nodes_from(usable)
    g.add_edges_from([(i, j) for i in usable for j in range(i + 1, spec.n)
                      if j in usable and spec.adj[i, j]])
    center = int(min(usable, key=lambda i: float(
        np.abs(spec.coords[i] - spec.coords[usable].mean(axis=0)).sum())))
    path = [center]
    visited = {center}
    while len(path) < len(usable):
        nbrs = [j for j in usable if spec.adj[path[-1], j]
                and j not in visited]
        if not nbrs:
            unvisited = [v for v in usable if v not in visited]
            if not unvisited:
                break
            target = min(unvisited, key=lambda v: spec.dist[path[-1], v])
            try:
                chain = nx.shortest_path(g, path[-1], target)
            except nx.NetworkXNoPath:
                path.append(target)
                visited.add(target)
                continue
            for v in chain[1:]:
                if v not in visited:
                    path.append(v)
                    visited.add(v)
            continue
        path.append(nbrs[0])
        visited.add(nbrs[0])
    return path


def greedy_graph_match_layout(circ, spec,
                              rng: np.random.Generator | None = None):
    """贪心图匹配布局（2026-11-13，qiskit DenseLayout/SabreLayout 初始化
    同款语义的自主实现）：电路 2Q 交互图按门数加权贪心嵌入设备耦合图。

    从最重交互边开始：两节点未放 → 占一条设备可用边；单节点已放 →
    放其设备邻接位置；得分 = 已放置邻居的交互权重 × [邻接指示]。
    只使用非禁用比特。返回 pi（pi[logical] = physical）。
    """
    n = circ.n
    weights = {}
    for inst in circ.ops:
        if inst.nq == 2 and inst.name != "swap":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            e = (a, b) if a < b else (b, a)
            weights[e] = weights.get(e, 0) + 1

    disabled = set(spec.disabled_qubits)
    usable = [q for q in range(spec.n) if q not in disabled]
    pi = np.full(n, -1, dtype=np.int64)
    used = set()

    def score_phys(p, v):
        """逻辑 v 放物理 p 的得分：已放置逻辑邻居的交互权重×邻接。"""
        s = 0.0
        for (a, b), w in weights.items():
            u = b if a == v else (a if b == v else None)
            if u is not None and pi[u] >= 0 and spec.adj[p, int(pi[u])]:
                s += w
        return s

    def best_phys(v):
        cand = [q for q in usable if q not in used]
        if not cand:
            return None
        return int(max(cand, key=lambda q: score_phys(q, v)))

    for (a, b), _w in sorted(weights.items(),
                             key=lambda x: -x[1]):
        if pi[a] >= 0 and pi[b] >= 0:
            continue
        if pi[a] < 0 and pi[b] < 0:
            # 找设备上一条可用边（优先与已放置部分邻接）
            best = None
            best_s = -1.0
            for p in usable:
                if p in used:
                    continue
                for q in usable:
                    if q in used or q <= p or not spec.adj[p, q]:
                        continue
                    s = score_phys(p, a) + score_phys(q, b)
                    if s > best_s:
                        best_s = s
                        best = (p, q)
            if best is None:
                best = (best_phys(a), best_phys(b))
                if best[0] == best[1]:
                    break
            pi[a], pi[b] = int(best[0]), int(best[1])
            used.update(best)
        else:
            v = a if pi[a] < 0 else b
            p = best_phys(v)
            if p is None:
                continue
            pi[v] = p
            used.add(p)

    for v in range(n):
        if pi[v] < 0:
            p = best_phys(v)
            if p is None:
                raise ValueError("设备可用比特不足")
            pi[v] = p
            used.add(p)
    return pi
