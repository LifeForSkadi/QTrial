"""qiskit-free 谱序启发式布局（自主实现：Fiedler 序 + 设备哈密顿蛇形路径）。"""
from __future__ import annotations

import numpy as np

from qtrail.devices.spec import DeviceSpec
from qtrail.problems import ProgramGraph


def usable_positions(spec: DeviceSpec) -> np.ndarray:
    """可用物理比特索引（剔除 disabled_qubits，真机 live 配置安全）。"""
    bad = set(getattr(spec, "disabled_qubits", []) or [])
    return np.array([q for q in range(spec.n) if q not in bad],
                    dtype=np.int64)


def heuristic_layout(graph: ProgramGraph, spec: DeviceSpec,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Spectral heuristic: Fiedler-vector order of logical qubits mapped onto
    a device Hamiltonian path (grid snake). Never fails, works without a
    policy."""
    n = graph.n
    lap = np.diag(graph.adj.sum(axis=1)) - graph.adj
    evals, evecs = np.linalg.eigh(lap)
    fiedler = evecs[:, 1] if n > 1 else np.zeros(n)
    logical_order = np.argsort(fiedler)

    path = [p for p in _device_snake(spec) if p in set(usable_positions(spec).tolist())]

    pi = np.zeros(n, dtype=np.int64)
    for k, logical in enumerate(logical_order):
        pi[logical] = path[k % len(path)]
    return pi


def noise_greedy_layout(graph: ProgramGraph, spec: DeviceSpec,
                        dist_eff: np.ndarray,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """噪声贪心候选（自研启发式）：逻辑比特按交互度降序逐个放置，
    空位评分 = 已放置邻居的交互加权距离和 + 噪声罚项。

    目的：为候选池提供与 RL 静态代价互补的「噪声优先」解——RL 布局在
    密集全连通线路（如 QFT）上可能被拓扑项主导而忽视缺陷热区，本候选
    显式压低高误差耦合器的使用。
    """
    n = graph.n
    calib = spec.calib
    # 物理比特噪声评分：1Q 误差 + 邻边 2Q 误差均值
    err = np.zeros(spec.n)
    for q in range(spec.n):
        nb = [b for b in range(spec.n) if spec.adj[q, b]]
        e2 = (float(np.mean([calib.err_2q.get((min(q, b), max(q, b)),
                                              calib.err_2q.get(
                                                  (max(q, b), min(q, b)), 0.0))
                             for b in nb])) if nb else 0.0)
        err[q] = float(calib.err_1q[q]) + e2
    err_norm = err / (err.max() + 1e-12)
    # 逻辑顺序：交互度降序
    deg = graph.adj.sum(axis=1)
    logical_order = list(np.argsort(-deg))
    pi = np.full(n, -1, dtype=np.int64)
    used = set()
    gamma = 0.05 * (n / 20.0)  # 噪声罚系数（随规模温和增长）
    usable = set(usable_positions(spec).tolist())
    for logical in logical_order:
        best_q, best_c = -1, np.inf
        for q in usable:
            if q in used:
                continue
            c = gamma * err_norm[q] * deg[logical]
            for j in range(n):
                w = graph.adj[logical, j]
                if w > 0 and pi[j] >= 0:
                    c += w * dist_eff[q, pi[j]]
            if c < best_c:
                best_q, best_c = q, c
        pi[logical] = best_q
        used.add(best_q)
    return pi


def strong_noise_layout(graph: ProgramGraph, spec: DeviceSpec,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """强噪声候选（自研启发式）：纯噪声优先的占用——逻辑比特按交互度
    降序放到误差最低的物理比特上，完全不考虑距离（连通性交给路由）。

    动机：全连通密集线路（QFT/QV）会饱和设备，距离项无法避免缺陷
    热区；纯噪声序能实现低于中位误差的有效误差（实测 qft_N054 超越
    中位误差理想界），路由出的 SWAP 由折叠完全吸收。
    """
    n = graph.n
    calib = spec.calib
    err = np.zeros(spec.n)
    for q in range(spec.n):
        nb = [b for b in range(spec.n) if spec.adj[q, b]]
        e2 = (float(np.mean([calib.err_2q.get((min(q, b), max(q, b)),
                                              calib.err_2q.get(
                                                  (max(q, b), min(q, b)), 0.0))
                             for b in nb])) if nb else 0.0)
        err[q] = float(calib.err_1q[q]) + e2
    usable = set(usable_positions(spec).tolist())
    order = [q for q in np.argsort(err) if q in usable]
    logical_order = list(np.argsort(-graph.adj.sum(axis=1)))
    pi = np.full(n, -1, dtype=np.int64)
    used = set()
    for logical in logical_order:
        for q in order:
            if q not in used:
                pi[logical] = q
                used.add(q)
                break
    return pi


def _device_snake(spec: DeviceSpec) -> list[int]:
    """设备哈密顿路径：完整网格用行向蛇形；其余拓扑贪心最近邻 + BFS 逃逸。"""
    import networkx as nx

    coords = spec.coords
    if np.allclose(coords, coords.astype(np.int64)):
        rows = sorted({int(c[0]) for c in coords})
        ncols = len({int(c[1]) for c in coords})
        if len(rows) * ncols == spec.n:
            path = []
            for ri, r in enumerate(rows):
                row_q = sorted([i for i in range(spec.n)
                                if int(coords[i][0]) == r],
                               key=lambda i: int(coords[i][1]))
                path += row_q if ri % 2 == 0 else row_q[::-1]
            return path

    g = nx.Graph()
    g.add_nodes_from(range(spec.n))
    g.add_edges_from([(i, j) for i in range(spec.n)
                      for j in range(i + 1, spec.n) if spec.adj[i, j]])
    center = int(np.argmin(np.abs(spec.coords - spec.coords.mean(axis=0))
                           .sum(axis=1)))
    path = [center]
    visited = {center}
    while len(path) < spec.n:
        nbrs = [j for j in range(spec.n) if spec.adj[path[-1], j]
                and j not in visited]
        if not nbrs:
            unvisited = [v for v in range(spec.n) if v not in visited]
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
