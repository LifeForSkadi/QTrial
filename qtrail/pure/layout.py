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
