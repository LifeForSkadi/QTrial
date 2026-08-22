"""qiskit-free 谱序启发式布局（自主实现：Fiedler 序 + 设备哈密顿蛇形路径）。"""
from __future__ import annotations

import numpy as np

from qtrail.devices.spec import DeviceSpec
from qtrail.problems import ProgramGraph


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

    path = _device_snake(spec)

    pi = np.zeros(n, dtype=np.int64)
    for k, logical in enumerate(logical_order):
        pi[logical] = path[k % len(path)]
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
