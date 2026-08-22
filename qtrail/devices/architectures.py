"""平台拓扑家族预设：Sycamore 式砖墙网格 / IBM heavy-hex / 参数化网格。

用途：未来混合拓扑训练的拓扑侧多样性来源 + 平台家族机器（天衍-294/504
拓扑未公开，运行时经 adapter.download_config 获取；此处的预设用于训练泛化）。
本模块允许 qiskit 依赖（仅用于导出 heavy-hex 边集），生成的 DeviceSpec
仍与纯 numpy 的 DeviceSpec 完全一致。
"""
from __future__ import annotations

import numpy as np

from qtrail.devices.calibration import CalibrationData, generate_synthetic_calibration
from qtrail.devices.spec import DeviceSpec, build_grid_spec
from qtrail.config import NoiseConfig


def _calib_for(n: int, edges: list, seed: int) -> CalibrationData:
    return generate_synthetic_calibration(n, edges, seed=seed,
                                          correlated_defects=True)


def build_sycamore53_spec(seed: int = 0) -> DeviceSpec:
    """Google Sycamore 式 53 比特：9 列×6 行网格去掉 (0,0) 角点，
    水平全连 + 垂直砖墙（(r+c) 偶数列连下行）。"""
    rows, cols = 6, 9
    removed = {(0, 0)}
    nodes = [(r, c) for r in range(rows) for c in range(cols)
             if (r, c) not in removed]
    index = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)
    edges = []
    for (r, c) in nodes:
        for (dr, dc) in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in index:
                continue
            if dc == 1 or (r + c) % 2 == 0:  # 水平全连；垂直砖墙
                edges.append((min(index[(r, c)], index[(nr, nc)]),
                              max(index[(r, c)], index[(nr, nc)])))
    edges = sorted(set(edges))
    calib = _calib_for(n, edges, seed)
    adj = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1
    coords = np.array([[r, c] for (r, c) in nodes], dtype=np.float32)
    return build_spec_from_edges(name="sycamore-53", n=n, edges=edges,
                                 coords=coords, calib=calib)


def build_heavyhex_spec(distance: int = 7, seed: int = 0,
                        name: str | None = None) -> DeviceSpec:
    """IBM 式 heavy-hex 拓扑（distance 7 → 115 比特，Eagle 家族代表）。"""
    from qiskit.transpiler import CouplingMap  # 仅此处允许 qiskit 依赖
    cm = CouplingMap.from_heavy_hex(distance)
    n = cm.size()
    edges = [(min(a, b), max(a, b)) for a, b in cm.get_edges() if a < b]
    edges = sorted(set(edges))
    calib = _calib_for(n, edges, seed)
    coords = np.zeros((n, 2), dtype=np.float32)
    return build_spec_from_edges(name=name or f"heavy-hex-{n}", n=n,
                                 edges=edges, coords=coords, calib=calib)


def build_grid_family_spec(rows: int, cols: int, seed: int = 0,
                           name: str | None = None) -> DeviceSpec:
    """参数化矩形网格（训练规模多样性：6×6 / 10×10 / 12×12 等）。"""
    from qtrail.config import DeviceConfig
    cfg = DeviceConfig(name=name or f"grid-{rows}x{cols}", rows=rows, cols=cols,
                       calibration_seed=seed, correlated_defects=True)
    from qtrail.devices import build_tianyan287_spec
    return build_tianyan287_spec(cfg)


def build_spec_from_edges(name: str, n: int, edges: list,
                          coords: np.ndarray, calib: CalibrationData,
                          noise: NoiseConfig | None = None) -> DeviceSpec:
    """从任意边集构建 DeviceSpec（非网格拓扑的通用路径）。"""
    import networkx as nx
    noise = noise or NoiseConfig()
    adj = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    dist = nx.floyd_warshall_numpy(g).astype(np.float32)

    median_err = float(np.median(list(calib.err_2q.values()))) if calib.err_2q else 1e-3
    gw = nx.Graph()
    gw.add_nodes_from(range(n))
    for (q0, q1), e in calib.err_2q.items():
        if adj[q0, q1]:
            phi = float(np.clip(e / max(median_err, 1e-12),
                                noise.err2q_clip[0], noise.err2q_clip[1]))
            t1_pen = float(np.clip(1.0 - min(calib.t1[q0], calib.t1[q1]) /
                                   max(calib.t1[q0], calib.t1[q1], 1e-9),
                                   0.0, noise.t1_penalty_clip))
            w = 1.0 + noise.alpha * (phi - 1.0) + noise.beta * t1_pen
            gw.add_edge(q0, q1, weight=max(w, 1e-6))
    noise_dist = (nx.floyd_warshall_numpy(gw, weight="weight").astype(np.float32)
                  if len(gw.edges) else dist.copy())
    noise_dist[np.isinf(noise_dist)] = float(n * 10)

    from qtrail.devices.spec import _featurize
    node_features = _featurize(n, coords, adj, calib)
    return DeviceSpec(name=name, n=n, adj=adj, coords=coords,
                      dist=dist, noise_dist=noise_dist,
                      node_features=node_features, calib=calib,
                      absent_couplers=[], qubit_labels=np.arange(n, dtype=np.int64))
