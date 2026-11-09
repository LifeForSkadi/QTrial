"""Device specification: coupling graph, distance matrices, node features."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from qtrail.config import NoiseConfig
from qtrail.devices.calibration import CalibrationData


@dataclass
class DeviceSpec:
    """Static description of a quantum device (pure numpy/networkx)."""
    name: str
    n: int                                  # number of physical qubits
    adj: np.ndarray                         # [N, N] int8 symmetric coupling graph
    coords: np.ndarray                      # [N, 2] float (row, col) for grids
    dist: np.ndarray                        # [N, N] float32 unweighted shortest path
    noise_dist: np.ndarray                  # [N, N] float32 noise-weighted shortest path
    node_features: np.ndarray               # [N, 7] float32 normalized
    calib: CalibrationData
    absent_couplers: list = field(default_factory=list)   # [(q0, q1), ...]
    qubit_labels: np.ndarray | None = None  # [N] original platform labels
    disabled_qubits: list = field(default_factory=list)   # internal indices

    # -------------------------------------------------------------- properties
    @property
    def edges(self) -> list:
        return [(i, j) for i in range(self.n) for j in range(i + 1, self.n) if self.adj[i, j]]

    @property
    def usable_positions(self) -> np.ndarray:
        """非禁用物理比特索引（布局候选/启发式/LS 重分配只应使用这些位置）。

        禁用比特在硬件中出现 = 拓扑结构变化（特种拓扑），所有映射环节
        必须统一规避——RL 解码经 d_mask 屏蔽，启发式/平凡候选/LS 经本
        属性过滤（2026-11-09 缺口 1 修复）。
        """
        bad = set(self.disabled_qubits)
        return np.array([q for q in range(self.n) if q not in bad],
                        dtype=np.int64)

    def distance_matrix(self, noise_lambda: float = 0.0) -> np.ndarray:
        """Effective distance: D_topo + lambda_n * (D_noise - D_topo)."""
        if noise_lambda == 0.0:
            return self.dist
        return self.dist + noise_lambda * (self.noise_dist - self.dist)

    def node_features_matrix(self) -> np.ndarray:
        return self.node_features

    def err_2q_matrix(self) -> np.ndarray:
        """[N, N] 2Q gate error matrix (inf where no edge)."""
        m = np.full((self.n, self.n), np.inf, dtype=np.float32)
        for (q0, q1), e in self.calib.err_2q.items():
            m[q0, q1] = m[q1, q0] = e
        return m


def usable_spec(spec: "DeviceSpec") -> "DeviceSpec":
    """剔除禁用比特与孤立比特后按可用子集重索引的规格。

    qiskit 路由要求线路比特数与耦合图一致，且禁用比特不可出现在布局
    中；live 配置（如 tianyan176 66 注册 65 可用、其中还有孤立比特）
    需先经此压缩——孤立比特（无耦合边）对映射无意义且会破坏
    CouplingMap 的节点一致性。
    """
    bad = set(spec.disabled_qubits)
    deg = spec.adj.sum(axis=1)
    keep = [q for q in range(spec.n) if q not in bad and deg[q] > 0]
    if len(keep) == spec.n:
        return spec
    idx = {q: i for i, q in enumerate(keep)}
    calib = spec.calib
    return DeviceSpec(
        name=spec.name, n=len(keep),
        adj=spec.adj[np.ix_(keep, keep)].copy(),
        coords=spec.coords[keep].copy(),
        dist=spec.dist[np.ix_(keep, keep)].copy(),
        noise_dist=spec.noise_dist[np.ix_(keep, keep)].copy(),
        node_features=spec.node_features[keep].copy(),
        calib=CalibrationData(
            t1=calib.t1[keep].copy(), t2=calib.t2[keep].copy(),
            err_1q=calib.err_1q[keep].copy(),
            err_ro=calib.err_ro[keep].copy(),
            err_2q={(idx[a], idx[b]): e for (a, b), e in calib.err_2q.items()
                    if a in idx and b in idx}),
        absent_couplers=[(idx[a], idx[b]) for a, b in spec.absent_couplers
                         if a in idx and b in idx],
        qubit_labels=(spec.qubit_labels[keep].copy()
                      if spec.qubit_labels is not None else None),
        disabled_qubits=[])


def build_grid_spec(name: str, rows: int, cols: int,
                    calib: CalibrationData,
                    absent_couplers: list | None = None,
                    disabled_qubits: list | None = None,
                    noise: NoiseConfig | None = None) -> DeviceSpec:
    """Build a DeviceSpec for a rows x cols 2D grid.

    Internal index = row * cols + col. Absent couplers / disabled qubits are
    removed from the coupling graph but kept in distance computation inputs
    (shortest paths route around them naturally).
    """
    absent = {(min(a), max(a)) for a in (absent_couplers or [])}
    disabled = set(disabled_qubits or [])
    n = rows * cols
    adj = np.zeros((n, n), dtype=np.int8)
    coords = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=np.float32)
    for i in range(n):
        if i in disabled:
            continue
        r, c = i // cols, i % cols
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                j = rr * cols + cc
                if j in disabled:
                    continue
                key = (min(i, j), max(i, j))
                if key in absent:
                    continue
                adj[i, j] = adj[j, i] = 1

    # unweighted all-pairs shortest path
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]])
    dist = nx.floyd_warshall_numpy(g).astype(np.float32)

    # noise-weighted distance via Dijkstra on noise edge weights
    noise = noise or NoiseConfig()
    median_err = float(np.median(list(calib.err_2q.values()))) if calib.err_2q else 1e-3
    gw = nx.Graph()
    gw.add_nodes_from(range(n))
    for (q0, q1) in calib.err_2q:
        if adj[q0, q1]:
            phi = float(np.clip(calib.err_2q[(q0, q1)] / max(median_err, 1e-12),
                                noise.err2q_clip[0], noise.err2q_clip[1]))
            t1_pen = float(np.clip(1.0 - min(calib.t1[q0], calib.t1[q1]) /
                                   max(calib.t1[q0], calib.t1[q1], 1e-9),
                                   0.0, noise.t1_penalty_clip))
            w = 1.0 + noise.alpha * (phi - 1.0) + noise.beta * t1_pen
            gw.add_edge(q0, q1, weight=max(w, 1e-6))
    if len(gw.edges) == 0:
        noise_dist = dist.copy()
    else:
        noise_dist = nx.floyd_warshall_numpy(gw, weight="weight").astype(np.float32)
    # unconnected pairs (disabled qubits) -> large finite value
    noise_dist[np.isinf(noise_dist)] = float(n * 10)
    noise_dist = np.minimum(noise_dist, np.float32(n * 10))

    node_features = _featurize(n, coords, adj, calib)

    return DeviceSpec(name=name, n=n, adj=adj, coords=coords,
                      dist=dist, noise_dist=noise_dist,
                      node_features=node_features, calib=calib,
                      absent_couplers=sorted(absent),
                      disabled_qubits=sorted(disabled))


def _featurize(n: int, coords: np.ndarray, adj: np.ndarray,
               calib: CalibrationData) -> np.ndarray:
    """7-dim device node features, per-device min-max normalized.

    [T1, T2, err_1q, err_ro, degree, row, col] -- row/col normalized by their
    own range; missing values (live configs) fall back to device medians.
    """
    def safe(arr):
        return np.asarray(arr, dtype=np.float32)

    deg = adj.sum(axis=1).astype(np.float32)
    rows = coords[:, 0]
    cols = coords[:, 1]

    def norm01(v):
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        return (v - lo) / (hi - lo + 1e-9) if hi > lo else np.zeros_like(v)

    feats = np.stack([
        norm01(safe(calib.t1)),
        norm01(safe(calib.t2)),
        norm01(safe(calib.err_1q)),
        norm01(safe(calib.err_ro)),
        norm01(deg),
        norm01(rows),
        norm01(cols),
    ], axis=1).astype(np.float32)
    return feats
