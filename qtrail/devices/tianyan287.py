"""Tianyan-287 device factory (Zuchongzhi-3.0-class 105-qubit 15x7 grid)."""
from __future__ import annotations

from pathlib import Path

from qtrail.config import DeviceConfig, NoiseConfig, load_device_config
from qtrail.devices.calibration import CalibrationData, generate_synthetic_calibration
from qtrail.devices.spec import DeviceSpec, build_grid_spec


def build_tianyan287_spec(cfg: DeviceConfig | None = None,
                          calib: CalibrationData | None = None) -> DeviceSpec:
    """Build the tianyan-287 DeviceSpec.

    Geometry comes from configs/device_tianyan287.yaml (single source of truth).
    Pass a CalibrationData to override the synthetic generator (e.g. the
    runtime adapter with live platform data).
    """
    cfg = cfg or load_device_config()
    n = cfg.rows * cfg.cols
    if calib is None:
        if cfg.calibration == "none":
            # degenerate calibration (all-median) so the stack still runs
            edges = _full_grid_edges(cfg.rows, cfg.cols, cfg.absent_couplers,
                                     cfg.disabled_qubits)
            calib = _median_calibration(n, edges)
        else:
            edges = _full_grid_edges(cfg.rows, cfg.cols, cfg.absent_couplers,
                                     cfg.disabled_qubits)
            calib = generate_synthetic_calibration(n, edges, seed=cfg.calibration_seed,
                                                   correlated_defects=cfg.correlated_defects)
    return build_grid_spec(name=cfg.name, rows=cfg.rows, cols=cfg.cols,
                           calib=calib, absent_couplers=cfg.absent_couplers,
                           disabled_qubits=cfg.disabled_qubits,
                           noise=cfg.noise)


def build_grid8x8_spec(seed: int = 0) -> DeviceSpec:
    """8x8 grid (CO-MAP paper's main evaluation device) for reproduction parity."""
    import numpy as np
    from qtrail.config import DeviceConfig
    cfg = DeviceConfig(name="grid-8x8", rows=8, cols=8, calibration_seed=seed,
                       correlated_defects=False)
    return build_tianyan287_spec(cfg)


def build_grid3x3_spec(seed: int = 0) -> DeviceSpec:
    """3x3 grid for unit/integration tests."""
    from qtrail.config import DeviceConfig
    cfg = DeviceConfig(name="grid-3x3", rows=3, cols=3, calibration_seed=seed,
                       correlated_defects=False)
    return build_tianyan287_spec(cfg)


def _full_grid_edges(rows: int, cols: int, absent: set, disabled: set) -> list:
    # absent/disabled 元素统一为 (min, max) 元组（调用方可能传 list of lists）
    absent = {(min(a), max(a)) for a in absent}
    disabled = set(disabled)
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i in disabled:
                continue
            for (dr, dc) in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = rr * cols + cc
                    if j in disabled or (min(i, j), max(i, j)) in absent:
                        continue
                    edges.append((i, j))
    return edges


def _median_calibration(n: int, edges: list) -> CalibrationData:
    import numpy as np
    return CalibrationData(
        t1=np.full(n, 72.0, dtype=np.float64),
        t2=np.full(n, 50.0, dtype=np.float64),
        err_1q=np.full(n, 1e-3, dtype=np.float64),
        err_ro=np.full(n, 8.2e-3, dtype=np.float64),
        err_2q={(q0, q1): 3.8e-3 for (q0, q1) in edges},
    )


# --------------------------------------------------------------------------
# 真机拓扑（选项 C：双配置并存，2026-11-09）
# --------------------------------------------------------------------------

REAL_TOPO_FILE = "device_tianyan287_real_topo.json"
REAL_CALIB_FILE = "device_tianyan287_real_calib.json"


def _load_real_topo() -> dict:
    import json
    from qtrail.config import CONFIGS_DIR
    p = CONFIGS_DIR / REAL_TOPO_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"真机拓扑配置缺失: {p}（用 tools/generate_real_device_config.py "
            "从平台 API 快照生成）")
    return json.loads(p.read_text(encoding="utf-8"))


def _snapshot_calibration(topo: dict, n: int) -> CalibrationData:
    """从仓库内校准快照构建 CalibrationData（禁用比特/边用可用集中位兜底）。"""
    import json
    import numpy as np
    from qtrail.config import CONFIGS_DIR
    p = CONFIGS_DIR / REAL_CALIB_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"真机校准快照缺失: {p}（用 tools/generate_real_device_config.py 生成）")
    c = json.loads(p.read_text(encoding="utf-8"))
    usable_q = [int(q) for q in c["usable_qubit_labels"]]
    med = lambda arr: float(np.median(arr))

    def per_qubit(field):
        # 快照键 = 平台标签（Q1..Q105 = 1..105）；内部索引 q ∈ [0, n)
        # 对应标签 q+1
        d = {int(q): float(v) for q, v in c[field].items()}
        vals = [d[q] for q in usable_q]
        return np.array([d.get(q + 1, med(vals)) for q in range(n)],
                        dtype=np.float64)

    err_2q = {}
    for key, v in c["err_2q"].items():
        a, b = (int(x) for x in key.split("-"))
        # 快照键 = 平台标签对；转内部索引（-1）
        err_2q[(a - 1, b - 1)] = float(v)
    return CalibrationData(t1=per_qubit("t1_us"), t2=per_qubit("t2_us"),
                           err_1q=per_qubit("err_1q"),
                           err_ro=per_qubit("err_ro"), err_2q=err_2q)


def _real_coords(edges: list, n: int) -> np.ndarray:
    """真机拓扑坐标：谱嵌入（保证邻接节点坐标相近；真机为竖边+周期对角
    边的弯曲排布，无线性网格坐标精确解）。"""
    import networkx as nx
    import numpy as np
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(a, b) for a, b in edges])
    np.random.seed(42)
    pos = nx.spectral_layout(g, scale=10)
    return np.array([[pos[i][0], pos[i][1]] for i in range(n)],
                    dtype=np.float32)


def build_tianyan287_real_spec(cfg: DeviceConfig | None = None,
                               calib: CalibrationData | None = None,
                               live: bool = False,
                               token: str | None = None) -> DeviceSpec:
    """真机拓扑 DeviceSpec（2026-11-09 平台 API 核验固化，选项 C）。

    拓扑：182 设计边（竖边 + "正正反反"周期对角边的弯曲排布）−
    14 禁用耦合器 = 168 可用边；4 个禁用比特（内部 21/26/34/89）；
    比特标签 Q1..Q105（qubit_labels）。

    校准优先级：显式 calib 参数 > live（平台 API，需 login key）>
    仓库内快照（configs/device_tianyan287_real_calib.json，
    2026-11-09 快照）> 合成兜底。

    与评测口径（build_tianyan287_spec = 188 边矩形网格 + 合成校准）并存：
    评测口径保持报告书数字可复现，真机口径供部署与真机对照实验使用。
    """
    import networkx as nx
    import numpy as np
    from qtrail.config import NoiseConfig
    from qtrail.devices.spec import DeviceSpec, _featurize

    cfg = cfg or DeviceConfig(name="tianyan-287-real")
    topo = _load_real_topo()
    n = len(topo["labels"])
    usable = {(a, b) for a, b in topo["usable_edges_internal"]}
    disabled = sorted(topo["disabled_qubits"])

    if calib is None:
        if live:
            try:
                from qtrail.devices.adapter import download_tianyan287_spec
                live_spec = download_tianyan287_spec(
                    "tianyan-287", token=token)
                if live_spec is not None:
                    # live 拓扑覆盖（以平台最新为准）
                    topo = {
                        "labels": sorted(live_spec.qubit_labels.tolist()),
                        "usable_edges_internal":
                            [(i, j) for i in range(live_spec.n)
                             for j in range(i + 1, live_spec.n)
                             if live_spec.adj[i, j]],
                        "disabled_qubits": list(live_spec.disabled_qubits),
                    }
                    n = live_spec.n
                    usable = set(topo["usable_edges_internal"])
                    disabled = topo["disabled_qubits"]
                    calib = live_spec.calib
            except Exception as e:  # noqa: BLE001
                print(f"[warn] live 校准拉取失败，回退快照: {e}")
        if calib is None:
            try:
                calib = _snapshot_calibration(topo, n)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 校准快照加载失败，合成兜底: {e}")
                from qtrail.devices.calibration import \
                    generate_synthetic_calibration
                calib = generate_synthetic_calibration(
                    n, sorted(usable), seed=0, correlated_defects=True)

    coords = _real_coords(sorted(usable), n)
    adj = np.zeros((n, n), dtype=np.int8)
    for a, b in usable:
        adj[a, b] = adj[b, a] = 1
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(usable)
    dist = nx.floyd_warshall_numpy(g).astype(np.float32)
    dist[np.isinf(dist)] = float(n * 10)

    noise = cfg.noise or NoiseConfig()
    median_err = float(np.median(list(calib.err_2q.values()))) \
        if calib.err_2q else 1e-3
    gw = nx.Graph()
    gw.add_nodes_from(range(n))
    for (q0, q1), e in calib.err_2q.items():
        if adj[q0, q1]:
            phi = float(np.clip(e / max(median_err, 1e-12),
                                noise.err2q_clip[0], noise.err2q_clip[1]))
            t1_pen = float(np.clip(
                1.0 - min(calib.t1[q0], calib.t1[q1]) /
                max(calib.t1[q0], calib.t1[q1], 1e-9),
                0.0, noise.t1_penalty_clip))
            w = 1.0 + noise.alpha * (phi - 1.0) + noise.beta * t1_pen
            gw.add_edge(q0, q1, weight=max(w, 1e-6))
    noise_dist = (nx.floyd_warshall_numpy(gw, weight="weight")
                  .astype(np.float32) if gw.number_of_edges() else dist.copy())
    noise_dist[np.isinf(noise_dist)] = float(n * 10)

    return DeviceSpec(name=cfg.name or "tianyan-287-real", n=n, adj=adj,
                      coords=coords, dist=dist, noise_dist=noise_dist,
                      node_features=_featurize(n, coords, adj, calib),
                      calib=calib, absent_couplers=[],
                      qubit_labels=np.arange(1, n + 1, dtype=np.int64),
                      disabled_qubits=disabled)
