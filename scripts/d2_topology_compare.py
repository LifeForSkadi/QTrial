# -*- coding: utf-8 -*-
"""D2 对照实验：三种设备模型下 pure 管线输出对比 + 真机可执行性检查。

三种模型（同一批线路、同一 c0.05 权重、同一种子）：
  A 合成现状：完整 188 边网格 + 合成校准（= 587 条基准评测口径，config 默认）
  B 芯片设计：182 边（188 - 6 个无耦合器位置，从 coupler_map 反推）+ 合成校准
  C 真机当前：182 边 - 14 禁用耦合器 = 168 可用边 + 4 禁用比特（Q22/Q27/Q35/Q90）
     + 平台真实校准（T1/T2/1Q/读出/CZ 误差）
  C2 真机+噪声距离：同 C 但 noise_dist 按真实校准重算（adapter 的 noise_dist 是
     拓扑 fallback，C2 补上噪声距离以隔离"拓扑影响"与"噪声距离影响"）

指标：swap_count / depth / twoq_depth / est_fidelity（各自校准口径）
+ 统一口径保真度（全部用真机校准重算，隔离校准分布差异）
+ QCIS 真机可执行性：CZ 边 ∈ 可用耦合器、比特 ∉ 禁用集（对 C/C2 应为 0 违例；
  对 A/B 计数违例，量化"默认配置产物直接上真机"的问题）。
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import load_qasm_file
from qtrail.pure.mapper import PureMapper
from qtrail.models import QAPolicy

CKPT = ROOT / "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
PROBE = ROOT / "data" / "tianyan_probe" / "tianyan-287_config.json"

QUEKO_CASES = [
    "data/Queko/BIGD/20QBT_45CYC_.0D1_.1D2_0.qasm",
    "data/Queko/BIGD/20QBT_45CYC_.0D1_.5D2_0.qasm",
    "data/Queko/BIGD/20QBT_45CYC_.1D1_.7D2_0.qasm",
]
EXAMPLE_CASES = ["examples/qft5.qasm", "examples/qaoa12.qasm",
                 "examples/ghz10.qasm", "examples/random_deep16.qasm"]


def parse_real_config():
    """解析平台配置 → (labels, coupler_map{标签对}, disabled_q_labels,
    disabled_coupler_names, calib 数组)。

    逐比特校准在 qubit.relatime.T1/T2、qubit.singleQubit['gate error']、
    readout.readoutArray['Readout Error']（param_list 长度 = 可用比特数 101）；
    逐耦合器误差在 twoQubitGate.czGate['gate error']（长度 = 可用耦合器 168）。
    顺序假设 = 按标签升序剔除禁用项；用 overview 聚合均值校验顺序正确性。
    """
    import json
    cfg = json.loads(PROBE.read_text(encoding="utf-8"))
    ov = cfg.get("overview", {}) or {}
    cm = ov.get("coupler_map", {}) or {}
    edges = []
    for name, pair in cm.items():
        q0, q1 = int(str(pair[0]).lstrip("Q")), int(str(pair[1]).lstrip("Q"))
        edges.append((min(q0, q1), max(q0, q1), name))
    labels = sorted({q for e in edges for q in e[:2]})
    dis_q = set(int(x.lstrip("Q")) for x in
                str(cfg.get("disabledQubits", "")).split(",") if x.strip())
    dis_c = set(str(x).strip() for x in
                str(cfg.get("disabledCouplers", "")).split(",") if x.strip())

    usable_q = [q for q in labels if q not in dis_q]          # 101
    usable_edges = [e for e in edges if e[2] not in dis_c]    # 168

    t1 = np.array(cfg["qubit"]["relatime"]["T1"]["param_list"], dtype=np.float64)
    t2 = np.array(cfg["qubit"]["relatime"]["T2"]["param_list"], dtype=np.float64)
    # 平台校准误差字段为百分数（如 0.45 = 0.45%），转小数
    e1 = np.array(cfg["qubit"]["singleQubit"]["gate error"]["param_list"],
                  dtype=np.float64) / 100.0
    ero = np.array(cfg["readout"]["readoutArray"]["Readout Error"]["param_list"],
                   dtype=np.float64) / 100.0
    e2_raw = np.array(cfg["twoQubitGate"]["czGate"]["gate error"]["param_list"],
                      dtype=np.float64) / 100.0
    assert len(t1) == len(usable_q), f"T1 len {len(t1)} != {len(usable_q)}"
    assert len(e2_raw) == len(usable_edges), \
        f"CZ err len {len(e2_raw)} != {len(usable_edges)}"

    # 顺序校验（均值应等于 overview 聚合值）
    print(f"  校验: T1 均值 {t1.mean():.2f} vs overview {ov.get('T1')}; "
          f"CZ 均值 {e2_raw.mean():.2f} vs overview {ov.get('cz_error')}")

    t1_map = {q: t1[i] for i, q in enumerate(usable_q)}
    t2_map = {q: t2[i] for i, q in enumerate(usable_q)}
    e1_map = {q: e1[i] for i, q in enumerate(usable_q)}
    ero_map = {q: ero[i] for i, q in enumerate(usable_q)}
    e2 = {}
    for i, (a, b, nm) in enumerate(usable_edges):
        e2[(min(a, b), max(a, b))] = float(e2_raw[i])
    return labels, edges, sorted(dis_q), dis_c, t1_map, t2_map, e1_map, ero_map, e2


def main():
    from qtrail.devices.calibration import CalibrationData
    from qtrail.devices.spec import build_grid_spec

    labels, edges, dis_q_lab, dis_c_names, t1, t2, e1, ero, e2 = parse_real_config()
    lab2idx = {q: i for i, q in enumerate(labels)}
    dis_q_idx = [lab2idx[q] for q in dis_q_lab]
    print(f"真机配置: 标签 {labels[0]}..{labels[-1]}, 设计边 {len(edges)}, "
          f"禁用比特 {dis_q_lab} -> 内部 {dis_q_idx}, 禁用耦合器 {len(dis_c_names)} 个")
    print(f"校准统计: T1 mean {np.mean(list(t1.values())):.1f}us, "
          f"T2 mean {np.mean(list(t2.values())):.1f}us, "
          f"1Q err mean {np.mean(list(e1.values())):.2e}, "
          f"ro err mean {np.mean(list(ero.values())):.2e}, "
          f"CZ err n={len(e2)} mean {np.mean(list(e2.values())):.2e}")

    # ---- 真机校准（内部索引；4 个禁用比特无数据，用可用比特中位兜底）
    idx_edges = [(lab2idx[a], lab2idx[b]) for a, b, _ in edges]
    _med = lambda d: float(np.median(list(d.values())))
    calib_real = CalibrationData(
        t1=np.array([t1.get(q, _med(t1)) for q in labels]),
        t2=np.array([t2.get(q, _med(t2)) for q in labels]),
        err_1q=np.array([e1.get(q, _med(e1)) for q in labels]),
        err_ro=np.array([ero.get(q, _med(ero)) for q in labels]),
        err_2q={(ia, ib): float(e2.get((a, b), 3.8e-3))
                for (a, b, _), (ia, ib) in zip(edges, idx_edges)})

    # ---- 真机拓扑坐标：谱嵌入（相邻节点坐标相近；真机为竖边+正正反反
    # 周期对角边的弯曲排布，非标准矩形/剪斜网格，无线性坐标精确解）
    import networkx as nx
    _g = nx.Graph()
    _g.add_edges_from(idx_edges)
    np.random.seed(42)
    _pos = nx.spectral_layout(_g, scale=10)
    coords = np.array([[_pos[i][0], _pos[i][1]] for i in range(105)],
                      dtype=np.float32)
    print("真机拓扑坐标：谱嵌入（保证邻接节点坐标相近）")

    # ---- 手工构建真实拓扑 DeviceSpec（任意边集 + 剪斜坐标）
    import networkx as nx
    from qtrail.config import NoiseConfig
    from qtrail.devices.calibration import generate_synthetic_calibration
    from qtrail.devices.spec import DeviceSpec, _featurize

    def build_real_spec(name, edge_set, calib, disabled_qubits):
        adj = np.zeros((105, 105), dtype=np.int8)
        for a, b in edge_set:
            adj[a, b] = adj[b, a] = 1
        g = nx.Graph()
        g.add_nodes_from(range(105))
        g.add_edges_from(edge_set)
        dist = nx.floyd_warshall_numpy(g).astype(np.float32)
        dist[np.isinf(dist)] = 105 * 10
        noise = NoiseConfig()
        median_err = float(np.median(list(calib.err_2q.values()))) \
            if calib.err_2q else 1e-3
        gw = nx.Graph()
        gw.add_nodes_from(range(105))
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
        nd = (nx.floyd_warshall_numpy(gw, weight="weight").astype(np.float32)
              if gw.number_of_edges() else dist.copy())
        nd[np.isinf(nd)] = 105 * 10
        return DeviceSpec(name=name, n=105, adj=adj, coords=coords.copy(),
                          dist=dist, noise_dist=nd,
                          node_features=_featurize(105, coords, adj, calib),
                          calib=calib, absent_couplers=[],
                          qubit_labels=np.arange(1, 106, dtype=np.int64),
                          disabled_qubits=sorted(disabled_qubits))

    # 完整矩形网格 188 边（用于 absent_couplers 统计）
    full_grid_edge_set = set()
    for i in range(105):
        r, c = i // 7, i % 7
        for dr, dc in ((1, 0), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 15 and 0 <= cc < 7:
                j = rr * 7 + cc
                full_grid_edge_set.add((min(i, j), max(i, j)))
    design_set = set(idx_edges)                      # 182 条真实设计边
    disabled_edges = {(lab2idx[a], lab2idx[b]) for a, b, nm in edges
                      if nm in dis_c_names}          # 14 条当前禁用
    dis_q_set = set(dis_q_idx)
    # 真机可用边 = 设计边 - 禁用耦合器 - 端点含禁用比特的边
    usable_set = {(a, b) for a, b in design_set
                  if (a, b) not in disabled_edges
                  and a not in dis_q_set and b not in dis_q_set}
    print(f"矩形网格边 {len(full_grid_edge_set)} / 设计边 {len(design_set)} "
          f"（交集 {len(full_grid_edge_set & design_set)}）/ 当前禁用耦合器 "
          f"{len(disabled_edges)} / 真机可用边 {len(usable_set)}")

    # ---- 四种设备模型
    specA = build_tianyan287_spec(load_device_config())   # 现状：188 矩形网格+合成
    calib_design = generate_synthetic_calibration(
        105, list(design_set), seed=load_device_config().calibration_seed,
        correlated_defects=True)
    specB = build_real_spec("tianyan-287-design-182", design_set,
                            calib_design, [])             # 芯片设计 182 剪斜
    specC = build_real_spec("tianyan-287-real", usable_set,
                            calib_real, dis_q_idx)        # 真机当前可用边
    specC2 = build_real_spec("tianyan-287-real-nd", usable_set,
                             calib_real, dis_q_idx)       # 同 C（noise_dist 已重算）

    for s in (specA, specB, specC, specC2):
        print(f"spec {s.name}: n={s.n} 边数={int(s.adj.sum()/2)} "
              f"禁用比特={s.disabled_qubits}")

    # ---- 加载权重（device_n=105，四种 spec 同规模，零重训直接迁移）
    import torch
    dev = "cpu"
    policy, ck = QAPolicy.load_checkpoint(str(CKPT), device_n=105,
                                          map_location=dev)
    policy.eval()
    from qtrail.config import Config
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    cases = EXAMPLE_CASES + QUEKO_CASES
    results = {}
    for spec in (specA, specB, specC, specC2):
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                            selection_rule="fidelity")
        for case in cases:
            circ = load_qasm_file(ROOT / case)
            t0 = time.time()
            res = mapper.map_circuit(circ, circuit_id=Path(case).stem)
            wall = time.time() - t0
            m = res["metrics"]
            # 统一口径保真度（真机校准）
            from qtrail.pure.metrics import estimate_fidelity
            fid_real_calib = estimate_fidelity(res["routed_circuit"],
                                               calib_real)
            # QCIS 可执行性（对真机当前拓扑）
            violations = []
            for inst in res["routed_circuit"].ops:
                qs = inst.qubits
                for q in qs:
                    if int(q) in dis_q_idx:
                        violations.append(("q", int(q)))
                if inst.name == "cz":
                    a, b = int(qs[0]), int(qs[1])
                    if a > b:
                        a, b = b, a
                    if (a, b) not in usable_set:
                        violations.append(("edge", a, b))
            results[(spec.name, case)] = dict(
                swaps=m["swap_count"], depth=m["depth"],
                twoq_depth=m["twoq_depth"], fid=m["est_fidelity"],
                fid_real=fid_real_calib, wall=round(wall, 1),
                n_viol=len(violations))
            print(f"[{spec.name}] {Path(case).name}: swap={m['swap_count']} "
                  f"depth={m['depth']} 2qd={m['twoq_depth']} "
                  f"fid={m['est_fidelity']:.3e} fid@real={fid_real_calib:.3e} "
                  f"{wall:.1f}s 真机违例={len(violations)}")

    # ---- 汇总
    print("\n===== 汇总（均值）=====")
    print(f"{'case':<38}{'模型':<28}{'swap':>7}{'depth':>9}{'2qd':>7}"
          f"{'fid':>11}{'fid@real':>11}{'违例':>5}")
    for case in cases:
        for sname in ("tianyan-287", "tianyan-287", "tianyan-287",
                      "tianyan-287-real-noisedist"):
            pass
    for case in cases:
        for spec in (specA, specB, specC, specC2):
            r = results[(spec.name, case)]
            print(f"{Path(case).name:<38}{spec.name:<28}{r['swaps']:>7}"
                  f"{r['depth']:>9}{r['twoq_depth']:>7}{r['fid']:>11.3e}"
                  f"{r['fid_real']:>11.3e}{r['n_viol']:>5}")


if __name__ == "__main__":
    main()
