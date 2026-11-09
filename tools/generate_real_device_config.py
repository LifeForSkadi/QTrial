# -*- coding: utf-8 -*-
"""从平台 API 快照（data/tianyan_probe/tianyan-287_config.json）生成
仓库内固化的真机设备配置（configs/device_tianyan287_real_*.json）。

用途：真机拓扑/校准会随平台状态变化，本工具把某次 download_config 的
结果固化为可提交的配置文件（快照）；运行时可用 --calibration live
（需 login key）拉取最新配置覆盖。重新固化：先跑 scripts/probe_real_machines.py
刷新 data/tianyan_probe/ 快照，再跑本脚本。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "tianyan_probe" / "tianyan-287_config.json"
OUT = ROOT / "configs" / "device_tianyan287_real_topo.json"
OUT_CALIB = ROOT / "configs" / "device_tianyan287_real_calib.json"


def main():
    cfg = json.loads(SRC.read_text(encoding="utf-8"))
    ov = cfg.get("overview", {}) or {}
    cm = ov.get("coupler_map", {}) or {}

    edges = []
    for name, pair in cm.items():
        q0, q1 = int(str(pair[0]).lstrip("Q")), int(str(pair[1]).lstrip("Q"))
        edges.append((min(q0, q1), max(q0, q1), name))
    labels = sorted({q for e in edges for q in e[:2]})
    lab2idx = {q: i for i, q in enumerate(labels)}
    dis_q = sorted(int(x.lstrip("Q")) for x in
                   str(cfg.get("disabledQubits", "")).split(",") if x.strip())
    dis_c = sorted(str(x).strip() for x in
                   str(cfg.get("disabledCouplers", "")).split(",")
                   if x.strip())

    topo = {
        "machine": "tianyan-287",
        "snapshot_time": str(cfg.get("calibrationTime", "unknown")),
        "labels": labels,                       # Q1..Q105
        "edges_internal": [                       # 182 条设计边（内部索引）
            [lab2idx[a], lab2idx[b]] for a, b, _ in edges],
        "disabled_qubits": [lab2idx[q] for q in dis_q],
        "disabled_couplers": [                     # 14 条禁用边（内部索引）
            [lab2idx[int(str(cm[nm][0]).lstrip("Q"))],
             lab2idx[int(str(cm[nm][1]).lstrip("Q"))]] for nm in dis_c],
        "usable_edges_internal": [                 # 可用边（内部索引）
            [lab2idx[a], lab2idx[b]] for a, b, nm in edges
            if nm not in set(dis_c) and a not in set(dis_q)
            and b not in set(dis_q)],
        "source": "platform download_config (probe 2026-11-09)",
    }
    OUT.write_text(json.dumps(topo, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"拓扑 -> {OUT}（设计边 {len(topo['edges_internal'])}，"
          f"禁用耦合器 {len(topo['disabled_couplers'])}，"
          f"禁用比特 {topo['disabled_qubits']}，"
          f"可用边 {len(topo['usable_edges_internal'])}）")

    # ---- 校准快照（百分数转小数）
    import numpy as np
    t1 = np.array(cfg["qubit"]["relatime"]["T1"]["param_list"], dtype=np.float64)
    t2 = np.array(cfg["qubit"]["relatime"]["T2"]["param_list"], dtype=np.float64)
    e1 = np.array(cfg["qubit"]["singleQubit"]["gate error"]["param_list"],
                  dtype=np.float64) / 100.0
    ero = np.array(cfg["readout"]["readoutArray"]["Readout Error"]["param_list"],
                   dtype=np.float64) / 100.0
    e2r = np.array(cfg["twoQubitGate"]["czGate"]["gate error"]["param_list"],
                   dtype=np.float64) / 100.0
    usable_q = [q for q in labels if q not in set(dis_q)]
    usable_e = [e for e in edges if e[2] not in set(dis_c)]
    assert len(t1) == len(usable_q) and len(e2r) == len(usable_e)
    calib = {
        "machine": "tianyan-287",
        "snapshot_time": str(cfg.get("calibrationTime", "unknown")),
        "usable_qubit_labels": usable_q,
        "t1_us": {str(q): float(t1[i]) for i, q in enumerate(usable_q)},
        "t2_us": {str(q): float(t2[i]) for i, q in enumerate(usable_q)},
        "err_1q": {str(q): float(e1[i]) for i, q in enumerate(usable_q)},
        "err_ro": {str(q): float(ero[i]) for i, q in enumerate(usable_q)},
        "err_2q": {f"{min(a, b)}-{max(a, b)}": float(e2r[i])
                   for i, (a, b, _) in enumerate(usable_e)},
        "note": "误差为小数（平台原始为百分数）；T1/T2 单位 us；"
                "禁用比特/禁用耦合器无校准数据（构建时用可用集中位兜底）",
        "source": "platform download_config (probe 2026-11-09)",
    }
    OUT_CALIB.write_text(json.dumps(calib, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"校准 -> {OUT_CALIB}")


if __name__ == "__main__":
    main()
