# -*- coding: utf-8 -*-
"""VF2++ 嵌入 A/B：自研 vf2pp（默认）vs 旧 networkx 路径。

用法：python scripts/ab_vf2pp.py [--production] [--large]

预注册判据（2026-11-13 第 6 项，见 vf2_postlayout_explained.md §8.4）：
  C1 正确性/合法性：产物零非法（2Q 相邻、无禁用比特、无 swap 残留）；
  C2 管线效果（语料 A/B）：2Q 不增、depth ≤+2%、logf 不劣于 -0.01；
  C3 墙钟：全管线增幅 ≤10%（large 组代表性样本）；
  C4 覆盖率：>20 活跃可嵌入图成功率 ≥50%（单元能力测试
     tests/test_vf2pp.py；真实 large 样本另报，含不可嵌入图）。
不达标 → 回退 networkx 路径（QTRAIL_VF2_IMPL=networkx 保持可切）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import load_qasm_file

CASES = [
    ("qft5", "examples/qft5.qasm"),
    ("ghz10", "examples/ghz10.qasm"),
    ("qaoa12", "examples/qaoa12.qasm"),
    ("heis16", "data/benchpress/qasm/square-heisenberg/"
               "square_heisenberg_N16.qasm"),
    ("qft20", "data/benchpress/qasm/qft/qft_N020.qasm"),
    ("N25", "data/benchpress/qasm/square-heisenberg/"
            "square_heisenberg_N25.qasm"),
]
LARGE = [
    ("heis_N100", "data/benchpress/qasm/square-heisenberg/"
                  "square_heisenberg_N100.qasm"),
    ("graphstate_50", "data/mqtbench/stratified/graphstate_50.qasm"),
    ("qv32", "data/benchpress/qasm/qasmbench-large/QV_n32/32.qasm"),
]


def run_one(path, impl, production=False, prod=None):
    os.environ["QTRAIL_VF2_IMPL"] = impl
    if production:
        import torch
        spec, policy, cfg, dev = prod
        torch.manual_seed(42)
        m = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                       selection_rule="fidelity", use_post=True,
                       pipeline="v5", routing_seeds=16,
                       linear_decompose=True)
    else:
        spec = build_tianyan287_spec(load_device_config())
        m = PureMapper(spec, policy=None, pipeline="v5",
                       selection_rule="fidelity", linear_decompose=True,
                       routing_seeds=1, seed=0)
    circ = load_qasm_file(path)
    t0 = time.time()
    res = m.map_circuit(circ, path)
    wall = time.time() - t0
    qc = res["routed_circuit"]
    met = res["metrics"]
    bad = 0
    for inst in qc.ops:
        if inst.nq == 2:
            if not spec.adj[int(inst.qubits[0]), int(inst.qubits[1])]:
                bad += 1
        for q in inst.qubits:
            if int(q) in set(spec.disabled_qubits):
                bad += 1
    bad += sum(1 for i in qc.ops if i.name == "swap")
    return {"n1q": sum(1 for i in qc.ops if i.nq == 1),
            "n2q": sum(1 for i in qc.ops if i.nq == 2),
            "depth": met["depth"], "logf": met["log_est_fidelity"],
            "wall": wall, "bad": bad}


def _load_production():
    import torch
    from qtrail.config import Config
    from qtrail.models import QAPolicy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    ckpt = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_" \
           "t0.5_c0.05_best.pt"
    policy, ck = QAPolicy.load_checkpoint(ckpt, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    return spec, policy, cfg, dev


def main():
    production = "--production" in sys.argv
    large = "--large" in sys.argv
    cases = list(CASES) + (LARGE if large else [])
    prod = _load_production() if production else None
    # 预热
    run_one("examples/qft5.qasm", "vf2pp", production, prod)
    print(f"mode: {'production' if production else 'proxy'}"
          f"{' + large' if large else ''}")
    ok = True
    for name, path in cases:
        a = run_one(path, "vf2pp", production, prod)
        b = run_one(path, "networkx", production, prod)
        print(f"{name:12} vf2pp 1Q={a['n1q']:6d} 2Q={a['n2q']:4d} "
              f"depth={a['depth']:5d} logf={a['logf']:9.3f} "
              f"wall={a['wall']:7.2f} bad={a['bad']}")
        print(f"{'':12} nx    1Q={b['n1q']:6d} 2Q={b['n2q']:4d} "
              f"depth={b['depth']:5d} logf={b['logf']:9.3f} "
              f"wall={b['wall']:7.2f} bad={b['bad']}")
        checks = [
            ("零非法", a["bad"] == 0),
            ("2Q 不增", a["n2q"] <= b["n2q"]),
            ("depth ≤+2%", a["depth"] <= b["depth"] * 1.02),
            ("logf ≥-0.01", a["logf"] >= b["logf"] - 0.01),
            # 墙钟判据针对 large 组（判据 C3）；小案例 <0.5s 的
            # 调度噪声用绝对 100ms 容差（60ms 基线上的 10ms 差
            # 不是机制回归）
            ("wall ≤+10%", a["wall"] <= b["wall"] * 1.10 + 0.10),
        ]
        for label, passed in checks:
            if not passed:
                ok = False
                print(f"  [FAIL] {label}")
    print("\n== 裁决:", "达标（vf2pp 默认）" if ok else "不达标（回退 networkx）", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
