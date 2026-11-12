# -*- coding: utf-8 -*-
"""保真度模型 A/B：product（历史）vs time（时间感知）。

用法：python scripts/ab_fidelity_model.py [--production]
默认代理口径（无 RL 策略、routing_seeds=1）；--production 用策略 +
16 种子（较慢）。

预注册判据（2026-11-13，第 5 项；实现前写定）：
  C1 交付 2Q 门数不增（time ≤ product）；
  C2 1Q 门数 ≤ product ×1.02；depth ≤ product ×1.02；
  C3 时间模型自评：time 产物的 log_est_fidelity_time ≥ product 产物的
     （按同一 time 模型复算，容差 1e-9）——新模型必须改善它自己的目标；
  C4 全管线墙钟 ≤ product ×1.05。
不达标 → 模型保持 opt-in（默认 product），裁决进台账；达标 → 默认切
time，并在 pending_decisions 记录 587 数字需按新口径重跑。
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
from qtrail.pure.metrics import _time_log_fidelity
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


def run_one(path, model, production=False, prod=None):
    os.environ["QTRAIL_FIDELITY_MODEL"] = model
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
    lt, t_us = _time_log_fidelity(qc, spec.calib)
    return {
        "n1q": sum(1 for i in qc.ops if i.nq == 1),
        "n2q": sum(1 for i in qc.ops if i.nq == 2),
        "depth": res["metrics"]["depth"],
        "logf_prod": res["metrics"]["log_est_fidelity_product"],
        "logf_time": lt,
        "t_us": t_us,
        "wall": wall,
    }


def main():
    production = "--production" in sys.argv
    prod = _load_production() if production else None
    # 预热（一次性导入/JIT）
    run_one("examples/qft5.qasm", "product", production, prod)
    print(f"mode: {'production' if production else 'proxy'}")
    ok = True
    for name, path in CASES:
        a = run_one(path, "time", production, prod)
        b = run_one(path, "product", production, prod)
        checks = [
            ("2Q 不增", a["n2q"] <= b["n2q"]),
            ("1Q ≤+2%", a["n1q"] <= b["n1q"] * 1.02),
            ("depth ≤+2%", a["depth"] <= b["depth"] * 1.02),
            ("time 自评不降", a["logf_time"] >= b["logf_time"] - 1e-9),
            ("wall ≤+5%", a["wall"] <= b["wall"] * 1.05),
        ]
        print(f"{name:8} time  1Q={a['n1q']:5d} 2Q={a['n2q']:4d} "
              f"depth={a['depth']:5d} t={a['t_us']:7.2f}us "
              f"logf_time={a['logf_time']:9.3f} "
              f"logf_prod={a['logf_prod']:9.3f} wall={a['wall']:6.2f}")
        print(f"{'':8} prod  1Q={b['n1q']:5d} 2Q={b['n2q']:4d} "
              f"depth={b['depth']:5d} t={b['t_us']:7.2f}us "
              f"logf_time={b['logf_time']:9.3f} "
              f"logf_prod={b['logf_prod']:9.3f} wall={b['wall']:6.2f}")
        for label, passed in checks:
            if not passed:
                ok = False
                print(f"  [FAIL] {label}")
    print("\n== 裁决:", "达标（time 可作默认）" if ok else "不达标（保持 opt-in）", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
