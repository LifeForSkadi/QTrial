# -*- coding: utf-8 -*-
"""A2 合成后端 A/B：自研 VW vs qiskit（执行顺序第 3 项预注册判据 C3/C4）。

用法：
  python scripts/ab_a2_backend.py            # 默认语料 heis16/qft20/N25
  python scripts/ab_a2_backend.py --qv32     # 加 qv32（较慢）

两条路径除 QTRAIL_A2_BACKEND 外完全相同（同布局/路由/后处理/决胜）。
判据（预注册）：vw 相对 qiskit：2Q 门数不增；1Q 增幅 ≤2%；depth 增幅
≤2%；log_est_fidelity 不劣于 -0.01；墙钟增幅 ≤5%。不达标即默认回退
qiskit 并记录负结果。
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
    ("heis16", "data/benchpress/qasm/square-heisenberg/"
               "square_heisenberg_N16.qasm"),
    ("qft20", "data/benchpress/qasm/qft/qft_N020.qasm"),
    ("N25", "data/benchpress/qasm/square-heisenberg/"
            "square_heisenberg_N25.qasm"),
]
CASES_QV = [("qv32", "data/benchpress/qasm/qasmbench-large/QV_n32/32.qasm")]


def _load_production():
    """生产配置：RL 策略 + routing_seeds=16 + seed=42（审计口径）。"""
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


def run_one(path, backend, production=False, prod=None):
    os.environ["QTRAIL_A2_BACKEND"] = backend
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
    n1q = sum(1 for i in qc.ops if i.nq == 1)
    n2q = sum(1 for i in qc.ops if i.nq == 2)
    met = res["metrics"]
    return {
        "n1q": n1q, "n2q": n2q, "depth": met["depth"],
        "logf": met["log_est_fidelity"], "wall": wall,
        "backend": __import__("qtrail.pure.synthesis",
                              fromlist=["A2_BACKEND"]).A2_BACKEND,
    }


def _warmup(production, prod):
    """预热一次性成本（numba JIT / torch 导入 / policy 首次前向）。

    墙钟 A/B 的顺序偏差修复（2026-11-13）：首个 run 会承担 ~1s 的
    一次性开销，若不预热，先跑的后端被人为加价。
    """
    run_one("examples/qft5.qasm", "vw", production, prod)
    run_one("examples/qft5.qasm", "qiskit", production, prod)


def main():
    production = "--production" in sys.argv
    cases = list(CASES)
    if "--qv32" in sys.argv:
        cases += CASES_QV
    prod = _load_production() if production else None
    _warmup(production, prod)
    print(f"mode: {'production (policy, seeds=16)' if production else 'proxy (no policy, seeds=1)'}")
    print(f"{'case':8} {'backend':8} {'1Q':>8} {'2Q':>6} {'depth':>7} "
          f"{'logf':>10} {'wall':>7}")
    ok = True
    for ci, (name, path) in enumerate(cases):
        # 交替顺序 + 每后端双次取最小墙钟（降低 CPU 调度噪声）
        order = ("vw", "qiskit") if ci % 2 == 0 else ("qiskit", "vw")
        res = {b: None for b in order}
        for _round in range(2):
            for backend in order:
                r = run_one(path, backend, production, prod)
                if res[backend] is None:
                    res[backend] = r
                else:
                    res[backend]["wall"] = min(res[backend]["wall"],
                                               r["wall"])
        a, b = res["vw"], res["qiskit"]
        print(f"{name:8} {'vw':8} {a['n1q']:8d} {a['n2q']:6d} "
              f"{a['depth']:7d} {a['logf']:10.4f} {a['wall']:7.2f}")
        print(f"{'':8} {'qiskit':8} {b['n1q']:8d} {b['n2q']:6d} "
              f"{b['depth']:7d} {b['logf']:10.4f} {b['wall']:7.2f}")
        checks = [
            ("2Q 不增", a["n2q"] <= b["n2q"]),
            ("1Q ≤+2%", a["n1q"] <= b["n1q"] * 1.02),
            ("depth ≤+2%", a["depth"] <= b["depth"] * 1.02),
            ("logf ≥-0.01", a["logf"] >= b["logf"] - 0.01),
            ("wall ≤+5%", a["wall"] <= b["wall"] * 1.05),
        ]
        for label, passed in checks:
            if not passed:
                ok = False
                print(f"  [FAIL] {label}: "
                      f"vw=({a['n1q']},{a['n2q']},{a['depth']},"
                      f"{a['logf']:.4f},{a['wall']:.2f}) "
                      f"qiskit=({b['n1q']},{b['n2q']},{b['depth']},"
                      f"{b['logf']:.4f},{b['wall']:.2f})")
    print("\n== 裁决:", "达标（vw 可作默认）" if ok else "不达标（回退）", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
