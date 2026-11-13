# -*- coding: utf-8 -*-
"""A/B：稀疏奇环结构布局先验（2026-11-13）。

对照：`QTRAIL_SPARSE_CYCLE_LAYOUT=0`（关闭候选）vs 默认开启。
机制：路由前对最大度 ≤2 且非二分的逻辑交互图做"删同色边 → 路径
嵌入"，把该嵌入作为初始布局交给 SABRE；唯一长程门由路由链化。

预注册判据（先于结果写定）：
  P1  graphstate_105：开启臂 2Q ≤ 0.5 × 关闭臂，且 logf 提升 > 1.0
      （自然对数）；否则视为无收益。
  P2  回归：graphstate_100 / ghz_n78 两臂的 2Q 门数与 2Q 深度相同
      （允许 |Δ| ≤ 1 的种子噪声）；若变差则拒绝该改动。
  P3  墙钟：graphstate_105 开启臂 ≤ 1.5 × 关闭臂。
  P4  合法性：开启臂产物 0 违规（门集/相邻/禁用/越界/swap/measure）。

输出：tables/ab_sparse_cycle/rows.jsonl + summary.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt")
CASES = [
    ("graphstate_105", "data/mqtbench/stratified/graphstate_105.qasm"),
    ("graphstate_100", "data/mqtbench/stratified/graphstate_100.qasm"),
    ("ghz_n78", "data/benchpress/qasm/qasmbench-large/ghz_n78/ghz_n78.qasm"),
]
OUT = Path("tables/ab_sparse_cycle")


def legality(final_qc, spec):
    probs = []
    disabled = set(spec.disabled_qubits)
    for idx, inst in enumerate(final_qc.ops):
        qs = [int(q) for q in inst.qubits]
        if inst.name == "swap":
            probs.append(f"swap@{idx}")
        if any(q >= spec.n for q in qs):
            probs.append(f"越界{qs}@{idx}")
        if any(q in disabled for q in qs):
            probs.append(f"禁用{qs}@{idx}")
        if inst.name == "cz" and not spec.adj[qs[0], qs[1]]:
            probs.append(f"cz非相邻{qs}@{idx}")
    names = {i.name for i in final_qc.ops}
    if not names <= {"rz", "sx", "x", "cz", "measure"}:
        probs.append(f"门集{names}")
    return probs


def run(spec, policy, cfg, name, text, enabled):
    os.environ["QTRAIL_SPARSE_CYCLE_LAYOUT"] = "1" if enabled else "0"
    torch.manual_seed(42)
    t0 = time.time()
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cpu", seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    res = mapper.map_circuit(parse_qasm(text), circuit_id=name)
    m = res["metrics"]
    return {
        "circuit": name, "arm": "on" if enabled else "off",
        "twoq": m["twoq_count"], "twoq_depth": m["twoq_depth"],
        "depth": m["depth"], "swaps": res["swap_count"],
        "logf": m["log_est_fidelity"], "fid": m["est_fidelity"],
        "wall": round(time.time() - t0, 2),
        "violations": legality(res["routed_circuit"], spec),
    }


def main():
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in CASES:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        off = run(spec, policy, cfg, name, text, False)
        on = run(spec, policy, cfg, name, text, True)
        rows += [off, on]
        print(f"{name}: off 2Q={off['twoq']} logf={off['logf']:.3f} "
              f"wall={off['wall']} | on 2Q={on['twoq']} "
              f"logf={on['logf']:.3f} wall={on['wall']}", flush=True)
    (OUT / "rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=float) + "\n"
                for r in rows), encoding="utf-8")

    by = {(r["circuit"], r["arm"]): r for r in rows}
    g5_off, g5_on = by[("graphstate_105", "off")], by[("graphstate_105", "on")]
    p1 = (g5_on["twoq"] <= 0.5 * g5_off["twoq"]
          and g5_on["logf"] > g5_off["logf"] + 1.0)
    p2 = all(abs(by[(c, "on")]["twoq"] - by[(c, "off")]["twoq"]) <= 1
             and abs(by[(c, "on")]["twoq_depth"]
                     - by[(c, "off")]["twoq_depth"]) <= 1
             for c in ("graphstate_100", "ghz_n78"))
    p3 = g5_on["wall"] <= 1.5 * g5_off["wall"]
    p4 = all(not r["violations"] for r in rows if r["arm"] == "on")
    lines = ["# 稀疏奇环布局先验 A/B（2026-11-13）\n\n",
             "| 线路 | 臂 | 2Q | 2Q 深度 | 深度 | swap | logf | 墙钟 |\n",
             "|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        lines.append(
            f"| {r['circuit']} | {r['arm']} | {r['twoq']} | "
            f"{r['twoq_depth']} | {r['depth']} | {r['swaps']} | "
            f"{r['logf']:.3f} | {r['wall']}s |\n")
    lines.append(
        f"\n判据：P1（graphstate_105 2Q 减半且 logf 提升 >1）"
        f"{'✓' if p1 else '✗'}；P2（稀疏二分回归不变）"
        f"{'✓' if p2 else '✗'}；P3（墙钟 ≤1.5×）{'✓' if p3 else '✗'}；"
        f"P4（0 违规）{'✓' if p4 else '✗'}。\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines[-2:]), flush=True)


if __name__ == "__main__":
    main()
