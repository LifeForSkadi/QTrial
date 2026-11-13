# -*- coding: utf-8 -*-
"""真机对比分析（v8 vs Qiskit O3）：经典保真度 / TVD / 预测对照。

用法：python scripts/real_machine_v9_analyze.py qft5 qft6 ghz10
输出：tables/real_machine_v9/summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path("tables/real_machine_v9")


def load_counts(name: str, arm: str):
    p = OUT / f"{name}_{arm}_counts.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d


def ideal_probs(name: str, n: int) -> np.ndarray:
    """逻辑顺序理想分布：index = Σ b_q 2^q（qiskit 小端约定）。"""
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    from qtrail.utils.qasm_io import sanitize_qasm
    src = Path("examples") / f"{name}.qasm"
    if not src.exists():
        src = Path("out/real_machine") / f"{name}.qasm"
    if not src.exists() and name.startswith("qft") and name[3:].isdigit():
        src = Path("data/benchpress/qasm/qft") / f"qft_N{int(name[3:]):03d}.qasm"
    qc = qasm2.loads(sanitize_qasm(src.read_text(encoding="utf-8")))
    qc.remove_final_measurements(inplace=True)
    sv = Statevector.from_instruction(qc)
    return np.abs(sv.data) ** 2


def metrics(counts_d, ideal):
    counts = counts_d["counts"]
    total = counts_d["total"]
    n = len(next(iter(counts))) if counts else 0
    p = np.zeros(2 ** n)
    for bits, c in counts.items():
        idx = sum((1 if ch == "1" else 0) << q for q, ch in enumerate(bits))
        if idx < len(p):
            p[idx] += c
    p /= max(total, 1)
    q = np.zeros_like(p)
    m = min(len(p), len(ideal))
    q[:m] = ideal[:m]
    q /= q.sum()
    fid = float((np.sqrt(p * q).sum()) ** 2)
    tvd = float(0.5 * np.abs(p - q).sum())
    # 峰态电路（GHZ/x）另报"理想支撑命中率"：support ≤4 时才有意义
    support = np.where(q > 1e-6)[0]
    hit = float(p[support].sum()) if len(support) <= 4 else None
    return total, fid, tvd, hit


def main():
    names = sys.argv[1:] or ["qft5", "qft6", "ghz10"]
    rows = []
    lines = ["# 天衍真机对比：v8 vs Qiskit O3（tianyan176）\n\n",
             "同一线路、同一实时拓扑、同 shots；v8 = 自研管线"
             "（pre_cancel 开），O3 = Qiskit optimization_level=3。\n\n",
             "| 线路 | 臂 | 2Q | 预测 logf | 预测保真度 | 实测经典保真度 | TVD |\n",
             "|---|---|---|---|---|---|---|\n"]
    for name in names:
        n = None
        for arm in ("v8", "o3"):
            mp = OUT / f"{name}_{arm}_metrics.json"
            cp = OUT / f"{name}_{arm}_counts.json"
            if not mp.exists() or not cp.exists():
                continue
            m = json.loads(mp.read_text(encoding="utf-8"))
            n = m["n"]
            ideal = ideal_probs(name, n)
            total, fid, tvd, hit = metrics(
                json.loads(cp.read_text(encoding="utf-8")), ideal)
            pred = float(np.exp(m["logf"]))
            rows.append({"circuit": name, "arm": arm, "n": n,
                         "1q": m["1q"], "2q": m["2q"], "logf": m["logf"],
                         "pred_fid": pred, "meas_fid": fid, "tvd": tvd,
                         "hit": hit, "shots": total})
            hit_s = f"{hit:.4f}" if hit is not None else "—"
            lines.append(f"| {name} | {arm} | {m['2q']} | {m['logf']:.2f} | "
                         f"{pred:.3f} | {fid:.4f} | {tvd:.4f} | {hit_s} |\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    (OUT / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", OUT / "summary.md")


if __name__ == "__main__":
    main()
