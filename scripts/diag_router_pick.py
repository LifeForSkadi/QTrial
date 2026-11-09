# -*- coding: utf-8 -*-
"""路由器默认配置选择：4 候选配置 × ~10 条线路（含计时）。

配置：无禁忌 / 边二值w5 / 比特平滑3 / 比特二值w1。
输出：tables/compare_models/diag_router_pick.jsonl + 汇总。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import load_qasm_file
from qtrail.pure.router import sabre_route

OUT = Path("tables/compare_models/diag_router_pick.jsonl")

CONFIGS = [
    ("无禁忌", dict(tabu_window=0)),
    ("边二值w5", dict()),
    ("比特平滑3", dict(tabu_scope="qubit", tabu_decay=0.5, tabu_mult=3)),
    ("比特二值w1", dict(tabu_scope="qubit", tabu_window=1)),
]

NAMES = ["benchpress/feynman/barenco_tof_3",
         "benchpress/feynman/barenco_tof_10",
         "benchpress/feynman/adder_8",
         "benchpress/feynman/gf2^4_mult",
         "benchpress/feynman/gf2^7_mult",
         "benchpress/feynman/gf2^16_mult",
         "benchpress/feynman/gf2^32_mult",
         "benchpress/dtc/dtc_100_cx_12345",
         "benchpress/clifford/clifford_100_12345"]


def find_file(name):
    if name.startswith("benchpress/"):
        parts = name.split("/")
        return Path("data/benchpress/qasm") / parts[1] / f"{parts[2]}.qasm"
    if name.startswith("queko_BIGD/"):
        return Path("data/Queko/BIGD") / f"{name.split('/')[1]}.qasm"
    return None


def main():
    spec = build_tianyan287_spec(load_device_config())
    results = []
    for name in NAMES:
        src = find_file(name)
        if src is None:
            print(f"[skip] {name}", flush=True)
            continue
        circ = load_qasm_file(src)
        layout = {i: i for i in range(circ.n)}
        row = {"circuit": name, "n": circ.n,
               "n2q": sum(1 for i in circ.ops if i.nq == 2)}
        for label, kw in CONFIGS:
            t0 = time.time()
            _, swaps, _ = sabre_route(circ, spec, dict(layout), seed=42, **kw)
            row[f"{label}_swaps"] = swaps
            row[f"{label}_wall"] = round(time.time() - t0, 2)
            print(f"  {name[:52]:52s} {label} = {swaps} "
                  f"({row[f'{label}_wall']}s)", flush=True)
        results.append(row)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n== 汇总 ==")
    keys = [f"{l}_swaps" for l, _ in CONFIGS]
    print(f"{'circuit':44s} " + " ".join(f"{k:>14s}" for k in keys))
    totals = {k: 0 for k in keys}
    for row in results:
        print(f"{row['circuit'][:44]:44s} " + " ".join(
            f"{row[k]:>14d}" for k in keys))
        for k in keys:
            totals[k] += row[k]
    print(f"{'TOTAL':44s} " + " ".join(f"{totals[k]:>14d}" for k in keys))


if __name__ == "__main__":
    main()
