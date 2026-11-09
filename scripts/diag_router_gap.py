# -*- coding: utf-8 -*-
"""路由器差距诊断 v3：自研 sabre_route（新评分） vs qiskit SabreSwap。

v2 结论：历史惩罚无界→超换 5 倍；去掉后 0.5^t 门权衰减+确定性 argmin→死循环。
v3 验证新评分（扩展集 0.5/n、门权恒定、有界禁忌、种子随机平局、10n 逃逸）
与 qiskit decay/lookahead 的同布局对照。qiskit 侧补 FullAncillaAllocation。

输出：tables/compare_models/diag_router_gap.jsonl + 打印汇总。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (ApplyLayout, BasisTranslator,
                                      EnlargeWithAncilla,
                                      FullAncillaAllocation, SabreSwap)
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler import Layout

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import load_qasm_file
from qtrail.pure.router import sabre_route

OUT = Path("tables/compare_models/diag_router_gap.jsonl")

CONFIGS = [
    ("无禁忌", dict(tabu_window=0)),
    ("前瞻.5", dict(tabu_window=0, lookahead_w=0.5)),
    ("前瞻.5噪声10", dict(tabu_window=0, lookahead_w=0.5, noise_w=10)),
    ("前瞻.5噪声30", dict(tabu_window=0, lookahead_w=0.5, noise_w=30)),
    ("噪声10", dict(tabu_window=0, noise_w=10)),
    ("噪声30", dict(tabu_window=0, noise_w=30)),
]


def find_file(name):
    if name.startswith("benchpress/"):
        parts = name.split("/")
        return Path("data/benchpress/qasm") / parts[1] / f"{parts[2]}.qasm"
    if name.startswith("queko_BIGD/"):
        return Path("data/Queko/BIGD") / f"{name.split('/')[1]}.qasm"
    return None


def qiskit_route(qc, cm, heuristic, seed=42, trials=1):
    # 与自研侧同口径：先基翻译（ccx→cx 等，BasisTranslator），再路由
    qc = PassManager([BasisTranslator(SessionEquivalenceLibrary,
                                      target_basis=["u3", "cx"])]).run(qc)
    init = Layout({qc.qubits[i]: i for i in range(qc.num_qubits)})
    pm = PassManager([SetLayout(init), FullAncillaAllocation(cm),
                      EnlargeWithAncilla(), ApplyLayout(),
                      SabreSwap(cm, heuristic=heuristic, seed=seed,
                                trials=trials)])
    routed = pm.run(qc)
    return routed.count_ops().get("swap", 0)


def main():
    spec = build_tianyan287_spec(load_device_config())
    cm = CouplingMap([(i, j) for i in range(spec.n)
                      for j in range(i + 1, spec.n) if spec.adj[i, j]])

    rows = [json.loads(l) for l in Path(
        "tables/compare_models/inj_v6_rows.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    ok = sorted([r for r in rows if "v5_fidelity" in r and r.get("n")],
                key=lambda x: x["n"])
    bands = [(0, 10), (11, 20), (21, 30), (31, 60), (61, 100), (101, 200)]
    # 禁忌机制参数网格：聚焦差距最大的线路（dtc_100/gf2^32/gf2^16）
    extra = ["benchpress/feynman/gf2^16_mult",
             "benchpress/feynman/gf2^32_mult",
             "benchpress/dtc/dtc_100_cx_12345"]
    by_name = {r["circuit"]: r for r in ok}
    picked = [by_name[name] for name in extra
              if name in by_name and find_file(name) is not None]
    print("cases:", [(r["circuit"], r["n"]) for r in picked], flush=True)

    results = []
    for r in picked:
        name, n = r["circuit"], r["n"]
        src = find_file(name)
        circ = load_qasm_file(src)
        n2q = sum(1 for i in circ.ops if i.nq == 2)
        layout = {i: i for i in range(n)}
        row = {"circuit": name, "n": n, "n2q_orig": n2q}
        for label, kw in CONFIGS:
            t0 = time.time()
            routed, swaps, _ = sabre_route(circ, spec, dict(layout),
                                           seed=42, **kw)
            row[f"ours_{label}"] = swaps
            row[f"ours_{label}_wall"] = round(time.time() - t0, 2)
            print(f"  {name} ours[{label}] = {swaps}", flush=True)
        try:
            qc = qasm2.load(str(src))
        except Exception as e:
            row["qiskit_load_err"] = f"{type(e).__name__}:{str(e)[:40]}"
            results.append(row)
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {name} qiskit load failed: {row['qiskit_load_err']}",
                  flush=True)
            continue
        t0 = time.time()
        try:
            row["qiskit_basic_t1"] = qiskit_route(qc, cm, "basic", trials=1)
        except Exception as e:
            row["qiskit_basic_t1"] = f"err:{type(e).__name__}:{str(e)[:40]}"
        row["qiskit_basic_t1_wall"] = round(time.time() - t0, 2)
        print(f"  {name} qiskit[basic] = {row['qiskit_basic_t1']}", flush=True)
        for t in (1, 4):
            t0 = time.time()
            try:
                row[f"qiskit_decay_t{t}"] = qiskit_route(
                    qc, cm, "decay", trials=t)
            except Exception as e:
                row[f"qiskit_decay_t{t}"] = f"err:{type(e).__name__}:{str(e)[:40]}"
            row[f"qiskit_decay_t{t}_wall"] = round(time.time() - t0, 2)
            print(f"  {name} qiskit[decay t={t}] = "
                  f"{row[f'qiskit_decay_t{t}']}", flush=True)
        if n <= 30:
            t0 = time.time()
            try:
                row["qiskit_lookahead_t1"] = qiskit_route(
                    qc, cm, "lookahead", trials=1)
            except Exception as e:
                row["qiskit_lookahead_t1"] = \
                    f"err:{type(e).__name__}:{str(e)[:40]}"
            row["qiskit_lookahead_t1_wall"] = round(time.time() - t0, 2)
            print(f"  {name} qiskit[lookahead] = "
                  f"{row['qiskit_lookahead_t1']}", flush=True)
        results.append(row)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n== 汇总（swap 数） ==")
    keys = [f"ours_{l}" for l, _ in CONFIGS] \
        + ["qiskit_basic_t1", "qiskit_decay_t1", "qiskit_decay_t4",
           "qiskit_lookahead_t1"]
    print(f"{'circuit':44s} " + " ".join(f"{k:>16s}" for k in keys))
    for row in results:
        print(f"{row['circuit'][:44]:44s} " + " ".join(
            f"{str(row.get(k, '--')):>16s}" for k in keys))


if __name__ == "__main__":
    main()
