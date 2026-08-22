"""噪声感知 O1/O3 基线评测：qiskit Target（合成校准）驱动的 VF2PostLayout 打分。

回答：同样引入噪声感知机制后，稀疏奖励 RL（QTrial+SABRE）是否仍有优势。

用法: python scripts/noise_aware_o1_bench.py [--small-only]
输出: tables/noise_aware/{bucket}_rows.jsonl（aware_o1/aware_o3 新列）
      tables/noise_aware/report.md（与 QTrial/盲目 O1/pytket/Cqlib 原生合并对比）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pipeline.metrics import compute_metrics
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements

CACHE = Path("data/mqtbench/stratified")
OUT = Path("tables/noise_aware")


def build_target(spec) -> Target:
    calib = spec.calib
    t = Target(num_qubits=spec.n)
    t.add_instruction(RZGate(0.0), properties={
        (q,): InstructionProperties(error=float(calib.err_1q[q]), duration=1.0)
        for q in range(spec.n)})
    t.add_instruction(CZGate(), properties={
        (a, b): InstructionProperties(error=float(e), duration=1.0)
        for (a, b), e in calib.err_2q.items() for a, b in ((a, b), (b, a))})
    t.add_instruction(XGate(), properties={(q,): None for q in range(spec.n)})
    t.add_instruction(SXGate(), properties={(q,): None for q in range(spec.n)})
    return t


def circuits(small_only=False):
    from qiskit import qasm2 as _q
    out = {"small": [], "medium": []}
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size <= 10:
            b = "small"
        elif size <= 50:
            b = "medium"
        else:
            continue  # 大层已砍（cqlib 决策），噪声感知基线同样只在 ≤50 层对比
        out[b].append((name, _q.loads(sanitize_qasm(p.read_text(encoding="utf-8")))))
    if small_only:
        out.pop("medium", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-only", action="store_true")
    ap.add_argument("--medium-only", action="store_true")
    args = ap.parse_args()

    dev_cfg = load_device_config()
    spec = build_tianyan287_spec(dev_cfg)
    target = build_target(spec)
    calib = spec.calib

    buckets = circuits()
    if args.small_only:
        buckets = {"small": buckets["small"]}
    if args.medium_only:
        buckets = {"medium": buckets["medium"]}

    OUT.mkdir(parents=True, exist_ok=True)
    for b, lst in buckets.items():
        with open(OUT / f"{b}_rows.jsonl", "w", encoding="utf-8") as f:
            for name, qc in lst:
                qc_clean, _ = strip_measurements(qc)
                row = {"circuit": name, "n": qc_clean.num_qubits}
                for opt in (1, 3):
                    t0 = time.time()
                    try:
                        out_qc = transpile(qc_clean, target=target,
                                           basis_gates=["rz", "sx", "x", "cz"],
                                           optimization_level=opt,
                                           seed_transpiler=42)
                        m = compute_metrics(
                            out_qc, out_qc.count_ops().get("swap", 0), calib)
                        row.update({
                            f"aware_o{opt}_swaps": m["swap_count"],
                            f"aware_o{opt}_depth": m["depth"],
                            f"aware_o{opt}_twoq_depth": m["twoq_depth"],
                            f"aware_o{opt}_fidelity": m["est_fidelity"],
                            f"aware_o{opt}_mean2q": m["mean_2q_err"],
                            f"aware_o{opt}_wall": round(time.time() - t0, 2),
                        })
                    except Exception as e:
                        row[f"aware_o{opt}_error"] = f"{type(e).__name__}: {str(e)[:60]}"
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
                print(f"  {name:30s} O1 {row.get('aware_o1_swaps', '-')}/"
                      f"{row.get('aware_o1_fidelity', '-')} | "
                      f"O3 {row.get('aware_o3_swaps', '-')}/"
                      f"{row.get('aware_o3_fidelity', '-')}"
                      f"{row.get('aware_o3_error', '')}", flush=True)
        print(f"[noise_aware] {b} done ({len(lst)} circuits)", flush=True)
    print("[noise_aware] all done")


if __name__ == "__main__":
    main()
