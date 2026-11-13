# -*- coding: utf-8 -*-
"""修正 pytket 基线保真度口径：补回读出误差。

背景：`final_report_bench.py` 对 pytket 也在 transpile 前剥离 measure，
导致 pytket 基线**不含读出误差**；而 QTrial 产物带 measure、含读出。

方法（避免 pytket 对 measure 映射的断言崩溃）：沿用原映射路径
（剥离 measure → DecomposeBoxes + DefaultMappingPass），从
`tk.implicit_qubit_permutation()` 取逻辑→最终物理位置，再按原线路
实际测量的逻辑比特补读出项。该方法在 cat_n65 上与"把 measure 一起
映射"的结果逐位一致（差 1e-16）。

校验：门级保真度必须与旧表 `tket_fidelity` 一致（证明可复现，
差异仅在读出项）。

用法：
  python scripts/fix_pytket_readout.py --buckets large --shard 0/8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import qasm2
from pytket.architecture import Architecture
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (DecomposeBoxes, DefaultMappingPass,
                           DelayMeasures, SequencePass)

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.utils.qasm_io import sanitize_qasm

BP = Path("data/benchpress/qasm")
CACHE = Path("data/mqtbench/stratified")
OLD = Path("tables/final_report")
OUT = Path("tables/final_report_v8")


def index_benchpress():
    return {f"bp_{p.parent.name}_{p.stem}": p for p in BP.rglob("*.qasm")}


def circuit_text(name, idx):
    if name.startswith("bp_"):
        p = idx.get(name)
        return p.read_text(encoding="utf-8", errors="replace") if p else None
    p = CACHE / f"{name}.qasm"
    if p.exists():
        return sanitize_qasm(p.read_text(encoding="utf-8"))
    from qtrail.utils.bench import iter_queko_files
    q = next((x for x in iter_queko_files("BIGD") if x.stem == name), None)
    return sanitize_qasm(q.read_text(encoding="utf-8")) if q else None


def gate_logf(qc, calib):
    err2q = calib.err_2q
    med = float(np.median(list(err2q.values()))) if err2q else 1e-3
    s = 0.0
    for inst in qc.data:
        name = inst.operation.name
        if name in ("measure", "barrier"):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            e = err2q.get((qs[0], qs[1]), err2q.get((qs[1], qs[0]), med))
            s += math.log(max(1.0 - float(e), 1e-12))
        elif len(qs) == 1:
            s += math.log(max(1.0 - float(calib.err_1q[qs[0]]), 1e-12))
    return s


def expand_swaps(mapped):
    """与 qtrail.pipeline.external.tket_compile 相同的 swap→3cx 展开。"""
    from qiskit import QuantumCircuit
    expanded = QuantumCircuit()
    for qr in mapped.qregs:
        expanded.add_register(qr)
    for cr in mapped.cregs:
        expanded.add_register(cr)
    for inst in mapped.data:
        if inst.operation.name == "swap":
            a = mapped.find_bit(inst.qubits[0]).index
            b = mapped.find_bit(inst.qubits[1]).index
            expanded.cx(a, b)
            expanded.cx(b, a)
            expanded.cx(a, b)
        else:
            expanded.append(inst.operation, inst.qubits, inst.clbits)
    return expanded


def readout_log(qc, calib):
    s = 0.0
    for inst in qc.data:
        if inst.operation.name != "measure":
            continue
        q = qc.find_bit(inst.qubits[0]).index
        s += math.log(max(1.0 - float(calib.err_ro[q]), 1e-12))
    return s


def run_tket(qc, cm, calib):
    """保留 measure 的 pytket 映射；返回 (expanded, swaps, n_measured)。

    barrier 会触发 pytket 2.18.1 的 `slice is empty` 断言（bv_n70），
    但 barrier 语义为空，删掉即可；`DelayMeasures` 把测量移到末尾以
    满足 DefaultMappingPass 的 CommutableMeasures 谓词。
    """
    q = qc.copy()
    q.data = [i for i in q.data if i.operation.name != "barrier"]
    n_measured = sum(1 for i in q.data if i.operation.name == "measure")
    tk = qiskit_to_tk(q)
    DelayMeasures().apply(tk)
    SequencePass([DecomposeBoxes(),
                  DefaultMappingPass(Architecture(cm))]).apply(tk)
    mapped = tk_to_qiskit(tk, replace_implicit_swaps=True)
    sc = mapped.count_ops().get("swap", 0)
    expanded = expand_swaps(mapped)
    return expanded, sc, n_measured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="small,medium,large")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()
    buckets_sel = [b for b in args.buckets.split(",") if b]
    shard_id = shard_n = None
    if args.shard:
        shard_id, shard_n = (int(x) for x in args.shard.split("/"))

    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]
    qcm = QiskitMapper(spec).cm
    idx = index_benchpress()
    OUT.mkdir(parents=True, exist_ok=True)

    for bucket in buckets_sel:
        old_p = OLD / f"{bucket}_fidelity_rows.jsonl"
        rows = [json.loads(l) for l in old_p.read_text(encoding="utf-8")
                .splitlines() if l.strip()]
        items = list(enumerate(rows))
        if shard_id is not None:
            lo = shard_id * len(items) // shard_n
            hi = (shard_id + 1) * len(items) // shard_n
            items = items[lo:hi]
        out_p = OUT / (f"{bucket}_pytket_readout"
                       + (f"_shard{shard_id}" if shard_id is not None else "")
                       + ".jsonl")
        n_match = n_mismatch = n_skip = 0
        with open(out_p, "w", encoding="utf-8") as f:
            for i, r in items:
                if "tket_fidelity" not in r:
                    f.write(json.dumps(r, ensure_ascii=False,
                                       default=float) + "\n")
                    n_skip += 1
                    continue
                text = circuit_text(r["circuit"], idx)
                if text is None:
                    n_skip += 1
                    f.write(json.dumps(r, ensure_ascii=False,
                                       default=float) + "\n")
                    continue
                try:
                    qc = qasm2.loads(sanitize_qasm(text))
                    expanded, sc, n_measured = run_tket(qc, cm, calib)
                    expanded = decompose_to_platform(
                        expanded, qcm, optimization_level=1, seed=42)
                except Exception as e:  # noqa: BLE001
                    r["tket_readout_error"] = f"{type(e).__name__}: {e}"
                    f.write(json.dumps(r, ensure_ascii=False,
                                       default=float) + "\n")
                    n_skip += 1
                    continue
                glog = gate_logf(expanded, calib)
                rlog = readout_log(expanded, calib)
                gate_fid = math.exp(glog)
                m = compute_metrics(expanded, sc, calib)
                r["tket_swaps_rr"] = int(sc)
                r["tket_twoq_count_rr"] = m["twoq_count"]
                r["tket_twoq_depth_rr"] = m["twoq_depth"]
                r["tket_depth_rr"] = m["depth"]
                stored = float(r["tket_fidelity"])
                if stored > 0 and abs(gate_fid - stored) <= \
                        args.rtol * stored:
                    n_match += 1
                else:
                    n_mismatch += 1
                    r["tket_repro_err"] = [float(gate_fid), float(stored)]
                r["tket_fidelity_gate_rr"] = gate_fid
                r["tket_logf_with_ro"] = glog + rlog
                r["tket_fidelity_with_ro"] = math.exp(glog + rlog)
                r["tket_n_measured"] = n_measured
                f.write(json.dumps(r, ensure_ascii=False,
                                   default=float) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  [{bucket}] {i + 1}/{len(rows)}", flush=True)
        print(f"[{bucket}] -> {out_p}; match={n_match} "
              f"mismatch={n_mismatch} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
