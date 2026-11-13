# -*- coding: utf-8 -*-
"""修正基线保真度口径：补回读出误差（2026-11-13 发现的口径不对称）。

问题：`scripts/final_report_bench.py` 在跑 qiskit 基线前把 measure/barrier
从输入线路中剥掉（`qc_clean.data = [...]`），而 `compute_metrics` 的
乘积保真度按 circuit 里实际存在的 measure 计读出项。结果：
- QTrial 的 `routed_circuit` 带 measure → **计读出**；
- 全部 qiskit 基线（盲目/感知 O1/O3）无 measure → **不计读出**。
在 65–105 比特线路上读出项约 0.3–0.5 倍，足以把结论从"胜"翻成"负"。

本脚本对每条线路按原口径（seed_transpiler=42、同 basis、同 target）重跑
四条 qiskit 基线，先校验重跑的门级保真度与旧行完全一致（证明可复现），
再用 `final_index_layout()` 补上每个逻辑比特最终所在物理比特的读出误差，
输出修正后的基线列。原始表不动，修正表另存。

用法：
  python scripts/fix_baseline_readout.py --buckets large
  python scripts/fix_baseline_readout.py --buckets small,medium,large
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import qasm2, transpile
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

BP = Path("data/benchpress/qasm")
CACHE = Path("data/mqtbench/stratified")
OLD = Path("tables/final_report")
OUT = Path("tables/final_report_v8")
ARMS = (("b_o1", 1, False), ("b_o3", 3, False),
        ("aw_o1", 1, True), ("aw_o3", 3, True))


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


def index_benchpress() -> dict[str, Path]:
    """只建名字→路径索引，不解析任何文件。"""
    idx = {}
    for p in BP.rglob("*.qasm"):
        idx[f"bp_{p.parent.name}_{p.stem}"] = p
    return idx


def load_qiskit_circuits(wanted: set[str]) -> dict:
    idx = index_benchpress()
    out = {}
    for name in sorted(wanted):
        try:
            if name.startswith("bp_"):
                p = idx.get(name)
                if p is None:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            elif (CACHE / f"{name}.qasm").exists():
                text = sanitize_qasm((CACHE / f"{name}.qasm").read_text(
                    encoding="utf-8"))
            else:
                p = next((q for q in iter_queko_files("BIGD")
                          if q.stem == name), None)
                if p is None:
                    continue
                text = sanitize_qasm(p.read_text(encoding="utf-8"))
            out[name] = qasm2.loads(sanitize_qasm(text))
        except Exception as e:
            print(f"  [load-skip] {name}: {type(e).__name__}", flush=True)
    return out


def gate_logf(qc, calib) -> float:
    """门级对数保真度（与 qtrail.pipeline.metrics 的乘积模型同口径，
    但全程 log 空间，避免大线路乘积下溢丢信息）。"""
    err2q = calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
    s = 0.0
    for inst in qc.data:
        name = inst.operation.name
        if name in ("measure", "barrier"):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            e = err2q.get((qs[0], qs[1]), err2q.get((qs[1], qs[0]),
                                                    median2q))
            s += math.log(max(1.0 - float(e), 1e-12))
        elif len(qs) == 1:
            s += math.log(max(1.0 - float(calib.err_1q[qs[0]]), 1e-12))
    return s


def readout_log(err_ro, layout) -> float:
    s = 0.0
    for p in layout:
        s += math.log(max(1.0 - float(err_ro[int(p)]), 1e-12))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="small,medium,large")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="重跑门级保真度与旧行的最大相对偏差")
    args = ap.parse_args()
    buckets_sel = [b for b in args.buckets.split(",") if b]

    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    target = build_target(spec)
    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]
    OUT.mkdir(parents=True, exist_ok=True)

    for bucket in buckets_sel:
        old_p = OLD / f"{bucket}_fidelity_rows.jsonl"
        if not old_p.exists():
            print(f"[{bucket}] 缺旧表 {old_p}", flush=True)
            continue
        rows = [json.loads(l) for l in old_p.read_text(encoding="utf-8")
                .splitlines() if l.strip()]
        wanted = {r["circuit"] for r in rows}
        qcs = load_qiskit_circuits(wanted)
        print(f"[{bucket}] rows={len(rows)} qiskit_loaded={len(qcs)}",
              flush=True)
        n_match = n_mismatch = n_missing = 0
        from qtrail.pipeline.metrics import compute_metrics as qk_metrics
        for i, r in enumerate(rows):
            name = r["circuit"]
            qc = qcs.get(name)
            if qc is None:
                n_missing += 1
                continue
            measured = [qc.find_bit(inst.qubits[0]).index
                        for inst in qc.data
                        if inst.operation.name == "measure"]
            qc_clean = qc.copy()
            qc_clean.data = [inst for inst in qc_clean.data
                             if inst.operation.name not in ("measure",
                                                            "barrier")]
            for prefix, opt, aware in ARMS:
                if f"{prefix}_fidelity" not in r:
                    continue
                kw = {"target": target} if aware else {"coupling_map": cm}
                out_qc = transpile(qc_clean,
                                   basis_gates=["rz", "sx", "x", "cz"],
                                   optimization_level=opt,
                                   seed_transpiler=42, **kw)
                m = qk_metrics(out_qc, 0, calib)
                gate_fid = m["est_fidelity"]
                stored = float(r[f"{prefix}_fidelity"])
                if stored > 0 and abs(gate_fid - stored) <= args.rtol * stored:
                    n_match += 1
                else:
                    n_mismatch += 1
                    r[f"{prefix}_repro_err"] = [
                        float(gate_fid), float(stored)]
                layout = out_qc.layout.final_index_layout()
                # 只对原线路真正测量的逻辑比特计读出（与 QTrial 的
                # routed_circuit 计法对称；无 measure 的线路两侧都不计）
                ro = readout_log(
                    calib.err_ro, [layout[q] for q in measured])
                glog = gate_logf(out_qc, calib)
                r[f"{prefix}_fidelity_gate"] = gate_fid
                r[f"{prefix}_logf_gate"] = glog
                r[f"{prefix}_logf_readout"] = ro
                r[f"{prefix}_logf_with_ro"] = glog + ro
                r[f"{prefix}_fidelity_with_ro"] = gate_fid * math.exp(ro)
                r[f"{prefix}_n_measured"] = len(measured)
            if (i + 1) % 10 == 0:
                print(f"  [{bucket}] {i + 1}/{len(rows)}", flush=True)
        out_p = OUT / f"{bucket}_baseline_readout.jsonl"
        out_p.write_text("".join(json.dumps(r, ensure_ascii=False,
                                            default=float) + "\n"
                                 for r in rows), encoding="utf-8")
        print(f"[{bucket}] {len(rows)} 行 -> {out_p}; "
              f"门级复现 match={n_match} mismatch={n_mismatch} "
              f"missing={n_missing}", flush=True)


if __name__ == "__main__":
    main()
