# -*- coding: utf-8 -*-
"""真机对比映射（v8 vs Qiskit O3）→ QCIS（天衍实时拓扑）。

对每条线路映射两臂：
  - v8：自研管线（PureMapper，pre_cancel 默认开，routing_seeds=8）
  - o3：Qiskit transpile(optimization_level=3) 同 target / 同 seed

测量顺序统一为**逻辑比特 0..n-1**（物理位取各臂 final layout），
因此平台返回的比特串可直接与理想分布按同一顺序比较。

用法：
  python scripts/real_machine_v9_map.py qft5 qft6 ghz10 --machine tianyan176
输出：
  out/real_machine_v9/<machine>/<name>_<arm>.qcis
  out/real_machine_v9/<machine>/<name>_<arm>_metrics.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices.adapter import download_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.export import to_qcis
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt")


def find_qasm(name: str) -> Path:
    cands = [Path("examples") / f"{name}.qasm",
             Path("out/real_machine") / f"{name}.qasm"]
    if name.startswith("qft") and name[3:].isdigit():
        cands.append(Path("data/benchpress/qasm/qft")
                     / f"qft_N{int(name[3:]):03d}.qasm")
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(name)


def build_target(spec):
    from qiskit.transpiler import InstructionProperties, Target
    from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
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


def with_logical_measures(circ: Circuit, final_layout: dict, n: int) -> Circuit:
    """把测量统一为逻辑顺序 0..n-1（物理位 = final_layout[q]）。"""
    out = Circuit(circ.n, name=circ.name)
    out.ops = [i for i in circ.ops if i.name != "measure"]
    out.measures = []
    for q in range(n):
        out.measures.append(
            Inst("measure", (int(final_layout.get(q, q)),), cbits=q))
    return out


def logf(circ: Circuit, spec) -> float:
    err2q = spec.calib.err_2q
    med = float(np.median(list(err2q.values()))) if err2q else 1e-3
    s = 0.0
    for inst in circ.ops:
        if inst.nq == 2:
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            e = err2q.get((a, b), err2q.get((b, a), med))
            s += math.log(max(1.0 - float(e), 1e-12))
        elif inst.nq == 1:
            s += math.log(max(1.0 - float(spec.calib.err_1q[
                int(inst.qubits[0])]), 1e-12))
    for m in circ.measures:
        s += math.log(max(1.0 - float(spec.calib.err_ro[
            int(m.qubits[0])]), 1e-12))
    return s


def counts_of(circ: Circuit):
    return (sum(1 for i in circ.ops if i.nq == 1),
            sum(1 for i in circ.ops if i.nq == 2),
            len(circ.measures))


def qc_to_circuit(qc) -> Circuit:
    c = Circuit(qc.num_qubits)
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        qs = tuple(int(qc.find_bit(q).index) for q in inst.qubits)
        params = tuple(float(p) for p in inst.operation.params)
        c.append(Inst(name, qs, params))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--machine", default="tianyan176")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    spec = download_tianyan287_spec(args.machine)
    if spec is None:
        raise SystemExit(f"cannot build live spec for {args.machine}")
    target = build_target(spec)
    policy, _ = QAPolicy.load_checkpoint(args.checkpoint, device_n=spec.n,
                                         map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    out_dir = Path("out/real_machine_v9") / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        p = find_qasm(name)
        text = p.read_text(encoding="utf-8")
        circ = parse_qasm(text)
        n = circ.n
        # 补齐测量到全部逻辑比特（QCIS 需要；理想分布按全态比较）
        circ.measures = [Inst("measure", (q,), cbits=q) for q in range(n)]

        # ---------------- v8
        t0 = time.time()
        torch.manual_seed(args.seed)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cpu",
                            seed=args.seed, selection_rule="fidelity",
                            use_post=True, pipeline="v5",
                            routing_seeds=args.seeds)
        res = mapper.map_circuit(circ, circuit_id=name)
        v8 = with_logical_measures(res["routed_circuit"],
                                   res["final_layout"], n)
        v8_qcis = to_qcis(v8, labels=spec.qubit_labels)
        (out_dir / f"{name}_v8.qcis").write_text(v8_qcis, encoding="utf-8")
        v8_1, v8_2, v8_ro = counts_of(v8)
        v8_m = {"machine": args.machine, "arm": "v8", "n": n,
                "source": str(p), "1q": v8_1, "2q": v8_2, "ro": v8_ro,
                "logf": round(logf(v8, spec), 4),
                "swaps": res["swap_count"],
                "wall_s": round(time.time() - t0, 2),
                "layout": {str(k): int(v) for k, v in res["layout"].items()},
                "final_layout": {str(k): int(v)
                                 for k, v in res["final_layout"].items()},
                "measure_phys": [int(v8.measures[q].qubits[0])
                                 for q in range(n)]}
        (out_dir / f"{name}_v8_metrics.json").write_text(
            json.dumps(v8_m, ensure_ascii=False, indent=1), encoding="utf-8")

        # ---------------- Qiskit O3
        from qiskit import qasm2, transpile
        from qtrail.utils.qasm_io import sanitize_qasm
        t0 = time.time()
        qc = qasm2.loads(sanitize_qasm(text))
        qc.data = [i for i in qc.data
                   if i.operation.name not in ("measure", "barrier")]
        out = transpile(qc, basis_gates=["rz", "sx", "x", "cz"],
                        optimization_level=3, seed_transpiler=args.seed,
                        target=target)
        fin = list(out.layout.final_index_layout())[:n]
        aw = qc_to_circuit(out)
        aw = with_logical_measures(aw, {q: int(fin[q]) for q in range(n)}, n)
        aw_qcis = to_qcis(aw, labels=spec.qubit_labels)
        (out_dir / f"{name}_o3.qcis").write_text(aw_qcis, encoding="utf-8")
        aw_1, aw_2, aw_ro = counts_of(aw)
        aw_m = {"machine": args.machine, "arm": "o3", "n": n,
                "source": str(p), "1q": aw_1, "2q": aw_2, "ro": aw_ro,
                "logf": round(logf(aw, spec), 4),
                "wall_s": round(time.time() - t0, 2),
                "final_layout": {str(q): int(fin[q]) for q in range(n)},
                "measure_phys": [int(aw.measures[q].qubits[0])
                                 for q in range(n)]}
        (out_dir / f"{name}_o3_metrics.json").write_text(
            json.dumps(aw_m, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{name}] v8 1q={v8_1} 2q={v8_2} logf={v8_m['logf']:.2f} | "
              f"o3 1q={aw_1} 2q={aw_2} logf={aw_m['logf']:.2f}", flush=True)


if __name__ == "__main__":
    main()
