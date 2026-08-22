"""大比特线路运行时间对比：QTrial pure（增量路由器）vs 感知 O3。

判据（用户设定）：QTrial 单线路耗时 ≤ 感知O3 的 5 倍。
不达标 → 自动进入 Numba/Cython 编译外层热循环的优化路线。
线路：完整 Benchpress 集中 51-105 比特可解析线路，按比特数均匀抽样。
输出：tables/speed_large_vs_o3.jsonl（逐线路原始数据）+ 汇总。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qiskit import transpile
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
OUT = Path("tables/speed_large_vs_o3.jsonl")
N_SAMPLE = 18


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


def sample_large_circuits():
    """51-105 比特可解析线路，按 (n, name) 排序后均匀抽样 N_SAMPLE 条。

    注意：qasm2.loads 只在抽样后的 18 条上调用——先全量纯解析（快），
    避免对上百条巨型 transpiled 文件做 qiskit 解析（分钟级/条）。
    """
    from qiskit import qasm2
    found = []
    for p in BP.rglob("*.qasm"):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            circ = parse_qasm(text)
        except Exception:
            continue
        if not (51 <= circ.n <= 105):
            continue
        found.append((circ.n, f"{p.parent.name}_{p.stem}", text))
    found.sort()
    if len(found) > N_SAMPLE:
        idx = [round(i * (len(found) - 1) / (N_SAMPLE - 1))
               for i in range(N_SAMPLE)]
        found = [found[i] for i in idx]
    out = []
    for n, name, text in found:
        try:
            qc = qasm2.loads(sanitize_qasm(text))
        except Exception:
            qc = None
        out.append((n, name, text, qc))
    return out


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    target = build_target(spec)

    policy, ck = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    cases = sample_large_circuits()
    print(f"speed test: {len(cases)} large circuits "
          f"({cases[0][0]}-{cases[-1][0]} qubits)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, name, ptext, qc in cases:
        row = {"circuit": name, "n": n}
        t0 = time.perf_counter()
        try:
            mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule="fidelity",
                                use_post=True)
            res = mapper.map_circuit(parse_qasm(ptext), circuit_id=name)
            row["qt_swaps"] = res["swap_count"]
            row["qt_fidelity"] = res["metrics"]["est_fidelity"]
        except Exception as e:
            row["qt_error"] = f"{type(e).__name__}: {str(e)[:60]}"
        row["qt_wall"] = round(time.perf_counter() - t0, 2)
        if qc is not None:
            qc_clean = qc.copy()
            qc_clean.data = [i for i in qc_clean.data
                             if i.operation.name not in ("measure", "barrier")]
            t0 = time.perf_counter()
            try:
                out_qc = transpile(qc_clean,
                                   basis_gates=["rz", "sx", "x", "cz"],
                                   optimization_level=3,
                                   seed_transpiler=42, target=target)
                row["aw_o3_swaps"] = out_qc.count_ops().get("swap", 0)
            except Exception as e:
                row["aw_o3_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            row["aw_o3_wall"] = round(time.perf_counter() - t0, 2)
        if "qt_wall" in row and "aw_o3_wall" in row:
            row["ratio"] = round(row["qt_wall"] / max(row["aw_o3_wall"], 1e-6), 2)
        rows.append(row)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        print(f"  {name:44s} n={n:3d} qt={row.get('qt_wall', '-')}s "
              f"aw_o3={row.get('aw_o3_wall', '-')}s "
              f"ratio={row.get('ratio', '-')}", flush=True)

    ratios = [r["ratio"] for r in rows if "ratio" in r]
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        mean = sum(ratios) / len(ratios)
        worst = ratios[-1]
        verdict = "PASS" if worst <= 5.0 else "FAIL"
        print(f"\nratio: mean={mean:.2f} median={med:.2f} worst={worst:.2f}")
        print(f"verdict vs 5x criterion: {verdict}"
              f"{' -> Numba/Cython route' if verdict == 'FAIL' else ''}")
        with open(OUT.with_suffix(".summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"mean={mean:.2f} median={med:.2f} worst={worst:.2f} "
                    f"verdict={verdict}\n")
            f.write(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
