"""分层评测：按量子比特规模分层对比 QTrial / SABRE O1 / SABRE O3 / pytket。

层划分：
  小型   <= 10 比特   （新生成 MQTBench n=5/8/10）
  中小型 11-20 比特   （重切现有：BNTF-16、BIGD-20、MQTBench n=15）
  中型   21-50 比特   （新生成 MQTBench n=25/50）
  大型   51-105 比特  （新生成 MQTBench n=100 + 满占用 n=105）
每层内 QUEKO BIGD 另按密度 0.1-0.7 逐档呈现。

用法: python scripts/eval_stratified.py [--skip-large]
输出: tables/stratified/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, TABLES_DIR, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import sabre_swap_count, sabre_transpile
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.utils.qasm_io import decompose_circuit, strip_measurements

CKPT = "checkpoints/tianyan-287_gat_combined_calib_best.pt"
CACHE = Path("data/mqtbench/stratified")

# (基准类型, 规模) — 新生成层
GENERATE = {
    "small": [("qft", 5), ("qft", 8), ("qft", 10), ("ghz", 5), ("ghz", 8),
              ("ghz", 10), ("qaoa", 6), ("qaoa", 8), ("wstate", 5), ("wstate", 8),
              ("dj", 8), ("bv", 10), ("graphstate", 5), ("graphstate", 8),
              ("qpeexact", 8), ("iqpe", 8)],
    "medium": [("qft", 25), ("qft", 50), ("ghz", 25), ("ghz", 50), ("qaoa", 25),
               ("qaoa", 50), ("wstate", 25), ("wstate", 50), ("dj", 25),
               ("bv", 50), ("graphstate", 25), ("graphstate", 50),
               ("qftentangled", 25), ("qwalk", 25)],
    "large": [("qft", 100), ("ghz", 100), ("qaoa", 100), ("wstate", 100),
              ("dj", 100), ("bv", 100), ("graphstate", 100), ("qftentangled", 100),
              ("ghz", 105), ("qft", 105), ("qaoa", 105), ("wstate", 105),
              ("dj", 105), ("graphstate", 105)],
}


def generate_benchmarks():
    """Generate + cache the new benchmark circuits; return {bucket: [(name, qc)]}."""
    from mqt.bench import BenchmarkLevel, get_benchmark
    out = {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for bucket, specs in GENERATE.items():
        out[bucket] = []
        for name, size in specs:
            cache_file = CACHE / f"{name}_{size}.qasm"
            try:
                if cache_file.exists():
                    from qtrail.utils.qasm_io import sanitize_qasm
                    from qiskit import qasm2
                    qc = qasm2.loads(sanitize_qasm(cache_file.read_text(encoding="utf-8")))
                else:
                    qc = get_benchmark(benchmark=name, level=BenchmarkLevel.INDEP,
                                       circuit_size=size, random_parameters=True)
                    qc = decompose_circuit(qc)
                    from qtrail.utils.qasm_io import qasm2_str
                    cache_file.write_text(qasm2_str(qc), encoding="utf-8")
                out[bucket].append((f"mqtbench/{name}_{size}", qc))
                print(f"  [gen] {name}_{size}: {qc.num_qubits}q, "
                      f"{sum(1 for i in qc.data if len(i.qubits)==2)} 2Q gates")
            except Exception as e:
                print(f"  [gen] {name}_{size} SKIP: {type(e).__name__}")
    return out


def tket_compile(qc, spec, calib) -> dict:
    """pytket 编译（口径统一：swap 展开 3 cx）。"""
    from pytket.architecture import Architecture
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.passes import (DecomposeBoxes, DefaultMappingPass, SequencePass)
    edges = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
             if spec.adj[i, j]]
    tk = qiskit_to_tk(qc)
    SequencePass([DecomposeBoxes(), DefaultMappingPass(Architecture(edges))]).apply(tk)
    mapped = tk_to_qiskit(tk, replace_implicit_swaps=True)
    sc = mapped.count_ops().get("swap", 0)
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
            expanded.cx(a, b); expanded.cx(b, a); expanded.cx(a, b)
        else:
            expanded.append(inst.operation, inst.qubits, inst.clbits)
    m = compute_metrics(expanded, sc, calib)
    return {"tket_swaps": sc, "tket_depth": m["depth"], "tket_fidelity": m["est_fidelity"]}


def evaluate(circuits, mapper, spec, calib, with_tket: bool):
    rows = []
    for name, qc in circuits:
        qc_clean, _ = strip_measurements(qc)
        row = {"circuit": name, "n_qubits": qc_clean.num_qubits}
        t0 = time.time()
        try:
            res = mapper.map_circuit(qc_clean, circuit_id=name)
            row.update({"swaps": res.swap_count, "depth": res.metrics["depth"],
                        "twoq": res.metrics["twoq_count"],
                        "fidelity": res.metrics["est_fidelity"],
                        "static_cost": res.metrics["static_cost"],
                        "method": res.method, "wall_s": round(time.time() - t0, 1)})
        except Exception as e:
            row["error"] = str(e)[:100]
        for opt in (1, 3):
            try:
                sc = res.baseline_swaps if opt == 1 and "error" not in row else None
                routed_b = None
                if sc is None:
                    _, routed_b = sabre_swap_count(qc_clean, mapper.cm,
                                                   optimization_level=opt, seed=42)
                    sc = routed_b.count_ops().get("swap", 0)
                final = sabre_transpile(qc_clean, mapper.cm, optimization_level=opt,
                                        seed=42)
                m = compute_metrics(final, sc, calib)
                row[f"o{opt}_swaps"] = sc
                row[f"o{opt}_depth"] = m["depth"]
                row[f"o{opt}_fidelity"] = m["est_fidelity"]
            except Exception as e:
                row[f"o{opt}_error"] = str(e)[:80]
        if with_tket:
            try:
                row.update(tket_compile(qc_clean, spec, calib))
            except Exception as e:
                row["tket_error"] = str(e)[:80]
        rows.append(row)
    return rows


def bucket_table(rows: list, title: str) -> str:
    ok = [r for r in rows if "error" not in r and "swaps" in r]
    if not ok:
        return f"## {title}\n\n无有效数据\n"
    lines = [f"## {title}（{len(ok)}/{len(rows)} 条）\n",
             "| 线路 | n | QTrial SWAP | O1 SWAP | O3 SWAP | tket SWAP | "
             "QTrial fid | O1 fid | O3 fid | tket fid | 方法 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(
            f"| {r['circuit'][:30]} | {r['n_qubits']} | {r['swaps']} | "
            f"{r.get('o1_swaps')} | {r.get('o3_swaps')} | {r.get('tket_swaps','-')} | "
            f"{r['fidelity']:.4f} | {r.get('o1_fidelity',0):.4f} | {r.get('o3_fidelity',0):.4f} | "
            f"{r.get('tket_fidelity',0):.4f} | {r['method']} |")
    lines.append("")
    return "\n".join(lines)


def reslice_existing() -> str:
    """重切现有评测数据（BIGD 按密度、BNTF、MQTBench n=15）。"""
    lines = ["# 现有数据重切（天衍-287，QTrial=噪声感知混合管线）\n"]
    bigd = Path("tables/queko_BIGD_rows.jsonl")
    if bigd.exists():
        rows = [json.loads(l) for l in bigd.read_text(encoding="utf-8").splitlines()]
        import re
        lines.append("## QUEKO BIGD（20 比特）按密度分档\n")
        lines.append("| 密度档 | n | QTrial SWAP | O1 SWAP | O3 SWAP | QTrial 保真度 | O1 保真度 |")
        lines.append("|---|---|---|---|---|---|---|")
        buckets = {}
        for r in rows:
            m = re.search(r"\.(\d+)D1_\.(\d+)D2_", r["circuit"])
            key = f"D1=0.{m.group(1)}, D2=0.{m.group(2)}" if m else "?"
            buckets.setdefault(key, []).append(r)
        for key in sorted(buckets):
            rs = buckets[key]
            lines.append(
                f"| {key} | {len(rs)} | {statistics.mean(r['swaps'] for r in rs):.1f} | "
                f"{statistics.mean(r['qiskit-o1_swaps'] for r in rs):.1f} | "
                f"{statistics.mean(r['qiskit-o3_swaps'] for r in rs):.1f} | "
                f"{statistics.mean(r['fidelity'] for r in rs):.4f} | "
                f"{statistics.mean(r['qiskit-o1_fidelity'] for r in rs):.4f} |")
        lines.append("")
    for bench, title in (("queko_BNTF", "QUEKO BNTF（16 比特）"),
                         ("mqtbench_n15", "MQTBench n=15")):
        p = Path(f"tables/{bench}_rows.jsonl")
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        ok = [r for r in rows if "error" not in r]
        lines.append(f"## {title}（{len(ok)} 条）\n")
        lines.append(f"- QTrial SWAP 均值 {statistics.mean(r['swaps'] for r in ok):.1f} "
                     f"vs O1 {statistics.mean(r['qiskit-o1_swaps'] for r in ok):.1f} "
                     f"vs O3 {statistics.mean(r['qiskit-o3_swaps'] for r in ok):.1f}")
        lines.append(f"- 保真度 QTrial {statistics.mean(r['fidelity'] for r in ok):.4f} "
                     f"vs O1 {statistics.mean(r['qiskit-o1_fidelity'] for r in ok):.4f}\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    import torch
    policy, ckpt = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                            map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=42)

    print("生成新基准...")
    new_benches = generate_benchmarks()
    if args.skip_large:
        new_benches.pop("large", None)

    out_dir = Path("tables/stratified")
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = ["# 分层评测报告（天衍-287，105 比特 15×7 网格）\n"]

    for bucket in ("small", "medium", "large"):
        if bucket not in new_benches:
            continue
        print(f"\n评测 {bucket} 层 ({len(new_benches[bucket])} 条)...")
        rows = evaluate(new_benches[bucket], mapper, spec, spec.calib,
                        with_tket=(bucket != "large"))
        with open(out_dir / f"{bucket}_rows.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
        sections.append(bucket_table(rows, f"新生成层：{bucket}"))

    sections.append(reslice_existing())
    (out_dir / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"\n报告 -> {out_dir}/report.md")


if __name__ == "__main__":
    main()
