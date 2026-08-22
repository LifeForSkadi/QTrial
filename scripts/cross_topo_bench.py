"""跨拓扑验证：6×6 网格 / heavy-hex-115 / Sycamore-53 上的噪声感知对比。

模型 = multi7topo 检查点（7 拓扑混合训练，跨机器泛化专用）；
编译器：QTrial fid 规则、噪声感知 O1/O3、盲目 O1、pytket。
线路：MQTBench ≤10 比特 15 条 + QUEKO dense (3,4) 10 条 + MQTBench 25 比特 6 条。
每拓扑使用自身合成校准（同种子缺陷模型）。
输出：tables/cross_topo/report.md
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices.architectures import (build_grid_family_spec,
                                          build_heavyhex_spec,
                                          build_sycamore53_spec)
from qtrail.models import QAPolicy
from qtrail.pipeline.external import tket_compile
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements, load_qasm2

OUT = Path("tables/cross_topo")
CACHE = Path("data/mqtbench/stratified")
CKPT = "checkpoints/multi7topo_gat_combined_calib_dep0.1_t0.5_best.pt"
TRIM = 0.1


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


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * TRIM))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def main():
    cfg = Config()
    policy, ckpt = QAPolicy.load_checkpoint(
        CKPT, device_n=105,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    mqt_small = [(p.stem, qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8"))))
                 for p in sorted(CACHE.glob("*.qasm"))
                 if int(p.stem.rsplit("_", 1)[1]) <= 10]
    queko = [(p.stem, load_qasm2(p)) for p in iter_queko_files("BIGD")
             if re.search(r"\.3D1_\.4D2_", p.name)]
    mqt_25 = [(p.stem, qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8"))))
              for p in sorted(CACHE.glob("*.qasm"))
              if p.stem.endswith("_25") and p.stem != "qwalk_25"]
    circuits_all = mqt_small + queko + mqt_25
    print(f"{len(mqt_small)} small + {len(queko)} queko + {len(mqt_25)} n25 "
          f"= {len(circuits_all)} circuits", flush=True)

    topologies = {
        "grid-6x6": build_grid_family_spec(6, 6, seed=0),
        "heavy-hex-115": build_heavyhex_spec(distance=7, seed=0),
        "sycamore-53": build_sycamore53_spec(seed=0),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    sections = ["# 跨拓扑验证：QTrial（multi7topo 模型）vs 噪声感知 O1/O3（统一校准模型）\n",
                "三拓扑各自合成校准（同缺陷种子）；统计：均值/截尾(去10%)/中位数\n"]

    for tname, spec in topologies.items():
        target = build_target(spec)
        calib = spec.calib
        cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
              if spec.adj[i, j]]
        rows = []
        for name, qc in circuits_all:
            qc, _ = strip_measurements(qc)
            if qc.num_qubits > spec.n:
                continue
            row = {"circuit": name, "n": qc.num_qubits}
            t0 = time.time()
            try:
                mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule="fidelity")
                res = mapper.map_circuit(qc, circuit_id=name)
                row["os_swaps"] = res.swap_count
                row["os_depth"] = res.metrics["depth"]
                row["os_twoq_depth"] = res.metrics["twoq_depth"]
                row["os_fidelity"] = res.metrics["est_fidelity"]
                row["os_mean2q"] = res.metrics.get("mean_2q_err")
                row["os_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["os_error"] = f"{type(e).__name__}: {str(e)[:50]}"
            for opt, prefix in ((1, "aw_o1"), (3, "aw_o3"), (1, "blind")):
                t0 = time.time()
                try:
                    kw = {"target": target} if prefix != "blind" else {"coupling_map": cm}
                    out_qc = transpile(qc, basis_gates=["rz", "sx", "x", "cz"],
                                       optimization_level=opt,
                                       seed_transpiler=42, **kw)
                    m = compute_metrics(out_qc, out_qc.count_ops().get("swap", 0),
                                        calib)
                    row[f"{prefix}_swaps"] = m["swap_count"]
                    row[f"{prefix}_depth"] = m["depth"]
                    row[f"{prefix}_twoq_depth"] = m["twoq_depth"]
                    row[f"{prefix}_fidelity"] = m["est_fidelity"]
                    row[f"{prefix}_mean2q"] = m.get("mean_2q_err")
                    row[f"{prefix}_wall"] = round(time.time() - t0, 2)
                except Exception as e:
                    row[f"{prefix}_error"] = f"{type(e).__name__}: {str(e)[:50]}"
            t0 = time.time()
            try:
                expanded, sc = tket_compile(qc, cm, spec.n)
                expanded = decompose_to_platform(expanded, Mapper(spec).cm,
                                                 optimization_level=1, seed=42)
                m = compute_metrics(expanded, sc, calib)
                row["tket_swaps"] = sc
                row["tket_depth"] = m["depth"]
                row["tket_fidelity"] = m["est_fidelity"]
                row["tket_mean2q"] = m.get("mean_2q_err")
                row["tket_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["tket_error"] = f"{type(e).__name__}: {str(e)[:50]}"
            rows.append(row)
            with open(OUT / f"{tname}_rows.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
            print(f"  [{tname}] {name:32s} os {row.get('os_swaps', '-')}/"
                  f"{row.get('os_fidelity', '-')} | aw_o3 "
                  f"{row.get('aw_o3_swaps', '-')}/{row.get('aw_o3_fidelity', '-')}",
                  flush=True)

        lines = [f"\n## {tname}（{spec.n} 比特，{len(rows)} 条线路）\n",
                 "| 指标 | QTrial fid | 感知O1 | 感知O3 | 盲目O1 | pytket |",
                 "|---|---|---|---|---|---|"]
        for metric, name in (("swaps", "SWAP"), ("depth", "深度"),
                             ("twoq_depth", "2Q 深度"),
                             ("fidelity", "保真度"), ("mean2q", "2Q 误差加权均值")):
            cells = []
            for prefix, fmt in (("os", ".2f"), ("aw_o1", ".2f"),
                                ("aw_o3", ".2f"), ("blind", ".2f"),
                                ("tket", ".2f")):
                vals = [r.get(f"{prefix}_{metric}") for r in rows
                        if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
                if not vals:
                    cells.append("-")
                elif metric in ("fidelity", "mean2q"):
                    cells.append(f"{statistics.mean(vals):.4f}/{trimmed(vals):.4f}")
                else:
                    cells.append(f"{statistics.mean(vals):.1f}/{trimmed(vals):.1f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        ok = [r for r in rows if "os_fidelity" in r and "aw_o3_fidelity" in r]
        w = sum(1 for r in ok if r["os_fidelity"] > r["aw_o3_fidelity"])
        wb = sum(1 for r in ok if r["os_fidelity"] > r["blind_fidelity"])
        lines.append(f"\nQTrial fid 保真度：vs 感知O3 **{w}/{len(ok)} 胜**、"
                     f"vs 盲目O1 **{wb}/{len(ok)} 胜**")
        sections.append("\n".join(lines))
        print(f"[cross_topo] {tname} done", flush=True)
    (OUT / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
