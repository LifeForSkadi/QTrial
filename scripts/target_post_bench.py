"""Target 后处理管线全量对比：QTrial fid + target_post vs 噪声感知 O1/O3 等。

线路：MQTBench ≤10（15）+ 11-50（14）+ QUEKO dense（30）+ sparse（10）。
输出：tables/target_post/report.md（与既有 QTrial fid / 感知 O1/O3 / 盲目 O1 合并）
"""
from __future__ import annotations

import argparse
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
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements, load_qasm2

OUT = Path("tables/target_post")
CACHE = Path("data/mqtbench/stratified")
CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-opt", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--top-per-seed", type=int, default=0, help="每种子入池上限（0=全量）")
    args = ap.parse_args()
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    target = build_target(spec)
    calib = spec.calib
    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]
    policy, ckpt = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    groups = {
        "mqt-small": [(p.stem, qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8"))))
                      for p in sorted(CACHE.glob("*.qasm"))
                      if int(p.stem.rsplit("_", 1)[1]) <= 10],
        "mqt-med": [(p.stem, qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8"))))
                    for p in sorted(CACHE.glob("*.qasm"))
                    if 11 <= int(p.stem.rsplit("_", 1)[1]) <= 50],
        "queko-dense": [(p.stem, load_qasm2(p)) for p in iter_queko_files("BIGD")
                        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name)],
        "queko-sparse": [(p.stem, load_qasm2(p)) for p in iter_queko_files("BIGD")
                         if re.search(r"\.0D1_\.1D2_", p.name)],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    sections = ["# Target 后处理管线对比（天衍-287）\n",
                "QTrial fid + target_post = RL 布局 + qiskit O1 预设（噪声感知 "
                "Target）；对照组 = QTrial fid（原管线）/ 噪声感知 O1/O3 / 盲目 O1；"
                "统计：均值/截尾(去10%)/中位数\n"]
    cols = [("QTrial+post", "tp"), ("QTrial fid", "of"), ("感知O1", "aw_o1"),
            ("感知O3", "aw_o3"), ("盲目O1", "blind")]

    for gname, circuits in groups.items():
        rows = []
        for name, qc in circuits:
            qc, _ = strip_measurements(qc)
            row = {"circuit": name, "n": qc.num_qubits}
            t0 = time.time()
            try:
                mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule="fidelity",
                                target_post=True, target_post_opt=args.post_opt,
                                target_post_seeds=args.seeds,
                                target_post_top_per_seed=args.top_per_seed or None)
                res = mapper.map_circuit(qc, circuit_id=name)
                row["tp_swaps"] = res.swap_count
                row["tp_depth"] = res.metrics["depth"]
                row["tp_twoq_depth"] = res.metrics["twoq_depth"]
                row["tp_fidelity"] = res.metrics["est_fidelity"]
                row["tp_mean2q"] = res.metrics.get("mean_2q_err")
                row["tp_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["tp_error"] = f"{type(e).__name__}: {str(e)[:50]}"
            t0 = time.time()
            try:
                mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule="fidelity")
                res = mapper.map_circuit(qc, circuit_id=name)
                row["of_swaps"] = res.swap_count
                row["of_depth"] = res.metrics["depth"]
                row["of_fidelity"] = res.metrics["est_fidelity"]
                row["of_mean2q"] = res.metrics.get("mean_2q_err")
                row["of_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["of_error"] = f"{type(e).__name__}: {str(e)[:50]}"
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
            rows.append(row)
            with open(OUT / f"{gname}_rows.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
            print(f"  {name:36s} post {row.get('tp_swaps', '-')}/"
                  f"{row.get('tp_fidelity', '-')} | aw_o3 "
                  f"{row.get('aw_o3_swaps', '-')}/{row.get('aw_o3_fidelity', '-')}",
                  flush=True)

        lines = [f"\n## {gname}（{len(rows)} 条线路）\n",
                 "| 指标 | " + " | ".join(c for c, _ in cols) + " |",
                 "|---|---|---|---|---|---|"]
        for metric, name, nd in (("swaps", "SWAP", 2), ("depth", "深度", 2),
                                 ("twoq_depth", "2Q 深度", 2),
                                 ("fidelity", "保真度", 4),
                                 ("mean2q", "2Q 误差加权均值", 4)):
            cells = []
            for _, prefix in cols:
                vals = [r.get(f"{prefix}_{metric}") for r in rows
                        if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
                if not vals:
                    cells.append("-")
                elif nd == 2:
                    cells.append(f"{statistics.mean(vals):.1f}/{trimmed(vals):.1f}")
                else:
                    cells.append(f"{statistics.mean(vals):.4f}/{trimmed(vals):.4f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        ok = [r for r in rows if "tp_fidelity" in r and "aw_o3_fidelity" in r
              and "of_fidelity" in r]
        w3 = sum(1 for r in ok if r["tp_fidelity"] > r["aw_o3_fidelity"])
        wo = sum(1 for r in ok if r["tp_fidelity"] > r["of_fidelity"])
        lines.append(f"\nQTrial+post 保真度：vs 感知O3 **{w3}/{len(ok)} 胜**、"
                     f"vs 原管线 **{wo}/{len(ok)} 胜**")
        sections.append("\n".join(lines))
        print(f"[target_post] {gname} done", flush=True)
    (OUT / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / f'report_opt{args.post_opt}.md'}")


if __name__ == "__main__":
    main()
