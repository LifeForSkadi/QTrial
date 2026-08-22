"""QUEKO 高密度线路上的噪声感知必要性实验（决定性实验）。

问题：MQTBench 线路全部可无 SWAP 嵌入，噪声感知 O3 的 VF2 试验占尽优势；
QUEKO 高密度线路不可嵌入（完美嵌入下界几十个 SWAP），该分布上 VF2 的
0-SWAP 神话消失——RL 布局 + 路由感知选择是否仍具必要性？

分组：
  dense  = (D1,D2) ∈ {(3,4),(3,5),(4,4)}（30 条，不可无 SWAP 嵌入）
  sparse = (0,1)（10 条，对照组，接近可嵌入）
编译器：QTrial fid/swap 规则、噪声感知 O1/O3、盲目 O1、pytket。
输出：tables/queko_noise/report.md + rows jsonl
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import transpile
from qiskit.circuit.library import CZGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionProperties, Target
import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.external import tket_compile
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import strip_measurements, load_qasm2

OUT = Path("tables/queko_noise")
CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
DENSE_FAMILIES = {(3, 4), (3, 5), (4, 4)}
SPARSE_FAMILIES = {(0, 1)}
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


def pick_circuits():
    dense, sparse = [], []
    for p in iter_queko_files("BIGD"):
        m = re.search(r"\.(\d)D1_\.(\d)D2_", p.name)
        if not m:
            continue
        fam = (int(m.group(1)), int(m.group(2)))
        if fam in DENSE_FAMILIES:
            dense.append(p)
        elif fam in SPARSE_FAMILIES:
            sparse.append(p)
    return dense, sparse


def trimmed_mean(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * TRIM))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def fmt3(v):
    return f"{v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" if v else "-"


def render(sections, title, rows):
    cols = [("QTrial fid", "os_fid"), ("QTrial swap", "os_swap"),
            ("感知O1", "aw_o1"), ("感知O3", "aw_o3"),
            ("盲目O1", "blind_o1"), ("pytket", "tket")]
    lines = [f"\n## {title}（{len(rows)} 条线路）\n",
             "| 指标（均值/截尾/中位） | " + " | ".join(c for c, _ in cols) + " |",
             "|---|---|---|---|---|---|---|"]

    def stat(prefix, metric):
        vals = [r.get(f"{prefix}_{metric}") for r in rows
                if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
        if not vals:
            return None
        return (round(statistics.mean(vals), 4),
                round(trimmed_mean(vals), 4),
                round(statistics.median(vals), 4))

    for metric, name, nd in (("swaps", "SWAP 数", 2), ("depth", "深度", 2),
                             ("twoq_depth", "2Q 深度", 2),
                             ("fidelity", "估计保真度", 4),
                             ("mean2q", "2Q 误差加权均值", 4)):
        cells = []
        for _, prefix in cols:
            s = stat(prefix, metric)
            if s is None:
                cells.append("-")
            elif nd == 2:
                cells.append(f"{s[0]:.2f}/{s[1]:.2f}/{s[2]:.2f}")
            else:
                cells.append(f"{s[0]:.4f}/{s[1]:.4f}/{s[2]:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    ok = [r for r in rows
          if isinstance(r.get("os_fid_fidelity"), (int, float))
          and isinstance(r.get("aw_o3_fidelity"), (int, float))]
    if ok:
        w_fid = sum(1 for r in ok if r["os_fid_fidelity"] > r["aw_o3_fidelity"])
        w_swap = sum(1 for r in ok if r["os_fid_swaps"] < r["aw_o3_swaps"])
        lines.append(f"\nQTrial fid规则：保真度 vs 感知O3 **{w_fid}/{len(ok)} 胜**，"
                     f"SWAP vs 感知O3 **{w_swap}/{len(ok)} 胜**")
    lines.append("")
    sections.append("\n".join(lines))


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    calib = spec.calib
    target = build_target(spec)
    policy, ckpt = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]

    dense, sparse = pick_circuits()
    print(f"dense {len(dense)} files, sparse {len(sparse)} files", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sections = ["# QUEKO 高密度线路：噪声感知必要性决定性实验（天衍-287）\n",
                "dense = BIGD (D1,D2)∈{(3,4),(3,5),(4,4)}；sparse = (0,1) 对照；"
                "噪声感知 O1/O3 = qiskit Target 驱动 VF2PostLayout；"
                "统计：均值/截尾(去10%)/中位数\n"]

    for group, files in (("dense", dense), ("sparse", sparse)):
        rows = []
        with open(OUT / f"{group}_rows.jsonl", "w", encoding="utf-8") as f:
            for p in files:
                qc = load_qasm2(p)
                qc, _ = strip_measurements(qc)
                row = {"circuit": p.stem, "n": qc.num_qubits}
                # QTrial 两规则
                for rule, prefix in (("fidelity", "os_fid"), ("swap", "os_swap")):
                    t0 = time.time()
                    try:
                        mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                        seed=42, selection_rule=rule)
                        res = mapper.map_circuit(qc, circuit_id=p.stem)
                        row[f"{prefix}_swaps"] = res.swap_count
                        row[f"{prefix}_depth"] = res.metrics["depth"]
                        row[f"{prefix}_twoq_depth"] = res.metrics["twoq_depth"]
                        row[f"{prefix}_fidelity"] = res.metrics["est_fidelity"]
                        row[f"{prefix}_mean2q"] = res.metrics.get("mean_2q_err")
                        row[f"{prefix}_wall"] = round(time.time() - t0, 2)
                    except Exception as e:
                        row[f"{prefix}_error"] = f"{type(e).__name__}: {str(e)[:60]}"
                # 噪声感知 O1/O3 + 盲目 O1
                for opt, prefix in ((1, "aw_o1"), (3, "aw_o3"), (1, "blind_o1")):
                    t0 = time.time()
                    try:
                        kw = {"target": target} if prefix.startswith("aw") else {
                            "coupling_map": cm}
                        out_qc = transpile(qc, basis_gates=["rz", "sx", "x", "cz"],
                                           optimization_level=opt,
                                           seed_transpiler=42, **kw)
                        m = compute_metrics(out_qc,
                                            out_qc.count_ops().get("swap", 0),
                                            calib)
                        row[f"{prefix}_swaps"] = m["swap_count"]
                        row[f"{prefix}_depth"] = m["depth"]
                        row[f"{prefix}_twoq_depth"] = m["twoq_depth"]
                        row[f"{prefix}_fidelity"] = m["est_fidelity"]
                        row[f"{prefix}_mean2q"] = m.get("mean_2q_err")
                        row[f"{prefix}_wall"] = round(time.time() - t0, 2)
                    except Exception as e:
                        row[f"{prefix}_error"] = f"{type(e).__name__}: {str(e)[:60]}"
                # pytket
                t0 = time.time()
                try:
                    expanded, sc = tket_compile(qc, cm, spec.n)
                    expanded = decompose_to_platform(expanded, Mapper(spec).cm,
                                                     optimization_level=1, seed=42)
                    m = compute_metrics(expanded, sc, calib)
                    row["tket_swaps"] = sc
                    row["tket_depth"] = m["depth"]
                    row["tket_twoq_depth"] = m["twoq_depth"]
                    row["tket_fidelity"] = m["est_fidelity"]
                    row["tket_mean2q"] = m.get("mean_2q_err")
                    row["tket_wall"] = round(time.time() - t0, 2)
                except Exception as e:
                    row["tket_error"] = f"{type(e).__name__}: {str(e)[:60]}"
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
                print(f"  {p.stem:36s} fid {row.get('os_fid_swaps', '-')}/"
                      f"{row.get('os_fid_fidelity', '-')} | aw_o3 "
                      f"{row.get('aw_o3_swaps', '-')}/"
                      f"{row.get('aw_o3_fidelity', '-')}", flush=True)
        render(sections, {"dense": "高密度（不可嵌入）", "sparse": "稀疏对照（可嵌入）"}[group],
               rows)
        print(f"[queko] {group} done", flush=True)
    (OUT / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
