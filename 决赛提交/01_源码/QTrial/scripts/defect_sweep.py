"""缺陷强度敏感性扫描：保真度优势随热区波动幅度的变化。

缺陷倍数档位：0（无缺陷，仅基线截断正态波动）/ 2 / 3 / 5（= 默认 5-10× 的
下沿）。每档重建校准（同种子同缺陷位置，只缩放误差倍数），重测：
QTrial fid 规则、噪声感知 O1/O3、盲目 O1。
线路：MQTBench ≤10 比特 15 条 + QUEKO dense (3,4) 族 10 条。
输出：tables/defect_sweep/report.md
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
from qtrail.devices import build_tianyan287_spec
from qtrail.devices.calibration import generate_synthetic_calibration
from qtrail.models import QAPolicy
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements, load_qasm2

OUT = Path("tables/defect_sweep")
CACHE = Path("data/mqtbench/stratified")
CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
FACTORS = (0, 2, 3, 5)
TRIM = 0.1


def make_spec_with_factor(dev_cfg, factor):
    """按缺陷倍数重建校准与 DeviceSpec（同种子 → 缺陷位置一致，仅缩放）。"""
    from qtrail.devices import tianyan287 as t287
    n = dev_cfg.rows * dev_cfg.cols
    edges = t287._full_grid_edges(dev_cfg.rows, dev_cfg.cols,
                                  set(dev_cfg.absent_couplers),
                                  set(dev_cfg.disabled_qubits))
    if factor == 0:
        calib = generate_synthetic_calibration(n, edges, seed=dev_cfg.calibration_seed,
                                               correlated_defects=False)
    else:
        calib = generate_synthetic_calibration(n, edges, seed=dev_cfg.calibration_seed,
                                               correlated_defects=True,
                                               defect_factor=float(factor))
    return build_tianyan287_spec(dev_cfg, calib=calib)


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
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    policy, ckpt = QAPolicy.load_checkpoint(
        CKPT, device_n=dev_cfg.rows * dev_cfg.cols,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    mqt = [(p.stem, qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8"))))
           for p in sorted(CACHE.glob("*.qasm"))
           if int(p.stem.rsplit("_", 1)[1]) <= 10]
    queko = [(p.stem, load_qasm2(p)) for p in iter_queko_files("BIGD")
             if re.search(r"\.3D1_\.4D2_", p.name)]
    circuits_all = mqt + queko
    print(f"{len(mqt)} MQTBench-small + {len(queko)} QUEKO-dense", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for factor in FACTORS:
        spec = make_spec_with_factor(dev_cfg, factor)
        target = build_target(spec)
        calib = spec.calib
        cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
              if spec.adj[i, j]]
        factor_rows = []
        for name, qc in circuits_all:
            qc, _ = strip_measurements(qc)
            row = {"circuit": name, "n": qc.num_qubits}
            t0 = time.time()
            try:
                mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule="fidelity")
                res = mapper.map_circuit(qc, circuit_id=name)
                row["os_swaps"] = res.swap_count
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
                    row[f"{prefix}_fidelity"] = m["est_fidelity"]
                    row[f"{prefix}_mean2q"] = m.get("mean_2q_err")
                    row[f"{prefix}_wall"] = round(time.time() - t0, 2)
                except Exception as e:
                    row[f"{prefix}_error"] = f"{type(e).__name__}: {str(e)[:50]}"
            factor_rows.append(row)
            with open(OUT / f"f{factor}_rows.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
        rows[factor] = factor_rows
        print(f"[defect_sweep] factor={factor} done ({len(factor_rows)} circuits)",
              flush=True)

    # 汇总
    sections = ["# 缺陷强度敏感性扫描（天衍-287，MQTBench≤10 15 条 + QUEKO-dense 10 条）\n",
                "同种子同缺陷位置，仅缩放误差倍数；QTrial fid 规则 / 噪声感知 O1/O3 / "
                "盲目 O1；统计：均值/截尾(去10%)/中位数\n"]
    for factor in FACTORS:
        fr = rows[factor]
        lines = [f"\n## 缺陷倍数 ×{factor}\n",
                 "| 指标 | QTrial fid | 感知O1 | 感知O3 | 盲目O1 |", "|---|---|---|---|---|"]
        for metric, name in (("swaps", "SWAP"), ("fidelity", "保真度"),
                             ("mean2q", "2Q 误差加权均值")):
            cells = []
            for prefix in ("os", "aw_o1", "aw_o3", "blind"):
                vals = [r.get(f"{prefix}_{metric}") for r in fr
                        if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
                if not vals:
                    cells.append("-")
                elif metric == "swaps":
                    cells.append(f"{statistics.mean(vals):.1f}/{trimmed(vals):.1f}")
                else:
                    cells.append(f"{statistics.mean(vals):.4f}/{trimmed(vals):.4f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        ok = [r for r in fr if "os_fidelity" in r and "aw_o3_fidelity" in r]
        w = sum(1 for r in ok if r["os_fidelity"] > r["aw_o3_fidelity"])
        lines.append(f"\nQTrial fid 保真度 vs 感知O3：**{w}/{len(ok)} 胜**")
        sections.append("\n".join(lines))
    (OUT / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
