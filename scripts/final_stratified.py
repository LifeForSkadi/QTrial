"""奖励重训闭环后的最终分层对比（验收门禁通过后运行）。

分层：≤10（15）/ 11-50（14）/ 51-105（5）。
基线：基本 O1 / 基本 O3 / 感知 O1 / 感知 O3 / pytket。
QTrial = 新模型（postfid_ft）+ target_post opt=3 + 8 种子 × top-3，fidelity 规则。
Cqlib 对比（可行集 ≤25 比特，31 条）：cqlib 原生双目标取优、
QTrial 新布局 + cqlib 路由、QTrial 完整管线——600s 硬超时如实记录。

用法: python scripts/final_stratified.py [--main-only] [--cqlib-only]
输出: tables/final_stratified/
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
from qtrail.pipeline.cqlib_route import cqlib_route
from qtrail.pipeline.external import tket_compile
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements, load_qasm2

OUT = Path("tables/final_stratified")
CACHE = Path("data/mqtbench/stratified")
NEW_CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
            "postfid_ft_best.pt")
TRIM = 0.1
LARGE_PICK = {"qft_105", "qaoa_105", "dj_105", "wstate_105", "ghz_105"}


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


def groups():
    small, med, large, feasible = [], [], [], []
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        qc = qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8")))
        if size <= 10:
            small.append((name, qc))
            feasible.append((name, qc))
        elif size <= 25:
            feasible.append((name, qc))
        if 11 <= size <= 50:
            med.append((name, qc))
        elif size > 50 and name in LARGE_PICK:
            large.append((name, qc))
    for p in iter_queko_files("BIGD"):
        if ".3D1_.4D2_" in p.name:
            feasible.append((p.stem, load_qasm2(p)))
    return {"small": small, "med": med, "large": large}, feasible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--cqlib-only", action="store_true")
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
        NEW_CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets, feasible = groups()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.cqlib_only:
        run_main(buckets, policy, cfg, spec, target, calib, cm)
    if not args.main_only:
        run_cqlib(feasible, policy, cfg, spec, calib)


def run_main(buckets, policy, cfg, spec, target, calib, cm):
    sections = ["# 最终分层对比（奖励重训闭环后，天衍-287）\n",
                "QTrial = postfid_ft 模型 + target_post opt=3 + 8 种子×top-3 "
                "（fidelity 规则）；基线 = 基本 O1/O3、感知 O1/O3、pytket；"
                "统计：均值/截尾(去10%)/中位数\n"]
    cols = [("QTrial", "qt"), ("基本O1", "b_o1"), ("基本O3", "b_o3"),
            ("感知O1", "aw_o1"), ("感知O3", "aw_o3"), ("pytket", "tket")]
    for bname, circuits in buckets.items():
        rows = []
        for name, qc in circuits:
            qc, _ = strip_measurements(qc)
            row = {"circuit": name, "n": qc.num_qubits}
            t0 = time.time()
            try:
                mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                seed=42, selection_rule="fidelity",
                                target_post=True, target_post_opt=3,
                                target_post_seeds=8,
                                target_post_top_per_seed=3)
                res = mapper.map_circuit(qc, circuit_id=name)
                row["qt_swaps"] = res.swap_count
                row["qt_depth"] = res.metrics["depth"]
                row["qt_twoq_depth"] = res.metrics["twoq_depth"]
                row["qt_fidelity"] = res.metrics["est_fidelity"]
                row["qt_mean2q"] = res.metrics.get("mean_2q_err")
                row["qt_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["qt_error"] = f"{type(e).__name__}: {str(e)[:50]}"
            for opt, prefix, aware in ((1, "b_o1", False), (3, "b_o3", False),
                                       (1, "aw_o1", True), (3, "aw_o3", True)):
                t0 = time.time()
                try:
                    kw = {"target": target} if aware else {"coupling_map": cm}
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
                    row[f"{prefix}_error"] = f"{type(e).__name__}: {str(e)[:50]}"
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
                row["tket_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["tket_error"] = f"{type(e).__name__}: {str(e)[:50]}"
            rows.append(row)
            with open(OUT / f"main_{bname}_rows.jsonl", "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
            print(f"  [{bname}] {name:32s} qt {row.get('qt_swaps', '-')}/"
                  f"{row.get('qt_fidelity', '-')} | aw_o3 "
                  f"{row.get('aw_o3_swaps', '-')}/{row.get('aw_o3_fidelity', '-')}",
                  flush=True)
        lines = [f"\n## {bname}（{len(rows)} 条线路）\n",
                 "| 指标 | " + " | ".join(c for c, _ in cols) + " |",
                 "|---|---|---|---|---|---|---|"]
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
        ok = [r for r in rows if "qt_fidelity" in r and "aw_o3_fidelity" in r]
        w3 = sum(1 for r in ok if r["qt_fidelity"] > r["aw_o3_fidelity"])
        wt = sum(1 for r in ok if r["qt_fidelity"] > r.get("tket_fidelity", 0))
        lines.append(f"\nQTrial 保真度：vs 感知O3 **{w3}/{len(ok)} 胜**、"
                     f"vs pytket **{wt}/{len(ok)} 胜**")
        sections.append("\n".join(lines))
        print(f"[final] main {bname} done", flush=True)
    (OUT / "report_main.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / 'report_main.md'}")


def run_cqlib(feasible, policy, cfg, spec, calib):
    sections = ["# Cqlib 可行集对比（≤25 比特，天衍-287）\n",
                "QTrial = postfid_ft + target_post；QTrial+Cqlib = 新布局注入平台"
                "原生 MCTS；cqlib 原生 = size/depth 双目标取优；600s 硬超时如实记录"
                "；统计：均值/截尾(去10%)/中位数\n"]
    rows = []
    for name, qc in feasible:
        qc, _ = strip_measurements(qc)
        row = {"circuit": name, "n": qc.num_qubits}
        t0 = time.time()
        try:
            mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=42,
                            selection_rule="fidelity", target_post=True,
                            target_post_opt=3, target_post_seeds=8,
                            target_post_top_per_seed=3)
            res = mapper.map_circuit(qc, circuit_id=name)
            row["qt_swaps"] = res.swap_count
            row["qt_depth"] = res.metrics["depth"]
            row["qt_fidelity"] = res.metrics["est_fidelity"]
            row["qt_wall"] = round(time.time() - t0, 2)
        except Exception as e:
            row["qt_error"] = f"{type(e).__name__}: {str(e)[:50]}"
        # cqlib 原生（双目标，600s 硬超时）
        for obj in ("size", "depth"):
            t0 = time.time()
            try:
                qc_back, swaps, _fm = cqlib_route(qc, spec, layout=None,
                                                  objective=obj, seed=42,
                                                  timeout_guard=600)
                from qtrail.pipeline.mapper import Mapper as _M
                final = decompose_to_platform(qc_back, _M(spec).cm,
                                              optimization_level=1, seed=42)
                m = compute_metrics(final, swaps, calib)
                row[f"cn_{obj}_swaps"] = m["swap_count"]
                row[f"cn_{obj}_depth"] = m["depth"]
                row[f"cn_{obj}_fidelity"] = m["est_fidelity"]
                row[f"cn_{obj}_wall"] = round(time.time() - t0, 2)
            except TimeoutError:
                row[f"cn_{obj}_error"] = "timeout(600s)"
            except Exception as e:
                row[f"cn_{obj}_error"] = f"{type(e).__name__}: {str(e)[:50]}"
        # QTrial 新布局 + cqlib 路由
        t0 = time.time()
        try:
            mapper_c = Mapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=42,
                              selection_rule="fidelity", routing_method="cqlib",
                              cqlib_objective="depth", cqlib_timeout=600)
            res_c = mapper_c.map_circuit(qc, circuit_id=name)
            row["qc_swaps"] = res_c.swap_count
            row["qc_depth"] = res_c.metrics["depth"]
            row["qc_fidelity"] = res_c.metrics["est_fidelity"]
            row["qc_wall"] = round(time.time() - t0, 2)
        except TimeoutError:
            row["qc_error"] = "timeout(600s)"
        except Exception as e:
            row["qc_error"] = f"{type(e).__name__}: {str(e)[:50]}"
        rows.append(row)
        with open(OUT / "cqlib_rows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            f.flush()
        print(f"  {name:36s} qt {row.get('qt_swaps', '-')}/{row.get('qt_fidelity', '-')} | "
              f"cn_size {row.get('cn_size_swaps', row.get('cn_size_error', '-'))} | "
              f"cn_depth {row.get('cn_depth_swaps', row.get('cn_depth_error', '-'))} | "
              f"qc {row.get('qc_swaps', row.get('qc_error', '-'))}", flush=True)

    cols = [("QTrial", "qt"), ("cqlib原生最优", "cn_best"),
            ("QTrial+Cqlib", "qc")]
    lines = ["\n## 可行集（" + str(len(rows)) + " 条线路）\n",
             "| 指标 | " + " | ".join(c for c, _ in cols) + " |",
             "|---|---|---|---|"]
    for metric, name, nd in (("swaps", "SWAP", 2), ("depth", "深度", 2),
                             ("fidelity", "保真度", 4)):
        cells = []
        for _, prefix in cols:
            if prefix == "cn_best":
                vals = []
                for r in rows:
                    a = r.get("cn_size_" + metric)
                    b = r.get("cn_depth_" + metric)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        vals.append(min(a, b) if metric != "fidelity" else max(a, b))
                s = (statistics.mean(vals), trimmed(vals)) if vals else None
            else:
                vals = [r.get(f"{prefix}_{metric}") for r in rows
                        if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
                s = (statistics.mean(vals), trimmed(vals)) if vals else None
            if s is None:
                cells.append("-")
            elif nd == 2:
                cells.append(f"{s[0]:.1f}/{s[1]:.1f}")
            else:
                cells.append(f"{s[0]:.4f}/{s[1]:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    ok = [r for r in rows if "qt_fidelity" in r and "qc_fidelity" in r]
    wq = sum(1 for r in ok if r["qt_fidelity"] > r["qc_fidelity"])
    timeout_n = sum(1 for r in rows if "cn_size_error" in r or "cn_depth_error" in r)
    lines.append(f"\nQTrial 保真度 vs QTrial+Cqlib：**{wq}/{len(ok)} 胜**；"
                 f"cqlib 原生超时 **{timeout_n}/{len(rows)}** 条")
    sections.append("\n".join(lines))
    (OUT / "report_cqlib.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / 'report_cqlib.md'}")


if __name__ == "__main__":
    main()
