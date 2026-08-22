"""最终方案分层性能报告 v2（完整 Benchpress 数据集版）。

数据集：MQTBench（43）+ QUEKO dense/sparse（40）+ **完整 Benchpress 集**
（IBM Qiskit 官方评测基准，全部 ≤105 比特可解析线路，不采样；
>105 比特超出设备规模如实跳过）。
QTrial = pure 管线（零 qiskit），**仅 fidelity 规则**。
基线：盲目 O1 / 盲目 O3 / 感知 O1 / 感知 O3 / pytket（qiskit 仅用于基线）。
**全部原始运行数据逐线路落盘**（jsonl，含指标/耗时/错误）——后续排除极值
等分析无需重跑。
统计：均值/截尾(去10%)/中位数。增量落盘 + 断点续跑。
输出：tables/final_report/
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
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/final_report")
CACHE = Path("data/mqtbench/stratified")
BP = Path("data/benchpress/qasm")
RULE = "fidelity"


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


def load_circuits():
    """三组线路：(name, pure_text, qiskit_qc)。完整 benchpress ≤105 比特。"""
    from qiskit import qasm2
    buckets = {"small": [], "medium": [], "large": []}
    skipped = 0
    for p in BP.rglob("*.qasm"):
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            circ = parse_qasm(text)
            if circ.n == 0:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        if circ.n > 105:
            skipped += 1
            continue
        b = "small" if circ.n <= 10 else ("medium" if circ.n <= 50 else "large")
        name = f"bp_{p.parent.name}_{p.stem}"
        qc = None
        try:
            qc = qasm2.loads(sanitize_qasm(text))
        except Exception:
            pass  # qiskit 解析失败：QTrial 仍可跑，基线列记错误
        buckets[b].append((name, text, qc))
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size > 105:
            continue
        b = "small" if size <= 10 else ("medium" if size <= 50 else "large")
        text = sanitize_qasm(p.read_text(encoding="utf-8"))
        buckets[b].append((name, text, qasm2.loads(text)))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name) \
                or re.search(r"\.0D1_\.1D2_", p.name):
            text = sanitize_qasm(p.read_text(encoding="utf-8"))
            buckets["medium"].append((p.stem, text, qasm2.loads(text)))
    return buckets, skipped


def trimmed(vals):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * 0.1))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def fmt3(v):
    return f"{v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" if v else "-"


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    target = build_target(spec)
    calib = spec.calib
    cm = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
          if spec.adj[i, j]]
    policy, ck = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    buckets, skipped = load_circuits()
    total = sum(len(v) for v in buckets.values())
    print(f"circuits: small {len(buckets['small'])} / medium "
          f"{len(buckets['medium'])} / large {len(buckets['large'])} "
          f"(total {total}, skipped {skipped})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sections = ["# 最终方案分层性能报告 v2（天衍-287，105 比特 15×7 网格）\n",
                "QTrial = 稀疏奖励 RL 布局 + 纯自研路由/后处理管线（零 qiskit），"
                "fidelity 规则；数据集 = MQTBench + QUEKO + **完整 Benchpress 集**"
                "（IBM Qiskit 官方评测基准，≤105 比特全部纳入，>105 超出设备跳过）；"
                "基线 = 盲目 O1/O3、感知 O1/O3、pytket；统计：均值/截尾(去10%)/"
                "中位数；**全部原始数据逐线路落盘（jsonl）**\n"]

    titles = {"small": "≤10 比特", "medium": "11-50 比特", "large": "51-105 比特"}
    cols = [("QTrial", "qt"), ("感知O3", "aw_o3"), ("感知O1", "aw_o1"),
            ("盲目O3", "b_o3"), ("盲目O1", "b_o1"), ("pytket", "tket")]

    for bucket in ("small", "medium", "large"):
        rows = []
        out_file = OUT / f"{bucket}_{RULE}_rows.jsonl"
        done_names = set()
        if out_file.exists():
            for line in out_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done_names.add(r["circuit"])
                    rows.append(r)
        for name, ptext, qc in buckets[bucket]:
            if name in done_names:
                continue
            pure_n = parse_qasm(ptext).n
            row = {"circuit": name, "n": pure_n}
            if qc is None:
                row["b_o1_error"] = "qiskit parse failed"
                row["b_o3_error"] = "qiskit parse failed"
                row["aw_o1_error"] = "qiskit parse failed"
                row["aw_o3_error"] = "qiskit parse failed"
                row["tket_error"] = "qiskit parse failed"
            qc_clean = qc.copy() if qc is not None else None
            if qc_clean is not None:
                qc_clean.data = [i for i in qc_clean.data
                                 if i.operation.name not in ("measure", "barrier")]
            t0 = time.time()
            try:
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cuda",
                                    seed=42, selection_rule=RULE, use_post=True)
                res = mapper.map_circuit(parse_qasm(ptext), circuit_id=name)
                row["qt_swaps"] = res["swap_count"]
                row["qt_depth"] = res["metrics"]["depth"]
                row["qt_twoq_depth"] = res["metrics"]["twoq_depth"]
                row["qt_fidelity"] = res["metrics"]["est_fidelity"]
                row["qt_mean2q"] = res["metrics"].get("mean_2q_err")
                row["qt_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["qt_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            for opt, prefix, aware in ((1, "b_o1", False), (3, "b_o3", False),
                                       (1, "aw_o1", True), (3, "aw_o3", True)):
                if qc_clean is None:
                    continue
                t0 = time.time()
                try:
                    kw = {"target": target} if aware else {"coupling_map": cm}
                    out_qc = transpile(qc_clean,
                                       basis_gates=["rz", "sx", "x", "cz"],
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
            if qc_clean is not None:
                t0 = time.time()
            else:
                t0 = 0.0
            try:
                if qc_clean is None:
                    raise ValueError("qiskit parse failed")
                expanded, sc = tket_compile(qc_clean, cm, spec.n)
                expanded = decompose_to_platform(expanded, QiskitMapper(spec).cm,
                                                 optimization_level=1, seed=42)
                m = compute_metrics(expanded, sc, calib)
                row["tket_swaps"] = sc
                row["tket_depth"] = m["depth"]
                row["tket_twoq_depth"] = m["twoq_depth"]
                row["tket_fidelity"] = m["est_fidelity"]
                row["tket_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["tket_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            rows.append(row)
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
            print(f"  [{bucket}] {name:48s} qt {row.get('qt_swaps', '-')}/"
                  f"{row.get('qt_fidelity', '-')} wall={row.get('qt_wall', '-')}s",
                  flush=True)

        lines = [f"\n## {titles[bucket]} · 规则={RULE}（{len(rows)} 条线路）\n",
                 "| 指标（均值/截尾/中位） | " + " | ".join(c for c, _ in cols) + " |",
                 "|---|---|---|---|---|---|---|"]
        for metric, name, nd in (("swaps", "SWAP 数", 2), ("depth", "深度", 2),
                                 ("twoq_depth", "2Q 深度", 2),
                                 ("fidelity", "估计保真度", 4)):
            cells = []
            for _, prefix in cols:
                vals = [r.get(f"{prefix}_{metric}") for r in rows
                        if isinstance(r.get(f"{prefix}_{metric}"), (int, float))]
                if not vals:
                    cells.append("-")
                elif nd == 2:
                    cells.append(fmt3((statistics.mean(vals), trimmed(vals),
                                       statistics.median(vals))))
                else:
                    cells.append(f"{statistics.mean(vals):.4f}/{trimmed(vals):.4f}/"
                                 f"{statistics.median(vals):.4f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        ok = [r for r in rows if "qt_fidelity" in r and "aw_o3_fidelity" in r
              and "tket_fidelity" in r]
        w3 = sum(1 for r in ok if r["qt_fidelity"] > r["aw_o3_fidelity"])
        wt = sum(1 for r in ok if r["qt_fidelity"] > r["tket_fidelity"])
        wb = sum(1 for r in ok if r["qt_fidelity"] > r["b_o1_fidelity"])
        lines.append(f"\nQTrial 保真度胜场（{len(ok)} 条全完成）："
                     f"vs 感知O3 **{w3}**、vs pytket **{wt}**、vs 盲目O1 **{wb}**")
        sections.append("\n".join(lines))
        print(f"[final_report] {bucket} done", flush=True)
    (OUT / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
