"""GOAT 全编译器分层对比 v3：纯研究管线 × 三种决胜规则 × 三层规模 + 截尾均值。

QTrial 研究管线（默认）：RL 多起点布局 + 自适应局部搜索 + 路由感知候选择优
（仅 RL 候选），SabreSwap 仅作路由器——无任何外部编译器结果采纳。
（--ensemble 可开启 SABRE/O3/tket 采纳的工程增强，见 map CLI。）

决胜规则：swap（SWAP 优先）/ fidelity（保真度优先）/ depth（深度优先）。
分层：≤10 / 11-50 / 51-105（含满占，pytket 作为对比基线全层纳入）。
统计：均值 / 截尾均值（去掉高低各 10% 极端值）/ 中位数。

用法: python scripts/goat_stratified.py [--rules swap,fidelity,depth] [--skip-large]
输出: tables/goat_stratified_pure/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import sabre_swap_count, sabre_transpile
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.external import tket_compile
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements, load_qasm2
from qtrail.utils.bench import iter_queko_files

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
CACHE = Path("data/mqtbench/stratified")
TRIM = 0.1  # 高低各去掉 10%


def build_circuits() -> dict:
    from qiskit import qasm2
    out = {"small": [], "medium": [], "large": []}
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        qc = qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8")))
        if size <= 10:
            out["small"].append((name, qc))
        elif size <= 50:
            out["medium"].append((name, qc))
        else:
            out["large"].append((name, qc))
    # 11-50 层补充 QUEKO BIGD 密度样本
    import re
    seen = set()
    for p in iter_queko_files("BIGD"):
        m = re.search(r"\.(\d+)D1_\.(\d+)D2_(\d)", p.name)
        if m and m.group(3) == "0":
            key = m.group(1)
            if key not in seen:
                seen.add(key)
                out["medium"].append((f"queko_{p.stem}", load_qasm2(p)))
    return out


def qmap_compile(qc, cm_edges, n_phys, calib):
    import re
    from qiskit import qasm2
    import sysconfig as _sc
    USER = r"C:\Users\SKADI\AppData\Roaming\Python\Python313\site-packages"
    _orig = _sc.get_paths

    def _patched(*a, **k):
        p = _orig(*a, **k)
        p["purelib"] = USER
        return p
    if _sc.get_paths is _orig:
        _sc.get_paths = _patched
    from mqt.core.ir import QuantumComputation
    from mqt.qmap.sc import Architecture, Method, map_, Configuration
    qcomp = QuantumComputation.from_qasm_str(qasm2.dumps(qc))
    cfg = Configuration()
    cfg.method = Method.heuristic
    mapped, _ = map_(qcomp, Architecture(n_phys, cm_edges), cfg)
    qasm_out = mapped.qasm2_str()
    if "qelib1.inc" not in qasm_out:
        qasm_out = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + qasm_out.split(
            "OPENQASM 2.0;", 1)[-1]
    n_swap = len(re.findall(r"swap q\[\d+\], q\[\d+\];", qasm_out))
    qasm_out = re.sub(r"swap q\[(\d+)\], q\[(\d+)\];",
                      r"cx q[\1], q[\2]; cx q[\2], q[\1]; cx q[\1], q[\2];", qasm_out)
    qasm_out = re.sub(r"\bp\(", "u1(", qasm_out)
    return compute_metrics(qasm2.loads(qasm_out), n_swap, calib)


def evaluate_bucket(circuits, mapper, spec, calib, with_qmap: bool,
                    with_tket: bool):
    rows = []
    for name, qc in circuits:
        qc_clean, _ = strip_measurements(qc)
        row = {"circuit": name, "n": qc_clean.num_qubits}
        # QTrial（混合竞技，规则由 mapper 决定）
        t0 = __import__("time").time()
        try:
            res = mapper.map_circuit(qc_clean, circuit_id=name)
            row.update({"swaps": res.swap_count, "depth": res.metrics["depth"],
                        "twoq": res.metrics["twoq_count"],
                        "fidelity": res.metrics["est_fidelity"],
                        "method": res.method,
                        "wall": round(__import__("time").time() - t0, 2)})
        except Exception as e:
            row["error"] = str(e)[:80]
        # SABRE O1/O3
        for opt in (1, 3):
            t0 = __import__("time").time()
            try:
                sc = res.baseline_swaps if opt == 1 and "error" not in row else None
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
                row[f"o{opt}_wall"] = round(__import__("time").time() - t0, 2)
            except Exception as e:
                row[f"o{opt}_error"] = str(e)[:60]
        # pytket（全层纳入）
        if with_tket:
            t0 = __import__("time").time()
            try:
                edges = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
                         if spec.adj[i, j]]
                expanded, sc = tket_compile(qc_clean, edges, spec.n)
                m = compute_metrics(expanded, sc, calib)
                row["tket_swaps"] = sc
                row["tket_depth"] = m["depth"]
                row["tket_fidelity"] = m["est_fidelity"]
                row["tket_wall"] = round(__import__("time").time() - t0, 2)
            except Exception as e:
                row["tket_error"] = str(e)[:60]
        # QMAP（≤50 比特）
        if with_qmap and qc_clean.num_qubits <= 50:
            t0 = __import__("time").time()
            try:
                edges = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
                         if spec.adj[i, j]]
                m = qmap_compile(qc_clean, edges, spec.n, calib)
                row["qmap_swaps"] = m["swap_count"]
                row["qmap_depth"] = m["depth"]
                row["qmap_fidelity"] = m["est_fidelity"]
                row["qmap_wall"] = round(__import__("time").time() - t0, 2)
            except Exception as e:
                row["qmap_error"] = str(e)[:60]
        rows.append(row)
    return rows


def trimmed_mean(vals, trim=TRIM):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * trim))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def summarize(rows: list) -> dict:
    def stat(key):
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
        if not vals:
            return None
        return (round(statistics.mean(vals), 2),
                round(trimmed_mean(vals), 2),
                round(statistics.median(vals), 2))

    s = {"n": len(rows)}
    for c in ("swaps", "depth", "twoq", "fidelity", "wall", "o1_swaps", "o1_depth",
              "o1_fidelity", "o1_wall", "o3_swaps", "o3_depth", "o3_fidelity",
              "o3_wall", "tket_swaps", "tket_depth", "tket_fidelity", "tket_wall",
              "qmap_swaps", "qmap_depth", "qmap_fidelity", "qmap_wall"):
        s[c] = stat(c)
    ok = [r for r in rows if "error" not in r and "swaps" in r]
    if ok:
        s["qtrial_wins_vs_o1"] = sum(1 for r in ok if r["swaps"] < r["o1_swaps"])
        s["qtrial_wins_vs_o3"] = sum(1 for r in ok if r["swaps"] < r["o3_swaps"])
        s["n_ok"] = len(ok)
    return s


def fmt3(v):
    """均值/截尾/中位"""
    return f"{v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" if v else "-"


def render_rule(sections: list, title: str, rows: list, rule: str):
    lines = [f"\n## {title} · 规则={rule}\n",
             "| 指标（均值/截尾/中位） | QTrial | SABRE O1 | SABRE O3 | pytket | QMAP |",
             "|---|---|---|---|---|---|"]
    s = summarize(rows)
    lines.append(f"| SWAP 数 | {fmt3(s['swaps'])} | {fmt3(s['o1_swaps'])} | "
                 f"{fmt3(s['o3_swaps'])} | {fmt3(s['tket_swaps'])} | {fmt3(s['qmap_swaps'])} |")
    lines.append(f"| 深度 | {fmt3(s['depth'])} | {fmt3(s['o1_depth'])} | "
                 f"{fmt3(s['o3_depth'])} | {fmt3(s['tket_depth'])} | {fmt3(s['qmap_depth'])} |")
    fids = lambda v: f"{v[0]:.4f}" if v else "-"
    lines.append(f"| 估计保真度（均值） | {fids(s['fidelity'])} | {fids(s['o1_fidelity'])} | "
                 f"{fids(s['o3_fidelity'])} | {fids(s['tket_fidelity'])} | {fids(s['qmap_fidelity'])} |")
    lines.append(f"| 编译耗时（均值, s） | {s['wall'][0] if s['wall'] else '-'} | "
                 f"{s['o1_wall'][0] if s['o1_wall'] else '-'} | {s['o3_wall'][0] if s['o3_wall'] else '-'} | "
                 f"{s['tket_wall'][0] if s['tket_wall'] else '-'} | {s['qmap_wall'][0] if s['qmap_wall'] else '-'} |")
    if s.get("n_ok"):
        lines.append(f"\nQTrial SWAP 胜场：vs O1 **{s['qtrial_wins_vs_o1']}/{s['n_ok']}**，"
                     f"vs O3 **{s['qtrial_wins_vs_o3']}/{s['n_ok']}**（平局计入不败）\n")
    sections.append("\n".join(lines))


def render_cross_rule(sections: list, bucket: str, title: str,
                      rule_results: dict):
    """三规则横向对比（截尾均值）。"""
    lines = [f"\n## {title} · 三规则横向对比（截尾均值）\n",
             "| 规则 | SWAP | 深度 | 保真度 | vs O1 胜场 | vs O3 胜场 |",
             "|---|---|---|---|---|---|"]
    for rule, rows in rule_results.items():
        s = summarize(rows)
        sw, de, fi = s.get("swaps"), s.get("depth"), s.get("fidelity")
        lines.append(f"| {rule} | {sw[1]:.2f} | {de[1]:.2f} | {fi[0]:.4f} | "
                     f"{s.get('qtrial_wins_vs_o1', 0)}/{s.get('n_ok', 0)} | "
                     f"{s.get('qtrial_wins_vs_o3', 0)}/{s.get('n_ok', 0)} |")
    lines.append("")
    sections.append("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="swap,fidelity,depth")
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()
    rules = [r.strip() for r in args.rules.split(",")]

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    import torch
    policy, ckpt = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                            map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    circuits = build_circuits()
    if args.skip_large:
        circuits.pop("large", None)

    out_dir = Path("tables/goat_stratified_pure")
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = ["# GOAT 全编译器分层对比 v2（天衍-287，105 比特 15×7 网格）\n",
                "QTrial = 稀疏奖励 RL（v2）+ 混合路由竞技；三种决胜规则 × 三层规模；"
                "统计口径：均值 / 截尾均值(去高低各10%) / 中位数；pytket 全层纳入\n"]

    bucket_titles = {"small": "≤10 比特", "medium": "11-50 比特",
                     "large": "51-105 比特（含满占）"}
    rule_results = {b: {} for b in circuits}

    for bucket in circuits:
        for rule in rules:
            print(f"评测 {bucket} 层 · 规则 {rule}（{len(circuits[bucket])} 条）...",
                  flush=True)
            mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=42,
                            selection_rule=rule)
            rows = evaluate_bucket(circuits[bucket], mapper, spec, spec.calib,
                                   with_qmap=(bucket != "large"), with_tket=True)
            with open(out_dir / f"{bucket}_{rule}_rows.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
            rule_results[bucket][rule] = rows
            render_rule(sections, bucket_titles[bucket], rows, rule)

        render_cross_rule(sections, bucket, bucket_titles[bucket],
                          rule_results[bucket])

    (out_dir / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"\n报告 -> {out_dir}/report.md")


if __name__ == "__main__":
    main()
