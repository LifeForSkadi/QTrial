"""Cqlib 注入接口分层对比评测（天衍-287，105 比特 15×7 网格）。

五路对比：
  sabre        Qiskit SABRE O1（基线）
  tket         pytket DefaultMappingPass（深度最强基线）
  cqlib_size   Cqlib 原生 MCTS（objective=size，FiDLS 族初始映射）
  cqlib_depth  Cqlib 原生 MCTS（objective=depth）
  qtrial_cqlib QTrial RL 布局 + LS → 注入 cqlib MCTS（objective=depth）
  qtrial_sabre QTrial 深度规则管线（SabreSwap 后端，参考）

统一口径：所有输出经 decompose_to_platform([rz,sx,x,cz]) 后由
compute_metrics 度量（SWAP 数、深度、2Q 深度、估计保真度、耗时）。
分层：≤10 / 11-50 / 51-105（含满占）；统计：均值/截尾均值/中位数。
Cqlib MCTS 有超时保护（默认 300s，51-105 层 600s），超时如实记为 timeout。

用法: python scripts/cqlib_inject_bench.py [--small-only] [--medium-only] [--large-only]
输出: tables/cqlib_inject/
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit.transpiler import CouplingMap

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import sabre_swap_count, sabre_transpile
from qtrail.pipeline.cqlib_route import cqlib_route
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import decompose_to_platform
from qtrail.pipeline.external import tket_compile
from qtrail.utils.qasm_io import sanitize_qasm, strip_measurements

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
CACHE = Path("data/mqtbench/stratified")
TRIM = 0.1

LARGE_PICK = ["qft_105", "qaoa_105", "dj_105", "wstate_105", "ghz_105"]
COMPILERS = ["sabre", "tket", "cqlib_size", "cqlib_depth", "qtrial_cqlib",
             "qtrial_sabre"]
METRICS = ("swaps", "depth", "twoq_depth", "fidelity", "wall")


def build_circuits(bucket_filter):
    from qiskit import qasm2
    out = {"small": [], "medium": [], "large": []}
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size <= 10:
            b = "small"
        elif size <= 50:
            b = "medium"
        else:
            b = "large"
        qc = qasm2.loads(sanitize_qasm(p.read_text(encoding="utf-8")))
        out[b].append((name, qc))
    # 51-105 层按挑选名单缩减（MCTS 运行时预算）
    out["large"] = [(n, qc) for n, qc in out["large"] if n in LARGE_PICK]
    if bucket_filter:
        out = {k: v for k, v in out.items() if k in bucket_filter}
    return out


def run_cqlib(qc, spec, layout, objective, timeout_s):
    """返回 (metrics_dict, error_string)。"""
    t0 = time.time()
    try:
        qc_back, swaps, _fm = cqlib_route(qc, spec, layout=layout,
                                          objective=objective, seed=42,
                                          timeout_guard=timeout_s)
        cm = CouplingMap([[i, j] for i in range(spec.n)
                          for j in range(i + 1, spec.n) if spec.adj[i, j]])
        final = decompose_to_platform(qc_back, cm, optimization_level=1,
                                      seed=42)
        m = compute_metrics(final, swaps, spec.calib)
        m["wall"] = round(time.time() - t0, 2)
        return m, None
    except TimeoutError:
        return None, "timeout"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:70]}"


def evaluate_bucket(circuits, spec, mapper_cqlib, mapper_sabre,
                    out_file: Path | None = None):
    rows = []
    for name, qc in circuits:
        qc_clean, _ = strip_measurements(qc)
        row = {"circuit": name, "n": qc_clean.num_qubits}
        timeout_s = 600.0 if qc_clean.num_qubits > 50 else 300.0

        # SABRE O1
        t0 = time.time()
        try:
            sc, _ = sabre_swap_count(qc_clean, mapper_sabre.cm,
                                     optimization_level=1, seed=42)
            final = sabre_transpile(qc_clean, mapper_sabre.cm,
                                    optimization_level=1, seed=42)
            m = compute_metrics(final, sc, spec.calib)
            row.update({"sabre_swaps": sc, "sabre_depth": m["depth"],
                        "sabre_twoq_depth": m["twoq_depth"],
                        "sabre_fidelity": m["est_fidelity"],
                        "sabre_wall": round(time.time() - t0, 2)})
        except Exception as e:
            row["sabre_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        # pytket（统一口径：与 cqlib/sabre 同样分解到平台基 [rz,sx,x,cz]）
        t0 = time.time()
        try:
            edges = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
                     if spec.adj[i, j]]
            expanded, sc = tket_compile(qc_clean, edges, spec.n)
            expanded = decompose_to_platform(expanded, mapper_sabre.cm,
                                             optimization_level=1, seed=42)
            m = compute_metrics(expanded, sc, spec.calib)
            row.update({"tket_swaps": sc, "tket_depth": m["depth"],
                        "tket_twoq_depth": m["twoq_depth"],
                        "tket_fidelity": m["est_fidelity"],
                        "tket_wall": round(time.time() - t0, 2)})
        except Exception as e:
            row["tket_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        # Cqlib 原生（size / depth）
        for obj in ("size", "depth"):
            m, err = run_cqlib(qc_clean, spec, None, obj, timeout_s)
            if m is None:
                row[f"cqlib_{obj}_error"] = err
                continue
            row.update({f"cqlib_{obj}_swaps": m["swap_count"],
                        f"cqlib_{obj}_depth": m["depth"],
                        f"cqlib_{obj}_twoq_depth": m["twoq_depth"],
                        f"cqlib_{obj}_fidelity": m["est_fidelity"],
                        f"cqlib_{obj}_wall": m["wall"]})

        # QTrial + Cqlib 注入（RL 布局 → MCTS depth）
        t0 = time.time()
        try:
            res = mapper_cqlib.map_circuit(qc_clean, circuit_id=name)
            row.update({"qtrial_cqlib_swaps": res.swap_count,
                        "qtrial_cqlib_depth": res.metrics["depth"],
                        "qtrial_cqlib_twoq_depth": res.metrics["twoq_depth"],
                        "qtrial_cqlib_fidelity": res.metrics["est_fidelity"],
                        "qtrial_cqlib_wall": round(time.time() - t0, 2),
                        "qtrial_cqlib_method": res.method})
        except TimeoutError:
            row["qtrial_cqlib_error"] = "timeout"
        except Exception as e:
            row["qtrial_cqlib_error"] = f"{type(e).__name__}: {str(e)[:70]}"

        # QTrial + SabreSwap（深度规则，参考）
        t0 = time.time()
        try:
            res = mapper_sabre.map_circuit(qc_clean, circuit_id=name)
            row.update({"qtrial_sabre_swaps": res.swap_count,
                        "qtrial_sabre_depth": res.metrics["depth"],
                        "qtrial_sabre_twoq_depth": res.metrics["twoq_depth"],
                        "qtrial_sabre_fidelity": res.metrics["est_fidelity"],
                        "qtrial_sabre_wall": round(time.time() - t0, 2)})
        except Exception as e:
            row["qtrial_sabre_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        rows.append(row)
        if out_file is not None:  # 逐条增量落盘（防长任务中途卡死丢结果）
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                f.flush()
        print(f"  {name:32s} n={row['n']:3d} | "
              f"sabre {row.get('sabre_swaps', '-')}/{row.get('sabre_depth', '-')} | "
              f"tket {row.get('tket_swaps', '-')}/{row.get('tket_depth', '-')} | "
              f"csize {row.get('cqlib_size_swaps', '-')}/{row.get('cqlib_size_depth', '-')}"
              f"{row.get('cqlib_size_error', '')} | "
              f"cdepth {row.get('cqlib_depth_swaps', '-')}/{row.get('cqlib_depth_depth', '-')}"
              f"{row.get('cqlib_depth_error', '')} | "
              f"O+C {row.get('qtrial_cqlib_swaps', '-')}/{row.get('qtrial_cqlib_depth', '-')}"
              f"{row.get('qtrial_cqlib_error', '')} | "
              f"O+S {row.get('qtrial_sabre_swaps', '-')}/{row.get('qtrial_sabre_depth', '-')}",
              flush=True)
    return rows


def trimmed_mean(vals, trim=TRIM):
    vals = sorted(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * trim))
    if 2 * k >= len(vals):
        return statistics.mean(vals)
    return statistics.mean(vals[k:len(vals) - k])


def fmt3(v):
    return f"{v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" if v else "-"


def summarize(rows):
    def stat(col):
        vals = [r[col] for r in rows
                if col in r and isinstance(r[col], (int, float))]
        if not vals:
            return None
        return (round(statistics.mean(vals), 2),
                round(trimmed_mean(vals), 2),
                round(statistics.median(vals), 2))

    s = {"n": len(rows)}
    for c in COMPILERS:
        for m in METRICS:
            s[f"{c}_{m}"] = stat(f"{c}_{m}")
    # Cqlib 原生最强 = 每条线路按指标取 size/depth 两目标中更优者
    for m in ("swaps", "depth", "twoq_depth"):
        vals = []
        for r in rows:
            a = r.get(f"cqlib_size_{m}")
            b = r.get(f"cqlib_depth_{m}")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                vals.append(min(a, b))
        if vals:
            s[f"cqlib_best_{m}"] = (round(statistics.mean(vals), 2),
                                    round(trimmed_mean(vals), 2),
                                    round(statistics.median(vals), 2))
    for m in ("fidelity",):
        vals = []
        for r in rows:
            a = r.get("cqlib_size_fidelity")
            b = r.get("cqlib_depth_fidelity")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                vals.append(max(a, b))
        if vals:
            s[f"cqlib_best_{m}"] = (round(statistics.mean(vals), 2),
                                    round(trimmed_mean(vals), 2),
                                    round(statistics.median(vals), 2))
    return s


def render(sections, title, rows):
    s = summarize(rows)
    cols = ["cqlib_best", "cqlib_depth", "qtrial_cqlib", "qtrial_sabre",
            "sabre", "tket"]
    header = "| 指标（均值/截尾/中位） | Cqlib原生最强 | Cqlib原生depth | " \
             "QTrial+Cqlib | QTrial+SABRE | SABRE O1 | pytket |"
    lines = [f"\n## {title}（{s['n']} 条线路）\n", header,
             "|---|---|---|---|---|---|---|"]
    for m, fmt in (("swaps", fmt3), ("depth", fmt3), ("twoq_depth", fmt3)):
        lines.append(f"| {m.upper().replace('_', ' ')} | "
                     + " | ".join(fmt(s.get(f"{c}_{m}")) for c in cols) + " |")
    fid = lambda v: f"{v[0]:.4f}" if v else "-"
    lines.append("| 估计保真度（均值） | "
                 + " | ".join(fid(s.get(f"{c}_fidelity")) for c in cols) + " |")
    wall = lambda v: f"{v[0]:.1f}" if v else "-"
    lines.append("| 编译耗时（均值 s） | "
                 + " | ".join(wall(s.get(f"{c}_wall")) for c in cols) + " |")
    # 胜场统计（全部编译器完成的口径）
    ok = [r for r in rows
          if all(k in r for k in ("qtrial_cqlib_swaps", "qtrial_cqlib_depth",
                                  "sabre_swaps", "tket_swaps",
                                  "cqlib_size_swaps", "cqlib_depth_swaps"))]
    if ok:
        w_tket_d = sum(1 for r in ok if r["qtrial_cqlib_depth"]
                       < r["tket_depth"])
        w_cbest_s = sum(1 for r in ok if r["qtrial_cqlib_swaps"]
                        < min(r["cqlib_size_swaps"], r["cqlib_depth_swaps"]))
        w_sabre_s = sum(1 for r in ok if r["qtrial_cqlib_swaps"]
                        < r["sabre_swaps"])
        lines.append(f"\nQTrial+Cqlib 胜场（{len(ok)} 条全完成线路）："
                     f"深度 vs tket **{w_tket_d}**、SWAP vs Cqlib 原生最强 "
                     f"**{w_cbest_s}**、SWAP vs SABRE O1 **{w_sabre_s}**")
    lines.append("")
    sections.append("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-only", action="store_true")
    ap.add_argument("--medium-only", action="store_true")
    ap.add_argument("--large-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="每层最多评测线路数")
    ap.add_argument("--circuits", default="", help="仅评测指定线路（逗号分隔）")
    ap.add_argument("--rows-dir", default="tables/cqlib_inject",
                    help="增量行输出目录")
    args = ap.parse_args()
    bucket_filter = None
    if args.small_only:
        bucket_filter = {"small"}
    elif args.medium_only:
        bucket_filter = {"medium"}
    elif args.large_only:
        bucket_filter = {"large"}

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    import torch
    policy, ckpt = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    circuits = build_circuits(bucket_filter)
    if args.circuits:
        wanted = {c.strip() for c in args.circuits.split(",") if c.strip()}
        circuits = {b: [(n, qc) for n, qc in lst if n in wanted]
                    for b, lst in circuits.items()}
        circuits = {b: lst for b, lst in circuits.items() if lst}
    if args.limit:
        for b in circuits:
            circuits[b] = circuits[b][:args.limit]
    out_dir = Path(args.rows_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.circuits:  # 全新全量运行：清空旧行文件
        for p in out_dir.glob("*_rows.jsonl"):
            p.unlink()
    sections = ["# Cqlib 注入接口分层对比（天衍-287，105 比特 15×7 网格）\n",
                "QTrial+Cqlib = RL 多起点布局 + 自适应局部搜索 → 注入天衍平台"
                "原生 MCTS 路由（objective=depth）；统一口径 decompose_to_platform"
                " + compute_metrics；统计：均值/截尾(去10%)/中位数\n"]

    titles = {"small": "≤10 比特", "medium": "11-50 比特",
              "large": "51-105 比特（含满占）"}
    for bucket in circuits:
        print(f"评测 {bucket} 层（{len(circuits[bucket])} 条）...", flush=True)
        # cqlib 后端 mapper：路由感知重排自动关闭，RL 布局直接注入
        mapper_cqlib = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                              seed=42, selection_rule="depth",
                              routing_method="cqlib",
                              cqlib_objective="depth",
                              cqlib_timeout=600.0)
        mapper_sabre = Mapper(spec, policy=policy, cfg=cfg, dev="cuda",
                              seed=42, selection_rule="depth",
                              routing_method="sabre")
        rows = evaluate_bucket(circuits[bucket], spec, mapper_cqlib,
                               mapper_sabre,
                               out_file=out_dir / f"{bucket}_rows.jsonl")
        render(sections, titles[bucket], rows)

    (out_dir / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"\n报告 -> {out_dir}/report.md")


if __name__ == "__main__":
    main()
