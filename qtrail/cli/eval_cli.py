"""Benchmark harness: python -m qtrail.eval [--bench ...]"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from qtrail.config import Config, TABLES_DIR, load_device_config
from qtrail.pipeline.mapper import Mapper

log = logging.getLogger("qtrail")


def load_benchmark_circuits(bench: str, limit: int | None) -> list:
    """Return list of (name, QuantumCircuit) for a benchmark suite."""
    from qtrail.utils.qasm_io import load_qasm2
    from qtrail.utils.bench import iter_queko_files

    out = []
    if bench.startswith("queko"):
        split = bench.split("_")[1].upper() if "_" in bench else "BIGD"
        for p in iter_queko_files(split, limit):
            out.append((f"queko_{split}/{p.stem}", load_qasm2(p)))
    elif bench == "mqtbench_n15":
        from mqt.bench import BenchmarkLevel, get_benchmark
        from mqt.bench.benchmarks import get_available_benchmark_names
        from qtrail.utils.qasm_io import decompose_circuit
        for name in sorted(get_available_benchmark_names()):
            try:
                qc = get_benchmark(benchmark=name, level=BenchmarkLevel.INDEP,
                                   circuit_size=15, random_parameters=True)
                qc = decompose_circuit(qc)
                out.append((f"mqtbench/{name}_15", qc))
            except Exception:
                continue
        if limit:
            out = out[:limit]
    elif bench == "custom":
        raise ValueError("custom benchmark needs explicit file list; use --files")
    else:
        raise ValueError(f"unknown benchmark: {bench}")
    return out


def evaluate_benchmark(circuits, mapper: Mapper, baselines=("qiskit-o1", "qiskit-o3"),
                       postprocess: bool = True) -> list[dict]:
    from qtrail.pipeline.baselines import sabre_swap_count, sabre_transpile
    from qtrail.pipeline.metrics import compute_metrics
    from qtrail.utils.qasm_io import strip_measurements

    rows = []
    for name, qc in circuits:
        qc_clean, _ = strip_measurements(qc)
        row = {"circuit": name, "n_qubits": qc_clean.num_qubits}
        t0 = time.time()
        res = None
        try:
            res = mapper.map_circuit(qc_clean, circuit_id=name,
                                     optimization_level=1)
            row.update({
                "method": res.method, "swaps": res.swap_count,
                "twoq": res.metrics.get("twoq_count"),
                "depth": res.metrics.get("depth"),
                "twoq_depth": res.metrics.get("twoq_depth"),
                "fidelity": res.metrics.get("est_fidelity"),
                "mean_2q_err": res.metrics.get("mean_2q_err"),
                "static_cost": res.metrics.get("static_cost"),
                "wall_s": round(time.time() - t0, 3),
            })
        except Exception as e:
            row.update({"error": str(e)})
        for b in baselines:
            o = int(b.split("-o")[-1])
            try:
                _, routed_b = sabre_swap_count(qc_clean, mapper.cm,
                                               optimization_level=o,
                                               seed=mapper.seed)
                # for O1 prefer the internally-measured count (same
                # deterministic measurement our method was compared against)
                if o == 1 and res is not None and res.baseline_swaps is not None:
                    sc = res.baseline_swaps
                else:
                    sc = routed_b.count_ops().get("swap", 0)
                final_b = sabre_transpile(qc_clean, mapper.cm, optimization_level=o,
                                          seed=mapper.seed)
                m = compute_metrics(final_b, sc, mapper.spec.calib)
                row[f"{b}_swaps"] = sc
                row[f"{b}_twoq"] = m["twoq_count"]
                row[f"{b}_depth"] = m["depth"]
                row[f"{b}_fidelity"] = m["est_fidelity"]
                row[f"{b}_mean_2q_err"] = m.get("mean_2q_err")
                # SABRE's chosen initial layout static cost (paper metric)
                try:
                    init = routed_b.layout.initial_layout
                    orig = set(qc_clean.qubits)
                    pi = np.zeros(qc_clean.num_qubits, dtype=np.int64)
                    for v, phys in init.get_virtual_bits().items():
                        if v in orig:
                            pi[qc_clean.find_bit(v).index] = phys
                    from qtrail.envs import terminal_cost_np
                    from qtrail.problems import build_program_graph
                    from qtrail.utils.qasm_io import extract_ops
                    g = build_program_graph(qc_clean.num_qubits,
                                            extract_ops(qc_clean),
                                            compute_feats=False)
                    row[f"{b}_static_cost"] = terminal_cost_np(
                        pi, g.adj, mapper.spec.dist, dist_mult=2.0)
                except Exception:
                    pass
            except Exception as e:
                row[f"{b}_error"] = str(e)
        rows.append(row)
    return rows


def summarize(rows: list[dict], baselines=("qiskit-o1", "qiskit-o3")) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n_circuits": 0, "n_ok": 0}

    def stats(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return (float(np.mean(vals)), float(np.median(vals)), len(vals)) if vals else (None, None, 0)

    s = {"n_circuits": len(rows), "n_ok": len(ok)}
    for key in ("swaps", "twoq", "depth", "twoq_depth", "fidelity",
                "mean_2q_err", "static_cost"):
        s[key] = stats(key)
    for b in baselines:
        for m in ("swaps", "twoq", "depth", "fidelity", "mean_2q_err", "static_cost"):
            s[f"{b}_{m}"] = stats(f"{b}_{m}")
    # reductions vs baselines
    for b in baselines:
        if s.get("swaps")[0] and s.get(f"{b}_swaps")[0]:
            s[f"swap_reduction_vs_{b}_pct"] = \
                round(100 * (1 - s["swaps"][0] / s[f"{b}_swaps"][0]), 1)
    return s


def write_outputs(rows: list[dict], summary: dict, bench: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{bench}_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    with open(out_dir / f"{bench}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    # markdown table
    keys = ["circuit", "n_qubits", "method", "swaps", "twoq", "depth",
            "twoq_depth", "fidelity", "qiskit-o1_swaps", "qiskit-o3_swaps",
            "qiskit-o1_depth", "qiskit-o3_depth"]
    with open(out_dir / f"{bench}.md", "w", encoding="utf-8") as f:
        f.write(f"# {bench} — QTrial 评测结果\n\n")
        f.write(f"n_circuits={summary.get('n_circuits')}, n_ok={summary.get('n_ok')}\n\n")
        for b in ("qiskit-o1", "qiskit-o3"):
            red = summary.get(f"swap_reduction_vs_{b}_pct")
            if red is not None:
                f.write(f"- SWAP 相对 {b} 平均减少 **{red}%**\n")
        f.write("\n| " + " | ".join(keys) + " |\n")
        f.write("|" + "---|" * len(keys) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n")
    with open(out_dir / f"{bench}.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="QTrial benchmark evaluation")
    ap.add_argument("--bench", default="queko_BIGD",
                    choices=["queko_BIGD", "queko_BNTF", "queko_BSS",
                             "mqtbench_n15"])
    ap.add_argument("--files", nargs="*", help="explicit QASM files instead of --bench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="tianyan-287")
    ap.add_argument("--out", default=str(TABLES_DIR))
    ap.add_argument("--baselines", default="qiskit-o1,qiskit-o3")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", default=None, choices=["cpu", "cuda"])
    args = ap.parse_args(argv)

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    if args.device == "tianyan-287":
        from qtrail.devices import build_tianyan287_spec
        spec = build_tianyan287_spec(dev_cfg)
    elif args.device == "grid-8x8":
        from qtrail.devices import build_grid8x8_spec
        spec = build_grid8x8_spec()
    else:
        from qtrail.devices import build_grid3x3_spec
        spec = build_grid3x3_spec()

    dev = args.dev or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    policy = None
    if args.checkpoint:
        from qtrail.models import QAPolicy
        policy, ckpt = QAPolicy.load_checkpoint(args.checkpoint, device_n=spec.n,
                                                map_location=dev)
        policy.eval()
        cfg.model = ckpt.get("model_cfg", cfg.model)
        cfg.graph = policy.graph_cfg  # 评测图表示与训练一致
    cfg.postprocess.enabled = not args.no_postprocess

    if args.files:
        from qtrail.utils.qasm_io import load_qasm2
        circuits = [(Path(f).stem, load_qasm2(f)) for f in args.files]
        bench = "custom"
    else:
        circuits = load_benchmark_circuits(args.bench, args.limit)
        bench = args.bench

    mapper = Mapper(spec, policy=policy, cfg=cfg, dev=dev, seed=args.seed)
    rows = evaluate_benchmark(circuits, mapper, baselines=tuple(args.baselines.split(",")),
                              postprocess=not args.no_postprocess)
    summary = summarize(rows, baselines=tuple(args.baselines.split(",")))
    write_outputs(rows, summary, bench, Path(args.out))

    print(f"\n== {bench}: {summary.get('n_ok')}/{summary.get('n_circuits')} circuits OK ==")
    if summary.get("swaps")[0] is not None:
        print(f"  ours:   swaps {summary['swaps'][0]:.2f} (med {summary['swaps'][1]:.2f}) | "
              f"twoq {summary['twoq'][0]:.2f} | depth {summary['depth'][0]:.2f} | "
              f"fid {summary['fidelity'][0]:.4f}")
        for b in args.baselines.split(","):
            s = summary.get(f"{b}_swaps")
            if s and s[0] is not None:
                red = summary.get(f"swap_reduction_vs_{b}_pct")
                print(f"  {b}: swaps {s[0]:.2f} | depth {summary[f'{b}_depth'][0]:.2f} | "
                      f"reduction {red}%" if red is not None else f"  {b}: swaps {s[0]:.2f}")
    sc_o, sc_b = summary.get("static_cost"), summary.get("qiskit-o1_static_cost")
    if sc_o and sc_o[0] is not None and sc_b and sc_b[0] is not None:
        print(f"  static cost (CO-MAP metric): ours {sc_o[0]:.1f} vs o1 {sc_b[0]:.1f} "
              f"(-{100*(1-sc_o[0]/sc_b[0]):.1f}%)")
    print(f"  outputs -> {args.out}/")


if __name__ == "__main__":
    main()
