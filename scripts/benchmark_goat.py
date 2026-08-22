"""GOAT 编译器对比实验：在 tianyan-287（15×7 网格）拓扑上统一指标评测

对比对象：
  - Qiskit SABRE O1/O3（= 集成版 LightSABRE，Rust 重写）
  - MQT QMAP（慕尼黑量子工具包，启发式 + 分层启发式，含精确 SAT 模式）
  - pytket（Quantinuum，默认映射管线）
  - QTrial（本项目，混合路由竞技）
统一指标：路由 SWAP 数、2Q 门数、深度、估计保真度（同一校准数据）、静态代价。

用法: python scripts/benchmark_goat.py [--limit 20] [--out tables/goat]
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

# ---- mqt.qmap DLL 路径补丁（用户目录安装环境）----
import sysconfig as _sc
_USER_SITE = str(Path(__file__).resolve().parent.parent.parent.parent /
                 ".." / ".." / "Users" / "SKADI" / "AppData" / "Roaming" /
                 "Python" / "Python313" / "site-packages")
_USER_SITE = r"C:\Users\SKADI\AppData\Roaming\Python\Python313\site-packages"
_orig_get_paths = _sc.get_paths


def _patched(*a, **k):
    p = _orig_get_paths(*a, **k)
    p["purelib"] = _USER_SITE
    return p


_sc.get_paths = _patched

from qtrail.config import Config, TABLES_DIR, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import coupling_map_from_spec, ROUTING_BASIS
from qtrail.utils.qasm_io import load_qasm2, strip_measurements
from qtrail.utils.bench import iter_queko_files

CKPT = "checkpoints/tianyan-287_gat_combined_calib_best.pt"


def bench_circuits(limit: int) -> list:
    out = []
    for p in list(iter_queko_files("BIGD"))[:limit]:
        out.append((f"queko/{p.stem}", load_qasm2(p)))
    from mqt.bench import BenchmarkLevel, get_benchmark
    from qtrail.utils.qasm_io import decompose_circuit
    for name in ("qft", "qaoa", "qpeexact"):
        qc = get_benchmark(benchmark=name, level=BenchmarkLevel.INDEP,
                           circuit_size=15, random_parameters=True)
        out.append((f"mqtbench/{name}_15", decompose_circuit(qc)))
    return out


def qiskit_sabre(qc, cm, calib, opt):
    from qtrail.pipeline.baselines import sabre_swap_count, sabre_transpile
    sc, routed = sabre_swap_count(qc, cm, optimization_level=opt, seed=42)
    final = sabre_transpile(qc, cm, optimization_level=opt, seed=42)
    m = compute_metrics(final, sc, calib)
    return {**m, "compiler": f"qiskit-o{opt}"}


def qmap_compile(qc, cm_edges, n_phys, calib, method="heuristic"):
    import re
    from qiskit import qasm2
    from mqt.core.ir import QuantumComputation
    from mqt.qmap.sc import Architecture, Method, map_, Configuration
    qcomp = QuantumComputation.from_qasm_str(qasm2.dumps(qc))
    arch = Architecture(n_phys, cm_edges)
    cfg = Configuration()
    cfg.method = Method.heuristic
    mapped, results = map_(qcomp, arch, cfg)
    qasm_out = mapped.qasm2_str()
    if "qelib1.inc" not in qasm_out:  # qmap 输出缺标准头，补上
        qasm_out = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + qasm_out.split(
            "OPENQASM 2.0;", 1)[-1]
    n_swap = len(re.findall(r"swap q\[\d+\], q\[\d+\];", qasm_out))
    # qiskit.qasm2 不认 swap/p：展开为等价标准门
    qasm_out = re.sub(r"swap q\[(\d+)\], q\[(\d+)\];",
                      r"cx q[\1], q[\2]; cx q[\2], q[\1]; cx q[\1], q[\2];",
                      qasm_out)
    qasm_out = re.sub(r"\bp\(", "u1(", qasm_out)
    qc_back = qasm2.loads(qasm_out)
    m = compute_metrics(qc_back, n_swap, calib)
    return {**m, "compiler": f"qmap-{method}"}


def pytket_compile(qc, cm_edges, n_phys, calib):
    from pytket.architecture import Architecture
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.passes import (DecomposeBoxes, DefaultMappingPass,
                               SequencePass)
    arch = Architecture(cm_edges)
    tk = qiskit_to_tk(qc)
    # 仅映射+路由（不启 peephole 优化，保证门集公平）；
    # replace_implicit_swaps=True 把 tket 的隐式排列物化为真实 SWAP
    SequencePass([DecomposeBoxes(), DefaultMappingPass(arch)]).apply(tk)
    mapped = tk_to_qiskit(tk, replace_implicit_swaps=True)
    sc = mapped.count_ops().get("swap", 0)
    # 统一口径：swap 展开为 3 cx 后再计指标（与 QTrial/Qiskit 基相同）
    from qiskit import QuantumCircuit
    expanded = QuantumCircuit()
    for qr in mapped.qregs:
        expanded.add_register(qr)
    for cr in mapped.cregs:
        expanded.add_register(cr)
    for inst in mapped.data:
        if inst.operation.name == "swap":
            a, b = mapped.find_bit(inst.qubits[0]).index, mapped.find_bit(inst.qubits[1]).index
            expanded.cx(a, b); expanded.cx(b, a); expanded.cx(a, b)
        else:
            expanded.append(inst.operation, inst.qubits, inst.clbits)
    m = compute_metrics(expanded, sc, calib)
    return {**m, "compiler": "pytket"}


def qtrial(mapper, qc, name):
    res = mapper.map_circuit(qc, circuit_id=name)
    return {**res.metrics, "compiler": "qtrial", "method": res.method}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=str(TABLES_DIR / "goat"))
    args = ap.parse_args()

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    cm = coupling_map_from_spec(spec)
    cm_edges = [[i, j] for i in range(spec.n) for j in range(i + 1, spec.n)
                if spec.adj[i, j]]

    import torch
    policy, ckpt = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                            map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=42)

    circuits = bench_circuits(args.limit)
    rows = []
    for name, qc in circuits:
        qc_clean, _ = strip_measurements(qc)
        row = {"circuit": name, "n_qubits": qc_clean.num_qubits}
        t0 = time.time()
        try:
            row.update(qtrial(mapper, qc_clean, name))
            row["qtrial_wall"] = round(time.time() - t0, 2)
        except Exception as e:
            row["qtrial_error"] = str(e)
        for fn in (lambda: qiskit_sabre(qc_clean, cm, spec.calib, 1),
                   lambda: qiskit_sabre(qc_clean, cm, spec.calib, 3),
                   lambda: qmap_compile(qc_clean, cm_edges, spec.n, spec.calib),
                   lambda: pytket_compile(qc_clean, cm_edges, spec.n, spec.calib)):
            try:
                r = fn()
                row[r["compiler"] + "_swaps"] = r["swap_count"]
                row[r["compiler"] + "_twoq"] = r["twoq_count"]
                row[r["compiler"] + "_depth"] = r["depth"]
                row[r["compiler"] + "_fidelity"] = r["est_fidelity"]
            except Exception as e:
                row["error_" + fn.__name__] = f"{type(e).__name__}: {e}"[:120]
        rows.append(row)
        print(f"{name[:34]:36s} swaps: qtrial={row.get('swap_count')} "
              f"o1={row.get('qiskit-o1_swaps')} o3={row.get('qiskit-o3_swaps')} "
              f"qmap={row.get('qmap-heuristic_swaps')} tket={row.get('pytket_swaps')}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "goat_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")

    def stat(key):
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
        return (statistics.mean(vals), statistics.median(vals), len(vals)) if vals else (None, None, 0)

    summary = {"n_circuits": len(rows)}
    for c, keys in (("qtrial", ("swap_count", "twoq_count", "depth", "est_fidelity")),
                    ("qiskit-o1", ("qiskit-o1_swaps", "qiskit-o1_twoq", "qiskit-o1_depth", "qiskit-o1_fidelity")),
                    ("qiskit-o3", ("qiskit-o3_swaps", "qiskit-o3_twoq", "qiskit-o3_depth", "qiskit-o3_fidelity")),
                    ("qmap-heuristic", ("qmap-heuristic_swaps", "qmap-heuristic_twoq", "qmap-heuristic_depth", "qmap-heuristic_fidelity")),
                    ("pytket", ("pytket_swaps", "pytket_twoq", "pytket_depth", "pytket_fidelity"))):
        summary[c] = {"swaps": stat(keys[0]), "twoq": stat(keys[1]),
                      "depth": stat(keys[2]), "fidelity": stat(keys[3])}
    with open(out / "goat_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    print("\n===== GOAT 对比摘要（均值 / 中位数）=====")
    for c in ("qtrial", "qiskit-o1", "qiskit-o3", "qmap-heuristic", "pytket"):
        s = summary[c]["swaps"]
        if s[0] is None:
            print(f"{c:16s} (无有效数据)")
            continue
        f = summary[c]["fidelity"]
        print(f"{c:16s} swaps {s[0]:7.1f}/{s[1]:5.1f} | 2q {summary[c]['twoq'][0]:7.1f} | "
              f"depth {summary[c]['depth'][0]:7.1f} | fid {f[0]:.4f} (n={s[2]})")
    print(f"输出 -> {out}/")


if __name__ == "__main__":
    main()
