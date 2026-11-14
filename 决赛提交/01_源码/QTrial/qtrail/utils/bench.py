"""Benchmark loaders: QUEKO files + MQTBench generation (mqt.bench)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from qtrail.config import DATA_DIR
from qtrail.problems.program_graph import ProgramGraph, build_program_graph
from qtrail.utils.qasm_io import extract_ops, load_qasm2

QUEKO_DIR = DATA_DIR / "Queko"
MQTBENCH_DIR = DATA_DIR / "mqtbench"


# ------------------------------------------------------------------ QUEKO
def iter_queko_files(split: str = "BIGD", limit: int | None = None):
    """Yield (path, name) of QUEKO benchmark QASM files."""
    d = QUEKO_DIR / split
    if not d.exists():
        raise FileNotFoundError(f"QUEKO split not found: {d} "
                                f"(download with tools/download_queko.py)")
    files = sorted(d.glob("*.qasm"))
    if limit:
        files = files[:limit]
    return files


def load_queko_program_graphs(split: str = "BIGD", limit: int | None = None,
                              use_features: bool = True,
                              temporal_alpha: float = 0.0,
                              keep_qasm: bool = False) -> list[ProgramGraph]:
    """Load QUEKO circuits as ProgramGraphs (decompose not needed: bare cx).

    keep_qasm=True 时把分解后的 QASM 文本存入 ops_meta（路由 oracle 训练用）。
    """
    from qiskit import qasm2
    graphs = []
    for p in iter_queko_files(split, limit):
        qc = load_qasm2(p, decompose=True)
        ops = extract_ops(qc)
        g = build_program_graph(qc.num_qubits, ops, circuit_id=f"{split}/{p.stem}",
                                compute_feats=use_features,
                                temporal_alpha=temporal_alpha)
        if keep_qasm:
            g.ops_meta["qasm"] = qasm2.dumps(qc)
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------- MQTBench
def generate_mqtbench_graphs(sizes=(10, 15, 20, 25), limit_per_type: int | None = None,
                             seed: int = 0, use_features: bool = True,
                             temporal_alpha: float = 0.0,
                             keep_qasm: bool = False) -> list[ProgramGraph]:
    """Generate MQTBench INDEP-level circuits as ProgramGraphs.

    Requires mqt.bench. Some benchmark types fail for certain sizes; those are
    skipped with a warning count.
    """
    from mqt.bench import BenchmarkLevel, get_benchmark
    from mqt.bench.benchmarks import get_available_benchmark_names

    names = get_available_benchmark_names()
    graphs: list[ProgramGraph] = []
    skipped = 0
    for name in sorted(names):
        for size in sizes:
            try:
                qc = get_benchmark(benchmark=name, level=BenchmarkLevel.INDEP,
                                   circuit_size=size, random_parameters=True)
                ops = extract_ops(qc)
                if qc.num_qubits > 40:
                    raise ValueError("too large for training pool")
                g = build_program_graph(qc.num_qubits, ops,
                                        circuit_id=f"mqtbench/{name}_{size}",
                                        compute_feats=use_features,
                                        temporal_alpha=temporal_alpha)
                if keep_qasm:
                    from qiskit import qasm2
                    g.ops_meta["qasm"] = qasm2.dumps(qc)
                graphs.append(g)
            except Exception:
                skipped += 1
            if limit_per_type and len(graphs) >= limit_per_type:
                return graphs
    if skipped:
        print(f"[bench] {skipped} mqtbench (name,size) combos skipped")
    return graphs


def save_graph_pool(graphs: list[ProgramGraph], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)


def load_graph_pool(path: str | Path) -> list[ProgramGraph]:
    with open(path, "rb") as f:
        return pickle.load(f)


def default_pool_path() -> Path:
    return MQTBENCH_DIR / "graph_pool.pkl"


# ----------------------------------------------------------------- QASMBench
def load_qasmbench_graphs(max_n: int | None = None,
                          temporal_alpha: float = 0.0,
                          use_features: bool = True,
                          limit_per_dir: int | None = None) -> list[ProgramGraph]:
    """Load QASMBench circuits (small/medium/large) as ProgramGraphs.

    QASMBench 覆盖 2-433 比特（大 N 训练缺口）。跳过 *_transpiled 变体。
    """
    from qtrail.config import DATA_DIR
    base = DATA_DIR / "QASMBench"
    graphs = []
    for d in ("small", "medium", "large"):
        dd = base / d
        if not dd.exists():
            continue
        files = sorted(dd.rglob("*.qasm"))
        files = [p for p in files if "transpiled" not in p.name]
        if limit_per_dir:
            files = files[:limit_per_dir]
        for p in files:
            try:
                qc = load_qasm2(p, decompose=True)
                if max_n is not None and qc.num_qubits > max_n:
                    continue
                ops = extract_ops(qc)
                g = build_program_graph(qc.num_qubits, ops,
                                        circuit_id=f"qasmbench/{d}/{p.stem}",
                                        compute_feats=use_features,
                                        temporal_alpha=temporal_alpha)
                graphs.append(g)
            except Exception:
                continue
    return graphs
