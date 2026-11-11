# -*- coding: utf-8 -*-
"""RL 布局 + 感知O3 全量重测（587 的 small+medium 层，2026-11-12）。

与 bench_layout_inject 同口径（无筛选版）：RL 布局（QiskitMapper O1，
c0.05 权重、seed=42）→ initial_layout 注入感知 O3 → compute_metrics。
线路集与 final_report_bench_v7.load_circuits 完全一致（small+medium）。
输出：tables/compare_models/rl_o3_587_rows.jsonl，断点续跑。
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import qasm2, transpile

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import build_target
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import PLATFORM_BASIS, load_qasm2, strip_measurements

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/compare_models/rl_o3_587_rows.jsonl")
CACHE = Path("data/mqtbench/stratified")
BP = Path("data/benchpress/qasm")


def load_circuits():
    """与 final_report_bench_v7 完全同口径的 small+medium 线路清单
    （存路径，运行时经 load_qasm2 统一 sanitize+预分解——与
    bench_layout_inject 同口径）。"""
    buckets = {"small": [], "medium": []}
    for p in BP.rglob("*.qasm"):
        try:
            circ = parse_qasm(p.read_text(encoding="utf-8",
                                          errors="replace"))
            if circ.n == 0:
                continue
        except Exception:
            continue
        if circ.n > 105:
            continue
        b = "small" if circ.n <= 10 else ("medium" if circ.n <= 50 else None)
        if b:
            buckets[b].append((f"bp_{p.parent.name}_{p.stem}", p))
    for p in sorted(CACHE.glob("*.qasm")):
        name = p.stem
        size = int(name.rsplit("_", 1)[1])
        if size > 105:
            continue
        b = "small" if size <= 10 else ("medium" if size <= 50 else None)
        if b:
            buckets[b].append((name, p))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.(3|4)D1_\.(4|5)D2_", p.name) \
                or re.search(r"\.0D1_\.1D2_", p.name):
            buckets["medium"].append((p.stem, p))
    return buckets


def main():
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    target = build_target(spec)

    policy, ckpt = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                            map_location="cpu")
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    mapper = QiskitMapper(spec, policy=policy, cfg=cfg, dev="cpu", seed=42)

    buckets = load_circuits()
    total = sum(len(v) for v in buckets.values())
    print(f"circuits: small {len(buckets['small'])} / medium "
          f"{len(buckets['medium'])} (total {total})", flush=True)

    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["circuit"]] = r

    rows = list(done.values())
    for bucket in ("small", "medium"):
        for i, (name, path) in enumerate(buckets[bucket]):
            if name in done:
                continue
            row = {"circuit": name, "bucket": bucket}
            try:
                qc = load_qasm2(path)  # sanitize + 预分解（同 inj 口径）
                qc_clean, _ = strip_measurements(qc)
                row["n"] = qc_clean.num_qubits
                t0 = time.time()
                res = mapper.map_circuit(qc_clean, circuit_id=name,
                                         optimization_level=1)
                init = {qc_clean.qubits[j]: int(res.layout[j])
                        for j in range(qc_clean.num_qubits)}
                routed = transpile(qc_clean, target=target,
                                   basis_gates=PLATFORM_BASIS,
                                   optimization_level=3,
                                   initial_layout=init,
                                   seed_transpiler=42)
                m = compute_metrics(routed,
                                    routed.count_ops().get("swap", 0),
                                    calib)
                row["rl_o3_fidelity"] = m["est_fidelity"]
                row["rl_o3_twoq"] = m["twoq_count"]
                row["rl_o3_depth"] = m["depth"]
                row["rl_o3_wall"] = round(time.time() - t0, 2)
            except Exception as e:
                row["rl_o3_error"] = f"{type(e).__name__}: {str(e)[:60]}"
            rows.append(row)
            with open(OUT, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  [{bucket}] {i + 1}/{len(buckets[bucket])} "
                      f"({name})", flush=True)
        print(f"[{bucket}] 完成 {len(buckets[bucket])} 条", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
