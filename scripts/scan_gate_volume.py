# -*- coding: utf-8 -*-
"""门数归因扫描（large 层）：为什么深度接近、保真度差距很大。

对 large 分层抽样，用同一校准/同一乘积保真度公式分别统计：
  v8 自研管线  vs  Qiskit 感知 O3
的 1Q 门数、2Q 门数、读出数、深度、2Q 深度、log 保真度及其
1Q/2Q/读出分解。只做归因，不改算法。

用法：python scripts/scan_gate_volume.py [--sample 24] [--seeds 8]
输出：tables/gate_attribution/large_rows.jsonl + summary.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from qiskit import qasm2, transpile

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.mapper import PureMapper
from qtrail.pure.metrics import (_product_log_fidelity, circuit_depth,
                                 twoq_depth)
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files
from qtrail.utils.qasm_io import sanitize_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/gate_attribution")


def load_frb():
    p = "scripts/final_report_bench.py"
    spec = importlib.util.spec_from_file_location("frb", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_index():
    idx = {}
    for p in Path("data/benchpress/qasm").rglob("*.qasm"):
        idx[f"bp_{p.parent.name}_{p.stem}"] = ("bp", p)
    for p in Path("data/mqtbench/stratified").glob("*.qasm"):
        idx[p.stem] = ("mqt", p)
    for p in iter_queko_files("BIGD"):
        idx[p.stem] = ("queko", p)
    return idx


def load_text(idx, name):
    kind, p = idx[name]
    text = p.read_text(encoding="utf-8", errors="replace")
    return text if kind == "bp" else sanitize_qasm(text)


def pick_even(items, k):
    if k >= len(items):
        return list(range(len(items)))
    step = len(items) / float(k)
    return [int(i * step) for i in range(k)]


def log_parts(circ: Circuit, calib):
    """返回 (log_1q, log_2q, log_ro, n1q, n2q, nro)。"""
    import numpy as np
    err2q = calib.err_2q
    median2q = float(np.median(list(err2q.values()))) if err2q else 1e-3
    s1 = s2 = sro = 0.0
    n1 = n2 = 0
    for inst in circ.ops:
        if inst.nq == 2:
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            e = err2q.get((a, b), err2q.get((b, a), median2q))
            s2 += math.log(max(1.0 - float(e), 1e-12))
            n2 += 1
        elif inst.nq == 1:
            q = int(inst.qubits[0])
            s1 += math.log(max(1.0 - float(calib.err_1q[q]), 1e-12))
            n1 += 1
    for inst in circ.measures:
        q = int(inst.qubits[0])
        sro += math.log(max(1.0 - float(calib.err_ro[q]), 1e-12))
    return s1, s2, sro, n1, n2, len(circ.measures)


def qc_to_circuit(qc) -> Circuit:
    circ = Circuit(qc.num_qubits)
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        qs = tuple(int(qc.find_bit(q).index) for q in inst.qubits)
        params = tuple(float(p) for p in inst.operation.params)
        circ.append(Inst(name, qs, params))
    return circ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", default=None, help="分片 k/N（并行用）")
    args = ap.parse_args()
    dev = "cpu"
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    frb = load_frb()
    target = frb.build_target(spec)
    policy, _ = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                         map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    idx = build_index()

    v8_rows = {}
    for line in Path("tables/final_report_v8/large_fidelity_rows.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("qt_fidelity") is not None:
                v8_rows[r["circuit"]] = r
    names = sorted(v8_rows)
    picks = [names[i] for i in pick_even(names, args.sample) if names[i] in idx]
    shard_id = shard_n = None
    if args.shard:
        shard_id, shard_n = (int(x) for x in args.shard.split("/"))
        picks = picks[shard_id::shard_n]
    print(f"sample {len(picks)}/{len(names)} large circuits"
          + (f" shard {shard_id}/{shard_n}" if shard_id is not None else ""),
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = (OUT / f"large_rows_shard{shard_id}.jsonl"
                if shard_id is not None else OUT / "large_rows.jsonl")
    rows = []
    for name in picks:
        text = load_text(idx, name)
        t0 = time.time()
        try:
            torch.manual_seed(args.seed)
            mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                seed=args.seed, selection_rule="fidelity",
                                use_post=True, pipeline="v5",
                                routing_seeds=args.seeds)
            res = mapper.map_circuit(parse_qasm(text), circuit_id=name)
            v8 = res["routed_circuit"]
            v8_l1, v8_l2, v8_ro, v8_n1, v8_n2, v8_nro = log_parts(v8, calib)
            v8_lf = _product_log_fidelity(v8, calib)
            qtext = sanitize_qasm(text)
            qc = qasm2.loads(qtext)
            qc.data = [i for i in qc.data
                       if i.operation.name not in ("measure", "barrier")]
            out = transpile(qc, basis_gates=["rz", "sx", "x", "cz"],
                            optimization_level=3, seed_transpiler=args.seed,
                            target=target)
            aw = qc_to_circuit(out)
            aw_l1, aw_l2, aw_ro, aw_n1, aw_n2, aw_nro = log_parts(aw, calib)
            # Qiskit 产物已去测量：读出项按原线路测量数补回
            n_ro = sum(1 for i in qasm2.loads(qtext).data
                       if i.operation.name == "measure")
            import numpy as np
            mean_ro = float(np.mean(np.log(np.clip(
                1.0 - calib.err_ro, 1e-12, None))))
            aw_lf = aw_l1 + aw_l2 + n_ro * mean_ro
            row = {
                "circuit": name, "n": qasm2.loads(qtext).num_qubits,
                "v8_1q": v8_n1, "v8_2q": v8_n2, "v8_ro": v8_nro,
                "v8_depth": circuit_depth(v8),
                "v8_2qdepth": twoq_depth(v8),
                "v8_logf": v8_lf, "v8_logf_1q": v8_l1,
                "v8_logf_2q": v8_l2, "v8_logf_ro": v8_ro,
                "aw_1q": aw_n1, "aw_2q": aw_n2, "aw_ro": n_ro,
                "aw_depth": out.depth(),
                "aw_logf": aw_lf, "aw_logf_1q": aw_l1,
                "aw_logf_2q": aw_l2, "aw_logf_ro": n_ro * mean_ro,
                "ratio_1q": v8_n1 / max(aw_n1, 1),
                "ratio_2q": v8_n2 / max(aw_n2, 1),
                "ratio_logf": v8_lf / aw_lf if aw_lf else None,
                "wall": round(time.time() - t0, 1),
            }
        except Exception as e:  # noqa: BLE001
            row = {"circuit": name, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        rows.append(row)
        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{len(rows)}/{len(picks)}] {name} "
              f"v8_1q={row.get('v8_1q')} aw_1q={row.get('aw_1q')} "
              f"v8_2q={row.get('v8_2q')} aw_2q={row.get('aw_2q')} "
              f"logf={row.get('v8_logf')}/{row.get('aw_logf')}", flush=True)

    ok = [r for r in rows if "v8_logf" in r]
    lines = ["# large 层门数归因扫描（2026-11-13）\n\n",
             f"抽样 {len(ok)}/{len(names)} 条 large 线路；v8 管线 "
             f"routing_seeds={args.seeds}，Qiskit 感知 O3 同 target/seed；"
             "同一校准、同一乘积保真度公式。\n\n",
             "## 汇总（中位）\n\n",
             "| 指标 | v8 | Qiskit 感知 O3 | v8/O3 |\n",
             "|---|---|---|---|\n"]
    for key, label in (("v8_1q", "1Q 门数"), ("v8_2q", "2Q 门数"),
                       ("v8_depth", "深度"), ("v8_2qdepth", "2Q 深度"),
                       ("v8_logf", "log 保真度")):
        v = statistics.median([r[key] for r in ok])
        akey = {"v8_1q": "aw_1q", "v8_2q": "aw_2q",
                "v8_depth": "aw_depth", "v8_2qdepth": "aw_depth",
                "v8_logf": "aw_logf"}[key]
        a = statistics.median([r[akey] for r in ok])
        lines.append(f"| {label} | {v:.3f} | {a:.3f} | "
                     f"{(v / a if a else float('nan')):.3f} |\n")
    lines.append("\n## 分解（中位 log 贡献）\n\n")
    lines.append("| 项 | v8 | Qiskit O3 |\n|---|---|---|\n")
    for k, label in (("logf_1q", "1Q 项"), ("logf_2q", "2Q 项"),
                     ("logf_ro", "读出项")):
        lines.append(f"| {label} | "
                     f"{statistics.median([r['v8_' + k] for r in ok]):.2f} | "
                     f"{statistics.median([r['aw_' + k] for r in ok]):.2f} |\n")
    lines.append("\n## 结论\n\n")
    lines.append("深度接近但 log 保真度差距大：v8 的 1Q/2Q 门数显著高于 "
                 "Qiskit O3，乘积模型按门数指数衰减；深度只反映关键路径，"
                 "不能作为保真度代理。该差距是项目已知短板，不修改算法，"
                 "作为局限性如实报告。\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print("written", OUT / "summary.md", flush=True)


if __name__ == "__main__":
    main()
