# -*- coding: utf-8 -*-
"""补跑超时线路（修复后代码）：5 条已知 timeout 单独重跑，
写 legality_rows_retry.jsonl（合并时优先于旧 timeout 行）。
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
TARGETS = [
    "bp_feynman_hwb11",
    "bp_feynman_hwb12",
    "bp_square-heisenberg_square_heisenberg_N49",
    "bp_vqe_uccsd_n28_vqe_uccsd_n28",
    "graphstate_25",
    "bp_basis_trotter_n4_basis_trotter_n4_transpiled",
]
OUT = Path("tables/final_report_v7/legality_rows_retry.jsonl")


def find_circuit(name):
    for p in Path("data/benchpress/qasm").rglob("*.qasm"):
        n = f"bp_{p.parent.name}_{p.stem}"
        if n == name:
            return p
    return None


def run_with_timeout(fn, timeout):
    box = {}

    def target():
        try:
            box["res"] = fn()
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    if "err" in box:
        raise box["err"]
    return box.get("res"), None


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    for name in TARGETS:
        p = find_circuit(name)
        if p is None:
            print(f'{name}: 文件未找到，跳过', flush=True)
            continue
        circ = parse_qasm(p.read_text(encoding="utf-8", errors="replace"))
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                            selection_rule="fidelity", use_post=True,
                            pipeline="v5", routing_seeds=16)
        t0 = time.time()
        row = {"circuit": name, "n": circ.n}
        try:
            res, terr = run_with_timeout(
                lambda: mapper.map_circuit(circ, circuit_id=name), 1800)
            if terr is not None:
                row["error"] = "timeout"
                print(f'{name}: 仍超时（1800s）', flush=True)
            else:
                from qtrail.pure.metrics import compute_metrics
                final_qc = res["routed_circuit"]
                m = res["metrics"]
                probs = []
                disabled = set(spec.disabled_qubits)
                names = set()
                for idx, inst in enumerate(final_qc.ops):
                    names.add(inst.name)
                    qs = [int(q) for q in inst.qubits]
                    if inst.name == "swap":
                        probs.append(f"swap残留@{idx}")
                    if any(q >= spec.n for q in qs):
                        probs.append(f"越界{qs}@{idx}")
                    if any(q in disabled for q in qs):
                        probs.append(f"禁用{qs}@{idx}")
                    if inst.name == "cz" and not spec.adj[qs[0], qs[1]]:
                        probs.append(f"cz非相邻{qs}@{idx}")
                if not names <= {"rz", "sx", "x", "cz", "measure"}:
                    probs.append(f"门集{names}")
                for idx2, mm in enumerate(final_qc.measures):
                    q = int(mm.qubits[0])
                    if q >= spec.n:
                        probs.append(f"measure{idx2}越界")
                    if q in disabled:
                        probs.append(f"measure{idx2}禁用")
                row["legal"] = not probs
                if probs:
                    row["violations"] = probs[:8]
                row["twoq"] = m["twoq_count"]
                row["depth"] = m["depth"]
                row["fid"] = m["est_fidelity"]
                row["wall"] = round(time.time() - t0, 1)
                print(f'{name}: 合法={row["legal"]} twoq={row["twoq"]} '
                      f'depth={row["depth"]} ({row["wall"]}s)', flush=True)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
            print(f'{name}: {row["error"]}', flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
