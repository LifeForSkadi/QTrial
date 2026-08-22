"""Fixed-subset reproduction script with tolerance check vs tables/reference.json.

Usage: python scripts/reproduce.py [--checkpoint PATH] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import Config, TABLES_DIR, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pipeline.mapper import Mapper
from qtrail.pipeline.baselines import sabre_swap_count
from qtrail.utils.qasm_io import load_qasm2, strip_measurements

REF_PATH = TABLES_DIR / "reference.json"
TOLERANCE = 0.02  # ±2% mean SWAPs


def fixed_subset() -> list:
    """5 MQTBench n=15 + 5 QUEKO-20 circuits (fixed, deterministic)."""
    from mqt.bench import BenchmarkLevel, get_benchmark
    from qtrail.utils.qasm_io import decompose_circuit
    circuits = []
    for name in ("qft", "qaoa", "vqe_su2", "qpeexact", "grover"):
        try:
            qc = get_benchmark(benchmark=name, level=BenchmarkLevel.INDEP,
                               circuit_size=15, random_parameters=True)
            circuits.append((f"mqtbench/{name}_15", decompose_circuit(qc)))
        except Exception:
            pass
    from qtrail.utils.bench import iter_queko_files
    for p in list(iter_queko_files("BIGD"))[:5]:
        circuits.append((f"queko/{p.stem}", load_qasm2(p)))
    return circuits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--write-reference", action="store_true",
                    help="write current results as the reference")
    args = ap.parse_args()

    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)

    policy = None
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        from qtrail.config import CHECKPOINTS_DIR
        cks = sorted(CHECKPOINTS_DIR.glob("tianyan-287_*_best.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt_path = str(cks[0]) if cks else None
    if ckpt_path:
        import torch
        from qtrail.models import QAPolicy
        policy, ckpt = QAPolicy.load_checkpoint(ckpt_path, device_n=spec.n)
        policy.eval()
        cfg.model = ckpt.get("model_cfg", cfg.model)

    circuits = fixed_subset()
    print(f"reproducing on {len(circuits)} circuits x {args.seeds} seeds, "
          f"checkpoint={ckpt_path}")

    per_circuit = {}
    for name, qc in circuits:
        qc_clean, _ = strip_measurements(qc)
        swaps = []
        sabre1 = []
        for seed in range(args.seeds):
            mapper = Mapper(spec, policy=policy, cfg=cfg, dev="cpu", seed=seed)
            res = mapper.map_circuit(qc_clean, circuit_id=name)
            swaps.append(res.swap_count)
            sc, _ = sabre_swap_count(qc_clean, mapper.cm, optimization_level=1,
                                     seed=seed)
            sabre1.append(sc)
        per_circuit[name] = {
            "mean_swaps": float(np.mean(swaps)),
            "std_swaps": float(np.std(swaps)),
            "sabre_o1_mean": float(np.mean(sabre1)),
        }
        print(f"  {name:32s} ours {np.mean(swaps):6.1f}±{np.std(swaps):4.1f}  "
              f"| sabre-o1 {np.mean(sabre1):6.1f}")

    if args.write_reference:
        REF_PATH.parent.mkdir(parents=True, exist_ok=True)
        REF_PATH.write_text(json.dumps(per_circuit, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"reference written -> {REF_PATH}")
        return

    if not REF_PATH.exists():
        print(f"no reference at {REF_PATH}; run with --write-reference first "
              "(or after training) to establish the baseline")
        return

    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    ok = True
    for name, r in ref.items():
        ours = per_circuit.get(name)
        if ours is None:
            print(f"  MISSING {name}"); ok = False; continue
        abs_d = abs(ours["mean_swaps"] - r["mean_swaps"])
        rel = abs_d / max(r["mean_swaps"], 1)
        # pass if within 2 absolute swaps OR within 2% relative
        if abs_d > 2.0 and rel > TOLERANCE:
            print(f"  DEVIATION {name}: ref {r['mean_swaps']:.1f} vs "
                  f"now {ours['mean_swaps']:.1f} ({rel*100:.1f}%)")
            ok = False
    print("REPRODUCIBILITY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
