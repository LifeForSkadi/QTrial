# -*- coding: utf-8 -*-
"""把线路用自研 v8 管线映射到指定天衍机器的**实时拓扑**并导出 QCIS。

用法：
  python scripts/real_machine_v8_map.py x5 ghz5 qft5 --machine tianyan176

输入：examples/<name>.qasm 或 out/real_machine/<name>.qasm
输出：out/real_machine/<machine>/<name>.qcis + _metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices.adapter import download_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.export import to_qcis
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt")


def find_qasm(name: str) -> Path:
    for p in (Path("examples") / f"{name}.qasm",
              Path("out/real_machine") / f"{name}.qasm"):
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--machine", default="tianyan176")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = download_tianyan287_spec(args.machine)
    if spec is None:
        raise SystemExit(f"cannot build live spec for {args.machine}")
    print(f"[spec] {args.machine}: n={spec.n} labels={spec.qubit_labels[:3]}"
          f"..{spec.qubit_labels[-1]} disabled={spec.disabled_qubits}",
          flush=True)
    policy, _ = QAPolicy.load_checkpoint(args.checkpoint, device_n=spec.n,
                                         map_location=args.dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    out_dir = Path("out/real_machine") / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        circ = parse_qasm(find_qasm(name).read_text(encoding="utf-8"))
        t0 = time.time()
        torch.manual_seed(args.seed)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=args.dev,
                            seed=args.seed, selection_rule="fidelity",
                            use_post=True, pipeline="v5", routing_seeds=8)
        res = mapper.map_circuit(circ, circuit_id=name)
        qcis = to_qcis(res["routed_circuit"], labels=spec.qubit_labels)
        (out_dir / f"{name}.qcis").write_text(qcis, encoding="utf-8")
        metrics = dict(res["metrics"])
        metrics["method"] = res.get("method")
        metrics["wall_s"] = round(time.time() - t0, 2)
        metrics["layout"] = {str(k): int(v) for k, v in res["layout"].items()}
        metrics["final_layout"] = {str(k): int(v)
                                   for k, v in res["final_layout"].items()}
        metrics["machine"] = args.machine
        (out_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8")
        print(f"[map] {name}: twoq={metrics['twoq_count']} "
              f"depth={metrics['depth']} fid={metrics['est_fidelity']:.4f} "
              f"qcis_lines={len(qcis.splitlines())} ({metrics['wall_s']}s)",
              flush=True)


if __name__ == "__main__":
    main()
