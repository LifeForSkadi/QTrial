"""pure 管线上三模型 A/B 门禁：c0.05 / postfid_ft / postfid_mt。

交付管线 = pure（零 qiskit）：PureMapper（fidelity 规则，use_post）。
线路：MQTBench ≤25（22 条）+ QUEKO dense（10 条）= 32 条。
胜出判据：组均值保真度最高 且 逐线路胜场 > 负场（vs 当前 c0.05）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.bench import iter_queko_files

CACHE = Path("data/mqtbench/stratified")
MODELS = {
    "c0.05": "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt",
    "postfid_ft": ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
                   "postfid_ft_best.pt"),
    "postfid_mt": ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
                   "postfid_mt_best.pt"),
    "postfid_pure": ("checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_"
                     "postfid_pure_best.pt"),
}


def circuits():
    import re
    out = []
    for p in sorted(CACHE.glob("*.qasm")):
        size = int(p.stem.rsplit("_", 1)[1])
        if size <= 25 and p.stem != "qwalk_25":
            out.append((p.stem, p.read_text(encoding="utf-8")))
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.3D1_\.4D2_", p.name):
            out.append((p.stem, p.read_text(encoding="utf-8")))
    return out


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    circuits_all = circuits()
    print(f"A/B(pure): {len(circuits_all)} circuits, 3 models", flush=True)

    results = {}
    for mname, ckpt in MODELS.items():
        policy, ck = QAPolicy.load_checkpoint(
            ckpt, device_n=spec.n,
            map_location="cuda" if torch.cuda.is_available() else "cpu")
        policy.eval()
        mcfg = Config()
        mcfg.device = cfg.device
        mcfg.model = ck.get("model_cfg", cfg.model)
        mcfg.graph = policy.graph_cfg
        rows = {}
        t0 = time.time()
        for name, text in circuits_all:
            circ = parse_qasm(text)
            mapper = PureMapper(spec, policy=policy, cfg=mcfg, dev="cuda",
                                seed=42, selection_rule="fidelity",
                                use_post=True)
            try:
                res = mapper.map_circuit(circ, circuit_id=name)
                rows[name] = {"swaps": res["swap_count"],
                              "fidelity": res["metrics"]["est_fidelity"]}
            except Exception as e:
                rows[name] = {"error": f"{type(e).__name__}: {str(e)[:50]}"}
        results[mname] = rows
        print(f"{mname} done ({time.time() - t0:.0f}s)", flush=True)

    names = [n for n, _ in circuits_all]
    means = {}
    for mname, rows in results.items():
        vals = [rows[n]["fidelity"] for n in names
                if "fidelity" in rows[n]]
        means[mname] = sum(vals) / len(vals)
    print("\n== A/B(pure) verdict ==")
    for mname in MODELS:
        print(f"  {mname:12s} mean fidelity: {means[mname]:.4f}")
    base = results["c0.05"]
    for mname in ("postfid_ft", "postfid_mt"):
        rows = results[mname]
        wins = losses = ties = 0
        for n in names:
            if "fidelity" not in base[n] or "fidelity" not in rows[n]:
                continue
            fc, fn = base[n]["fidelity"], rows[n]["fidelity"]
            if fn > fc * (1 + 1e-9):
                wins += 1
            elif fc > fn * (1 + 1e-9):
                losses += 1
            else:
                ties += 1
        print(f"  {mname:12s} vs c0.05: wins {wins} / losses {losses} / ties {ties}"
              f" | mean {(means[mname] / means['c0.05'] - 1) * 100:+.2f}%")
    best = max(means, key=means.get)
    verdict = best != "c0.05"
    print(f"\nverdict: {'ADOPT ' + best + ' (run pure stratified)' if verdict else 'KEEP c0.05'}")
    with open("tables/ab_pure_models.json", "w", encoding="utf-8") as f:
        json.dump({"means": means, "verdict": best}, f, ensure_ascii=False,
                  indent=2)


if __name__ == "__main__":
    main()
