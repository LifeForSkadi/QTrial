# -*- coding: utf-8 -*-
"""端到端冒烟：新路由器（无禁忌评分）+ v5 管线（A1/A2/A3）在 inj 抽样线路上的
保真度与耗时，routing_seeds 1 vs 3 对比（对照已记录的 inj_fidelity）。
"""
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
from qtrail.pure.qasm import load_qasm_file

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
OUT = Path("tables/compare_models/smoke_router_v7.jsonl")


def find_file(name):
    if name.startswith("benchpress/"):
        parts = name.split("/")
        return Path("data/benchpress/qasm") / parts[1] / f"{parts[2]}.qasm"
    if name.startswith("queko_BIGD/"):
        return Path("data/Queko/BIGD") / f"{name.split('/')[1]}.qasm"
    return None


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dev={dev}", flush=True)
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    rows = [json.loads(l) for l in Path(
        "tables/compare_models/inj_v6_rows.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    by_name = {r["circuit"]: r for r in rows}
    # 小/中/大 + queko 中档各一条
    want = ["benchpress/feynman/barenco_tof_10",
            "benchpress/feynman/adder_8",
            "benchpress/feynman/gf2^7_mult",
            "benchpress/dtc/dtc_100_cx_12345"]
    queko_mid = next((r["circuit"] for r in rows
                      if r["circuit"].startswith("queko_BIGD")
                      and r.get("n") and 20 <= r["n"] <= 60), None)
    if queko_mid:
        want.append(queko_mid)

    for name in want:
        src = find_file(name)
        inj = by_name[name]
        row = {"circuit": name, "n": inj.get("n"),
               "inj_fidelity": inj.get("inj_fidelity")}
        if src is None:
            row["error"] = "file missing"
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        for seeds in (1, 3):
            try:
                torch.manual_seed(42)
                circ = load_qasm_file(src)
                mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev,
                                    seed=42, selection_rule="fidelity",
                                    pipeline="v5", routing_seeds=seeds)
                t0 = time.time()
                res = mapper.map_circuit(circ, circuit_id=name)
                m = res["metrics"]
                row[f"s{seeds}_fid"] = m["est_fidelity"]
                row[f"s{seeds}_depth"] = m["depth"]
                row[f"s{seeds}_swaps"] = res["swap_count"]
                row[f"s{seeds}_twoq"] = m["twoq_count"]
                row[f"s{seeds}_wall"] = round(time.time() - t0, 2)
                print(f"{name} seeds={seeds} fid={m['est_fidelity']:.4g} "
                      f"inj={inj.get('inj_fidelity')} twoq={m['twoq_count']} "
                      f"wall={row[f's{seeds}_wall']}s", flush=True)
            except Exception as e:
                row[f"s{seeds}_error"] = f"{type(e).__name__}: {str(e)[:50]}"
                print(f"{name} seeds={seeds} ERROR: {row[f's{seeds}_error']}",
                      flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
