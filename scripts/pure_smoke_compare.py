"""pure 管线 vs qiskit 支撑版的首轮性能对照（内部参照，不交付）。

线路：QUEKO dense (3,4) 6 条 + MQTBench ≤10 6 条。
pure = PureMapper（fidelity 规则，use_post）；qiskit 版 = target_post opt3
（fidelity 规则，8 种子×top-3）——读取既有结果表对照。
"""
from __future__ import annotations

import json
import re
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

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"


def main():
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ckpt = QAPolicy.load_checkpoint(
        CKPT, device_n=spec.n,
        map_location="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cfg.model = ckpt.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    circuits = []
    for p in iter_queko_files("BIGD"):
        if re.search(r"\.3D1_\.4D2_(0|1|2|3|4|5)", p.name):
            circuits.append((p.stem, p.read_text(encoding="utf-8")))
    # 参考对照：qiskit 版 target_post 的 dense 组结果（queko-dense_rows）
    ref = {}
    ref_path = Path("tables/target_post/seeds8_top3/queko-dense_rows.jsonl")
    if ref_path.exists():
        for line in ref_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ref[r["circuit"]] = r

    print(f"{'circuit':42s} {'pure_s':>6s} {'pure_d':>6s} {'pure_f':>7s} "
          f"{'qsk_s':>6s} {'qsk_f':>7s} {'wall':>5s}")
    for name, text in circuits:
        circ = parse_qasm(text)
        t0 = time.time()
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev="cuda", seed=42,
                            selection_rule="fidelity", use_post=True)
        res = mapper.map_circuit(circ, circuit_id=name)
        r = ref.get(name, {})
        qs = r.get("tp_swaps", "-")
        qf = r.get("tp_fidelity", "-")
        print(f"{name:42s} {res['swap_count']:6d} {res['metrics']['depth']:6d} "
              f"{res['metrics']['est_fidelity']:7.4f} {str(qs):>6s} "
              f"{str(qf):>7s} {time.time() - t0:5.1f}s")


if __name__ == "__main__":
    main()
