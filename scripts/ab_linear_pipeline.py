# -*- coding: utf-8 -*-
"""decompose_linear 全管线 A/B（2026-11-12，补充四十判据的生产口径）：
同线路同种子跑生产管线 linear_decompose=False/True，对比最终产物
（fid/logf/twoq/depth/墙钟/合法性）——判据：linear 不劣化
（heis16 锚点 = 当前代码基线，qv32 = 判据绝对数）。
"""
import math
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

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
TARGETS = {
    "heis16": "data/benchpress/qasm/square-heisenberg/square_heisenberg_N16.qasm",
    "qv32": "data/benchpress/qasm/qasmbench-large/QV_n32/32.qasm",
    "qft20": "data/benchpress/qasm/qft/qft_N020.qasm",
}
_PLATFORM_1Q = {"rz", "sx", "x"}


def log_fidelity(circ, calib):
    err2q = calib.err_2q
    med = float(sum(float(v) for v in err2q.values())
                / max(len(err2q), 1))
    s = 0.0
    for inst in circ.ops:
        if inst.nq == 2:
            e = err2q.get((inst.qubits[0], inst.qubits[1]),
                          err2q.get((inst.qubits[1], inst.qubits[0]), med))
            s += math.log(max(1.0 - float(e), 1e-12))
        else:
            s += math.log(max(1.0 - float(calib.err_1q[inst.qubits[0]]),
                              1e-12))
    for inst in circ.measures:
        s += math.log(max(1.0 - float(calib.err_ro[inst.qubits[0]]), 1e-12))
    return s


def legality_scan(circ, spec):
    bad = []
    disabled = set(spec.disabled_qubits)
    for inst in circ.ops:
        if inst.name not in _PLATFORM_1Q | {"cz"}:
            bad.append(f"门集:{inst.name}")
        for q in inst.qubits:
            if q >= spec.n or q in disabled:
                bad.append(f"比特:{inst.name}{inst.qubits}")
        if inst.name == "cz":
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if not spec.adj[a, b]:
                bad.append(f"非相邻:{a},{b}")
    return bad


def run_one(name, path, spec, policy, cfg, dev, linear):
    circ = parse_qasm(Path(path).read_text(encoding="utf-8",
                                          errors="replace"))
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16,
                        linear_decompose=linear)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id=name)
    wall = time.time() - t0
    m = res["metrics"]
    rc = res["routed_circuit"]
    lf = log_fidelity(rc, spec.calib)
    bad = legality_scan(rc, spec)
    n1q = sum(1 for i in rc.ops if i.name in _PLATFORM_1Q)
    tag = "linear" if linear else "fixpt "
    print(f"[{tag}] {name}: fid={m['est_fidelity']:.4e} logf={lf:.2f} "
          f"twoq={m['twoq_count']} 1Q={n1q} depth={m['depth']} "
          f"wall={wall:.0f}s 非法={len(bad)}"
          + (f" {bad[:2]}" if bad else ""), flush=True)
    return m, lf, wall, bad, n1q


def main():
    spec = build_tianyan287_spec(load_device_config())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    ok_all = True
    for name, path in TARGETS.items():
        print(f"\n== {name} ==", flush=True)
        m_a, lf_a, w_a, bad_a, n1q_a = run_one(name, path, spec, policy,
                                               cfg, dev, linear=False)
        m_b, lf_b, w_b, bad_b, n1q_b = run_one(name, path, spec, policy,
                                               cfg, dev, linear=True)
        ok = (not bad_b and lf_b >= lf_a - 1e-9
              and m_b["twoq_count"] <= m_a["twoq_count"] + 0.5
              and n1q_b <= n1q_a + 0.5)
        if name == "qv32":
            ok = ok and n1q_b <= 16500 and m_b["depth"] <= 3600
        print(f"[裁决] {name}: fid {m_a['est_fidelity']:.3e}→"
              f"{m_b['est_fidelity']:.3e} logf {lf_a:.2f}→{lf_b:.2f} "
              f"1Q {n1q_a}→{n1q_b} depth {m_a['depth']}→{m_b['depth']} "
              f"→ {'达标' if ok else '不达标'}", flush=True)
        ok_all = ok_all and ok
    print(f"\n全管线总裁决：{'全部达标' if ok_all else '不达标'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
