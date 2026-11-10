# -*- coding: utf-8 -*-
"""h16 的 1Q 差距结构诊断：我们的产物 vs RL+O3 产物（同布局同口径）
统计：总 1Q、rz/sx/x 构成、H 块（rz(π/2)-sx-rz(π/2)）数、1Q 游程长度分布。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

from qiskit import qasm2, transpile

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pipeline.baselines import build_target
from qtrail.pipeline.mapper import Mapper as QiskitMapper
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.utils.qasm_io import PLATFORM_BASIS, load_qasm2, strip_measurements

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"


def stats(circ):
    ops = circ.ops
    n1q = sum(1 for i in ops if i.nq == 1)
    n_rz = sum(1 for i in ops if i.name == "rz")
    n_sx = sum(1 for i in ops if i.name == "sx")
    n_x = sum(1 for i in ops if i.name == "x")
    n_h = 0
    i = 0
    while i < len(ops) - 2:
        if (ops[i].name == "rz" and ops[i + 1].name == "sx"
                and ops[i + 2].name == "rz"
                and ops[i].qubits == ops[i + 1].qubits
                and ops[i + 2].qubits == ops[i + 1].qubits
                and abs(abs(ops[i].params[0]) - math.pi / 2) < 1e-9
                and abs(abs(ops[i + 2].params[0]) - math.pi / 2) < 1e-9):
            n_h += 1
            i += 3
        else:
            i += 1
    # 1Q 游程长度分布（cz 之间的同比特连续 1Q）
    runs = []
    cur = 0
    last = None
    for op in ops:
        if op.nq == 1:
            if last == op.qubits:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur, last = 1, op.qubits
        else:
            if cur:
                runs.append(cur)
            cur, last = 0, None
    if cur:
        runs.append(cur)
    from collections import Counter
    return (f'1Q={n1q} rz={n_rz} sx={n_sx} x={n_x} H块={n_h} '
            f'游程分布={dict(sorted(Counter(runs).items()))}')


def main():
    import torch
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib
    target = build_target(spec)
    path = ('data/benchpress/qasm/square-heisenberg/'
            'square_heisenberg_N16.qasm')

    # 我们的侧（与 bar 同口径）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg
    circ = parse_qasm(Path(path).read_text(encoding="utf-8"))
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id="h16")
    qc_ours = mapper._full_result[2]
    print(f'ours: {stats(qc_ours)} ({time.time()-t0:.0f}s)', flush=True)

    # qiskit 侧（与 rl_o3 同口径）
    policy2, ck2 = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                            map_location="cpu")
    policy2.eval()
    cfg2 = Config()
    cfg2.device = load_device_config()
    cfg2.model = ck2.get("model_cfg", cfg2.model)
    cfg2.graph = policy2.graph_cfg
    qmapper = QiskitMapper(spec, policy=policy2, cfg=cfg2, dev="cpu",
                           seed=42)
    qc = load_qasm2(Path(path))
    qc_clean, _ = strip_measurements(qc)
    res2 = qmapper.map_circuit(qc_clean, circuit_id="h16",
                               optimization_level=1)
    init = {qc_clean.qubits[j]: int(res2.layout[j])
            for j in range(qc_clean.num_qubits)}
    routed = transpile(qc_clean, target=target, basis_gates=PLATFORM_BASIS,
                       optimization_level=3, initial_layout=init,
                       seed_transpiler=42)
    opsc = routed.count_ops()
    n1q = sum(v for k, v in opsc.items() if k in ("rz", "sx", "x"))
    print(f'qiskit: twoq={opsc.get("cz", 0)} 1q={n1q} '
          f'detail={dict(opsc)}', flush=True)


if __name__ == "__main__":
    main()
