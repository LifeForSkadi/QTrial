# -*- coding: utf-8 -*-
"""qv 类 large 层 fid=0.0 机制诊断（2026-11-12）。

乘积模型 Π(1−ε) 在路由后上万门时数值下溢 → 两侧同为 0.0，
胜负率成为伪指标。本脚本对同一线路跑两条管线（自研 v5 /
RL+O3 基线），除乘积 fid 外输出【对数保真度 Σ log(1−ε)】
（数值稳定、同序保序）与机制层指标（swaps/2Q/深度/墙钟/合法性），
判定 large 层损失是真实机制差距还是度量伪影。
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

TARGETS = [
    "data/benchpress/qasm/qasmbench-large/QV_n32/32.qasm",
]


def log_fidelity(circ, calib):
    """Σ log(1−ε)：与乘积 fid 同序、不下溢。"""
    err2q = calib.err_2q
    med = float(torch.tensor(0.0)) if not err2q else \
        float(sum(float(v) for v in err2q.values()) / len(err2q))
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


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    calib = spec.calib

    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    for tp in TARGETS:
        p = Path(tp)
        circ = parse_qasm(p.read_text(encoding="utf-8", errors="replace"))
        print(f"\n== {p.stem} n={circ.n} 门={len(circ.ops)} ==", flush=True)

        # ---- 自研 v5
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                            selection_rule="fidelity", use_post=True,
                            pipeline="v5", routing_seeds=16)
        t0 = time.time()
        try:
            res = mapper.map_circuit(circ, circuit_id=p.stem)
            wall = time.time() - t0
            m = res["metrics"]
            rc = res["routed_circuit"]
            lf = log_fidelity(rc, calib)
            n_sw = sum(1 for i in rc.ops if i.name == "swap")
            print(f"[v5 ] fid={m['est_fidelity']:.3e} logf={lf:.2f} "
                  f"swaps={n_sw} twoq={m['twoq_count']} "
                  f"depth={m['depth']} wall={wall:.0f}s", flush=True)
        except Exception as e:
            print(f"[v5 ] 失败: {type(e).__name__}: {str(e)[:80]}",
                  flush=True)

        # ---- RL + 感知 O3 基线
        from qiskit import qasm2, transpile
        from qtrail.pipeline.baselines import build_target
        from qtrail.pipeline.mapper import Mapper as QiskitMapper
        from qtrail.utils.qasm_io import PLATFORM_BASIS, load_qasm2, \
            strip_measurements
        target = build_target(spec)
        qm = QiskitMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42)
        qc = load_qasm2(p)
        qc_clean, _ = strip_measurements(qc)
        t0 = time.time()
        try:
            rl = qm.map_circuit(qc_clean, circuit_id=p.stem,
                                optimization_level=1)
            init = {qc_clean.qubits[i]: int(rl.layout[i])
                    for i in range(qc_clean.num_qubits)}
            routed = transpile(qc_clean, target=target,
                               basis_gates=PLATFORM_BASIS,
                               optimization_level=3,
                               initial_layout=init, seed_transpiler=42)
            wall = time.time() - t0
            from qtrail.pipeline.metrics import compute_metrics as cm_q
            mq = cm_q(routed, routed.count_ops().get("swap", 0), calib)
            lfq = log_fidelity_q(routed, calib)
            print(f"[o3 ] fid={mq['est_fidelity']:.3e} logf={lfq:.2f} "
                  f"swaps={mq['swap_count']} twoq={mq['twoq_count']} "
                  f"depth={mq['depth']} wall={wall:.0f}s", flush=True)
        except Exception as e:
            print(f"[o3 ] 失败: {type(e).__name__}: {str(e)[:80]}",
                  flush=True)


def log_fidelity_q(qc, calib):
    """qiskit 线路版：与 log_fidelity 同口径（basis 同 rz/sx/x/cz）。"""
    err2q = calib.err_2q
    med = float(sum(float(v) for v in err2q.values()) / max(len(err2q), 1))
    s = 0.0
    for inst in qc.data:
        if inst.operation.name == "measure":
            q = qc.find_bit(inst.qubits[0]).index
            s += math.log(max(1.0 - float(calib.err_ro[q]), 1e-12))
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            e = err2q.get((qs[0], qs[1]), err2q.get((qs[1], qs[0]), med))
            s += math.log(max(1.0 - float(e), 1e-12))
        elif len(qs) == 1:
            s += math.log(max(1.0 - float(calib.err_1q[qs[0]]), 1e-12))
    return s


if __name__ == "__main__":
    main()
