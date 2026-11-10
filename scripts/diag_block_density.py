# -*- coding: utf-8 -*-
"""块级 1Q 密度拆解（2026-11-13）：heisenberg N16 上量化 A2 块合成器
输出 vs qiskit tqp 的块内 1Q 结构。

假设：1Q 差距（736 vs 436）主体 = A2 块模板 1Q 密度（10.2/cz vs 4.4/cz），
而非 1Q 后处理可补。验证 = 统计双方产物 cz 段之间的 1Q 门数分布 + 3-cz
块结构 + H 半块（rz(±π/2)）占比。
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

import torch

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
BP = Path("data/benchpress/qasm")
PI = 3.141592653589793


def seg_stats(ops, label):
    """cz 之间的 1Q 段统计：段内门数分布、每比特游程组成、H 半块占比。"""
    segs = []
    cur = []
    for inst in ops:
        if inst.name == "cz":
            if cur:
                segs.append(cur)
                cur = []
        elif inst.nq == 1 and inst.name in ("rz", "sx", "x"):
            cur.append(inst)
    if cur:
        segs.append(cur)
    # 段内门数分布
    seg_len = Counter(len(s) for s in segs)
    # 段内每比特游程
    runs = []
    h_half = 0
    n_1q = 0
    for s in segs:
        by_q = {}
        for inst in s:
            q = int(inst.qubits[0])
            by_q.setdefault(q, []).append(inst)
            n_1q += 1
            if inst.name == "rz":
                a = inst.params[0] % (2 * PI)
                if abs(a - PI / 2) < 1e-6 or abs(a - 3 * PI / 2) < 1e-6:
                    h_half += 1
        for q, seq in by_q.items():
            names = "".join(
                g[0] + ("(h)" if g[0] == "rz" and g[1]
                        and (abs(g[1][0] % (2 * PI) - PI / 2) < 1e-6
                             or abs(g[1][0] % (2 * PI) - 3 * PI / 2) < 1e-6)
                        else "")
                for g in [(i.name, i.params) for i in seq])
            runs.append(names)
    print(f"[{label}] cz={ops and sum(1 for i in ops if i.name=='cz')} "
          f"1Q={n_1q} 段数={len(segs)} 段长分布={dict(seg_len)}")
    print(f"[{label}] H半块rz(±π/2)数={h_half}")
    rc = Counter(runs)
    print(f"[{label}] 常见游程组成 {rc.most_common(8)}")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev_cfg = load_device_config()
    cfg = Config()
    cfg.device = dev_cfg
    spec = build_tianyan287_spec(dev_cfg)
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    p = None
    for cand in BP.rglob("*.qasm"):
        if "square_heisenberg_N16" in cand.name:
            p = cand
            break
    assert p is not None
    circ = parse_qasm(p.read_text(encoding="utf-8"))

    t0 = time.time()
    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule="fidelity", use_post=True,
                        pipeline="v5", routing_seeds=16)
    res = mapper.map_circuit(circ, circuit_id="heis16")
    ops = res["routed_circuit"].ops
    print(f"管线墙钟 {time.time()-t0:.0f}s")
    seg_stats(ops, "我们产物")

    # qiskit 感知 O3 对照（同口径：build_target + transpile O3，RL 布局）
    from qiskit import transpile
    from qtrail.pipeline.baselines import build_target, PLATFORM_BASIS
    from qtrail.pipeline.mapper import Mapper as QiskitMapper
    from qtrail.utils.qasm_io import load_qasm2, strip_measurements
    t1 = time.time()
    text = p.read_text(encoding="utf-8")
    qc = load_qasm2(p)
    qc_clean, _ = strip_measurements(qc)
    target = build_target(spec)
    qmapper = QiskitMapper(spec, policy=policy, cfg=cfg, dev="cpu",
                           seed=42)
    res_q = qmapper.map_circuit(qc_clean, circuit_id="heis16",
                                optimization_level=1)
    init = {qc_clean.qubits[j]: int(res_q.layout[j])
            for j in range(qc_clean.num_qubits)}
    out = transpile(qc_clean, target=target,
                    basis_gates=PLATFORM_BASIS,
                    optimization_level=3,
                    initial_layout=init,
                    seed_transpiler=42)
    qops = []
    for inst in out.data:
        if inst.operation.name == "rz":
            qops.append(("rz", (float(inst.operation.params[0]),),
                         (out.find_bit(inst.qubits[0]).index,)))
        elif inst.operation.name == "sx":
            qops.append(("sx", (), (out.find_bit(inst.qubits[0]).index,)))
        elif inst.operation.name == "x":
            qops.append(("x", (), (out.find_bit(inst.qubits[0]).index,)))
        elif inst.operation.name == "cz":
            qops.append(("cz", (), (out.find_bit(inst.qubits[0]).index,
                                    out.find_bit(inst.qubits[1]).index,)))
    # 转 Inst 形
    from qtrail.pure.circuit import Inst
    qinsts = []
    for name, params, qs in qops:
        qinsts.append(Inst(name, qs, params))
    print(f"qiskit 管线墙钟 {time.time()-t1:.0f}s")
    seg_stats(qinsts, "qiskit O3")


if __name__ == "__main__":
    main()
