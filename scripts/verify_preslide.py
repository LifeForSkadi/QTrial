# -*- coding: utf-8 -*-
"""解碎片验证：h16 身份路由上
(A) 仅 loop 分解（对照，cz≈199）
(B) preslide + 前置 A2 + loop 分解
(C) 前置 A2 无 preslide + loop 分解（旧 mapper 行为）
门数对比 + (B) 的 raw-vs-raw 语义 fid。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.post_v2 import (
    _preslide_1q, collect_resynthesize_2q_blocks, decompose_to_platform_v2)


def to_qc(ops, n):
    qc = QuantumCircuit(n)
    for inst in ops:
        qs = [int(q) for q in inst.qubits]
        if inst.name == 'rz':
            qc.rz(inst.params[0], qs[0])
        elif inst.name == 'sx':
            qc.sx(qs[0])
        elif inst.name == 'x':
            qc.x(qs[0])
        elif inst.name == 'cz':
            qc.cz(qs[0], qs[1])
        elif inst.name == 'swap':
            qc.swap(qs[0], qs[1])
        elif inst.name == 'cx':
            qc.cx(qs[0], qs[1])
        elif inst.name == 'u3':
            qc.u(*inst.params, qs[0])
        elif inst.name == 'u1':
            qc.p(inst.params[0], qs[0])
    return qc


def count_1q(ops):
    return sum(1 for i in ops if i.nq == 1 and i.name in ("rz", "sx", "x"))


def fid_of(routed, out):
    mq = max((max(int(q) for q in i.qubits)
              for i in routed.ops + out.ops), default=0)
    nq = max(2, mq + 1)
    sv1 = np.array(Statevector(to_qc(routed.ops, nq)).data)
    sv2 = np.array(Statevector(to_qc(out.ops, nq)).data)
    return abs(np.vdot(sv1, sv2)) ** 2


def main():
    import time
    spec = build_tianyan287_spec(load_device_config())
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}
    routed, swaps, fin = sabre_route(circ, spec, layout, seed=42, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=None, cand_scope='front')

    # (A) 仅 loop
    t0 = time.time()
    out_a = decompose_to_platform_v2(routed, spec=spec, resynth_every=1)
    print(f'(A) 仅loop: 1Q={count_1q(out_a.ops)} cz={out_a.count("cz")} '
          f'depth={out_a.layer_depth()} ({time.time()-t0:.0f}s)', flush=True)

    # (C) 前置 A2 无 preslide（旧行为）
    rc = routed.copy()
    t0 = time.time()
    rc = collect_resynthesize_2q_blocks(rc, spec)
    out_c = decompose_to_platform_v2(rc, spec=spec, resynth_every=1)
    print(f'(C) 前置A2无slide: 1Q={count_1q(out_c.ops)} '
          f'cz={out_c.count("cz")} depth={out_c.layer_depth()} '
          f'({time.time()-t0:.0f}s)', flush=True)

    # (B) preslide + 前置 A2 + loop
    rb = routed.copy()
    t0 = time.time()
    _preslide_1q(rb.ops)
    rb = collect_resynthesize_2q_blocks(rb, spec)
    out_b = decompose_to_platform_v2(rb, spec=spec, resynth_every=1)
    print(f'(B) slide+前置A2: 1Q={count_1q(out_b.ops)} '
          f'cz={out_b.count("cz")} depth={out_b.layer_depth()} '
          f'fid={fid_of(routed, out_b):.6f} ({time.time()-t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    main()
