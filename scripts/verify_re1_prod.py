# -*- coding: utf-8 -*-
"""生产配置（re=1 + 块级择优守卫）验证：h16 门数 + raw-vs-raw 语义 fid。
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
from qtrail.pure.post_v2 import _decompose_ablation


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


def main():
    spec = build_tianyan287_spec(load_device_config())
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}
    routed, swaps, fin = sabre_route(circ, spec, layout, seed=42, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=None, cand_scope='front')
    import time
    t0 = time.time()
    out = _decompose_ablation(routed, spec=spec, resynth_every=1)
    print(f're=1+择优: 1Q={count_1q(out.ops)} cz={out.count("cz")} '
          f'total={len(out.ops)} depth={out.layer_depth()} '
          f'({time.time()-t0:.0f}s)', flush=True)
    mq = max((max(int(q) for q in i.qubits)
              for i in routed.ops + out.ops), default=0)
    nq = max(circ.n, mq + 1)
    sv1 = np.array(Statevector(to_qc(routed.ops, nq)).data)
    sv2 = np.array(Statevector(to_qc(out.ops, nq)).data)
    print(f'语义 fid={abs(np.vdot(sv1, sv2))**2:.6f}（应=1）', flush=True)


if __name__ == '__main__':
    main()
