# -*- coding: utf-8 -*-
"""Tuccil 后处理产物语义验证：fold(路由产物) vs 后处理产物
（初始布局恒等下两者同坐标系，直接态矢量对比）。"""
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
from qtrail.pure.post_v2 import fold_only, _decompose_ablation


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
        elif inst.name == 'u3':
            qc.u(*inst.params, qs[0])
        elif inst.name == 'u1':
            qc.p(inst.params[0], qs[0])
        elif inst.name == 'cx':
            qc.cx(qs[0], qs[1])
    return qc


def main():
    spec = build_tianyan287_spec(load_device_config())
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}
    routed, swaps, fin = sabre_route(circ, spec, layout, seed=42, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=None, cand_scope='front')
    folded, arr = fold_only(routed)
    out = _decompose_ablation(routed, spec=spec, resynth_every=1)
    n = circ.n
    sv1 = np.array(Statevector(to_qc(folded.ops, n)).data)
    sv2 = np.array(Statevector(to_qc(out.ops, n)).data)
    fid = abs(np.vdot(sv1, sv2)) ** 2
    print(f'fold(路由) vs 后处理产物 语义 fid={fid:.6f}（应=1）')


if __name__ == '__main__':
    main()
