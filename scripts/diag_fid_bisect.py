# -*- coding: utf-8 -*-
"""语义 fid 二分定位：找出 decompose 哪一层破坏语义。
① 微型无 swap 线路全管线（验证 harness）
② heisenberg N16：仅 v4 分解（fixpoint=False, resynth=0）
③ heisenberg N16：不动点循环但无 A2（resynth=0）
④ heisenberg N16：全管线 re=3
全部 raw vs raw 同帧（N=max+1）比 Statevector。
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
from qtrail.pure.circuit import Circuit
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


def fid_between(ca, cb):
    mq = max((max(int(q) for q in i.qubits)
              for i in ca.ops + cb.ops), default=0)
    nq = max(2, mq + 1)
    sv1 = np.array(Statevector(to_qc(ca.ops, nq)).data)
    sv2 = np.array(Statevector(to_qc(cb.ops, nq)).data)
    return abs(np.vdot(sv1, sv2)) ** 2


def main():
    spec = build_tianyan287_spec(load_device_config())

    # ① 微型无 swap 线路：harness 自检
    tiny = Circuit(2)
    tiny.ops = [__import__('qtrail.pure.circuit', fromlist=['Inst'])
                .Inst('rz', (0,), (0.3,)),
                __import__('qtrail.pure.circuit', fromlist=['Inst'])
                .Inst('cx', (0, 1)),
                __import__('qtrail.pure.circuit', fromlist=['Inst'])
                .Inst('rz', (1,), (0.5,)),
                __import__('qtrail.pure.circuit', fromlist=['Inst'])
                .Inst('cx', (0, 1)),
                __import__('qtrail.pure.circuit', fromlist=['Inst'])
                .Inst('rz', (0,), (0.7,))]
    out1 = _decompose_ablation(tiny, spec=spec, resynth_every=3)
    print(f'① tiny 全管线 fid={fid_between(tiny, out1):.6f}（应=1）',
          flush=True)

    # heisenberg N16 路由（同口径）
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}
    routed, swaps, fin = sabre_route(circ, spec, layout, seed=42, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=None, cand_scope='front')

    # ② 仅 v4
    out2 = _decompose_ablation(routed, spec=spec, fixpoint=False)
    print(f'② h16 仅v4 fid={fid_between(routed, out2):.6f}（应=1）',
          flush=True)
    # ③ 循环无 A2
    out3 = _decompose_ablation(routed, spec=spec, resynth_every=0)
    print(f'③ h16 循环无A2 fid={fid_between(routed, out3):.6f}（应=1）',
          flush=True)
    # ④ 全管线
    out4 = _decompose_ablation(routed, spec=spec, resynth_every=3)
    print(f'④ h16 全管线re3 fid={fid_between(routed, out4):.6f}（应=1）',
          flush=True)


if __name__ == '__main__':
    main()
