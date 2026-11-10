# -*- coding: utf-8 -*-
"""Tuccil 端到端贡献消融（修复后代码，h16 身份布局）：
(a) re=1 全组件（Tuccil + 通用 + 择优）
(b) re=1 无 Tuccil（monkeypatch fit_block_tuccil 恒失败 → 通用解）
(c) 无 A2（resynth=0，纯 1Q 循环）
每配置输出 1Q/cz/depth + raw-vs-raw 语义 fid。
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
from qtrail.pure.post_v2 import decompose_to_platform_v2


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

    for tag, kw in (("(a) re=1 全组件", dict(resynth_every=1)),
                    ("(c) 无A2", dict(resynth_every=0))):
        t0 = time.time()
        out = decompose_to_platform_v2(routed, spec=spec, **kw)
        print(f'{tag}: 1Q={count_1q(out.ops)} cz={out.count("cz")} '
              f'depth={out.layer_depth()} fid={fid_of(routed, out):.6f} '
              f'({time.time()-t0:.0f}s)', flush=True)

    # (b) 无 Tuccil：monkeypatch 拟合恒失败
    import qtrail.pure.synthesis as syn
    orig_fit = syn.fit_block_tuccil

    def fail_fit(*a, **k):
        raise RuntimeError("tuccil disabled")

    syn.fit_block_tuccil = fail_fit
    t0 = time.time()
    out = decompose_to_platform_v2(routed, spec=spec, resynth_every=1)
    syn.fit_block_tuccil = orig_fit
    print(f'(b) re=1 无Tuccil: 1Q={count_1q(out.ops)} '
          f'cz={out.count("cz")} depth={out.layer_depth()} '
          f'fid={fid_of(routed, out):.6f} ({time.time()-t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    main()
