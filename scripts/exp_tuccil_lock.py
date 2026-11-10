# -*- coding: utf-8 -*-
"""Tuccil merge 跳过标记消融：heisenberg N16（身份布局路由）上
lock_emissions on/off x resynth_every 1/3 的 1Q/2Q/深度/墙钟对比。

基线（lock off, resynth_every=1）= 1Q 1152（第七次压缩前最优态）。
语义验证：fold_only 恒等式（lock 只改变收集边界，不改语义）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.router import sabre_route
from qtrail.pure.post_v2 import fold_only, decompose_to_platform_v2


def count_1q(ops):
    return sum(1 for i in ops if i.nq == 1 and i.name in ("rz", "sx", "x"))


def main():
    spec = build_tianyan287_spec(load_device_config())
    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())
    layout = {i: i for i in range(circ.n)}
    t0 = time.time()
    routed, swaps, fin = sabre_route(circ, spec, layout, seed=42, trials=1,
                                     lookahead_w=0.0, w_ext=0.0,
                                     w_layer=None, cand_scope='front')
    print(f'路由: {len(routed.ops)} 门 swap={swaps} '
          f'({time.time()-t0:.0f}s)')
    folded, arr = fold_only(routed)
    print(f'fold_only 基准门数={len(folded.ops)}')

    for re in (1, 3):
        for lock in (False, True):
            t0 = time.time()
            out = decompose_to_platform_v2(routed, spec=spec,
                                           resynth_every=re,
                                           lock_emissions=lock)
            n1q = count_1q(out.ops)
            ncz = out.count('cz')
            dep = out.layer_depth()
            nlock = sum(1 for i in out.ops
                        if getattr(i, 'tu_locked', False))
            # fold 恒等式：fold(路由) 与 产物 语义一致（fold 在初始布局
            # 恒等下坐标一致，fold_only 含 swap 语义）
            fold2, _ = fold_only(out)
            n2 = len(fold2.ops)
            print(f're={re} lock={int(lock)}: 1Q={n1q} cz={ncz} '
                  f'total={len(out.ops)} depth={dep} '
                  f'tu_locked={nlock} fold后门数={n2} '
                  f'({time.time()-t0:.0f}s)')
            if re == 3 and lock:
                # 语义权威验证（2026-11-13 修正二）：decompose 把 swap
                # 全部展开，fold(out) 无 swap 可跟踪、坐标帧不归位——
                # 正确做法是 raw routed vs raw out 在同一物理帧裸比
                # （N=max_qubit+1，含 swap 门；路由器空位穿越合法）。
                import numpy as np
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector

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

                mq = max((max(int(q) for q in i.qubits)
                          for i in routed.ops + out.ops), default=0)
                nq = max(circ.n, mq + 1)
                sv1 = np.array(Statevector(to_qc(routed.ops, nq)).data)
                sv2 = np.array(Statevector(to_qc(out.ops, nq)).data)
                fid = abs(np.vdot(sv1, sv2)) ** 2
                print(f'  lock=1 re=3 语义 fid={fid:.6f}（应=1）')


if __name__ == '__main__':
    main()
