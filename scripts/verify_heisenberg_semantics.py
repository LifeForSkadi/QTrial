# -*- coding: utf-8 -*-
"""heisenberg 产物语义验证：算子重叠（1.0 = 等价到全局相位/置换）。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm


def qiskit_from_ops(ops, n):
    qc = QuantumCircuit(n)
    for inst in ops:
        name = inst.name
        qs = [int(q) for q in inst.qubits]
        if name == 'rz':
            qc.rz(inst.params[0], qs[0])
        elif name == 'sx':
            qc.sx(qs[0])
        elif name == 'x':
            qc.x(qs[0])
        elif name == 'cz':
            qc.cz(qs[0], qs[1])
        elif name == 'u3':
            qc.u(inst.params[0], inst.params[1], inst.params[2], qs[0])
        elif name == 'u1':
            qc.p(inst.params[0], qs[0])
        elif name == 'cx':
            qc.cx(qs[0], qs[1])
        elif name == 'h':
            qc.h(qs[0])
        elif name == 'swap':
            qc.swap(qs[0], qs[1])
    return qc


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(
        'checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt',
        device_n=spec.n, map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get('model_cfg', cfg.model)
    cfg.graph = policy.graph_cfg

    circ = parse_qasm(open(
        'data/benchpress/qasm/square-heisenberg/'
        'square_heisenberg_N16.qasm', encoding='utf-8').read())

    torch.manual_seed(42)
    mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                        selection_rule='fidelity', use_post=True,
                        pipeline='v5', routing_seeds=16)
    t0 = time.time()
    res = mapper.map_circuit(circ, circuit_id='h16')
    qc_final = mapper._full_result[2]
    print(f'产物比特数={qc_final.n} 原始={circ.n} '
          f'fid={res["metrics"]["est_fidelity"]:.4e} '
          f'wall={time.time()-t0:.0f}s')

    if qc_final.n == circ.n:
        op_orig = Operator(qiskit_from_ops(circ.ops, circ.n)).data
        op_final = Operator(qiskit_from_ops(qc_final.ops, qc_final.n)).data
        # 置换容差：对 16 比特用"最大重叠置换"太贵——
        # 先比较无置换的重叠，再与 relabel 的期望置换核对
        fid = abs(np.trace(op_orig.conj().T @ op_final)) / (2 ** circ.n)
        print(f'算子重叠（无置换补偿）: {fid:.6f}')
        # 若 <0.99：尝试 final_layout 的逆置换
        if fid < 0.99 and mapper._full_result[1] is not None:
            fin = mapper._full_result[1]
            perm = [fin.get(i, i) for i in range(circ.n)]
            P = np.zeros((2 ** circ.n, 2 ** circ.n), dtype=complex)
            for x in range(2 ** circ.n):
                y = 0
                for i in range(circ.n):
                    if (x >> i) & 1:
                        y |= 1 << perm[i]
                P[y, x] = 1
            op_perm = P @ op_final
            fid2 = abs(np.trace(op_orig.conj().T @ op_perm)) / (2 ** circ.n)
            print(f'算子重叠（final_layout 逆置换补偿）: {fid2:.6f}')
    else:
        print('产物比特数 != 16——完整等价验证需布局映射（另测）')


if __name__ == '__main__':
    main()
