# -*- coding: utf-8 -*-
"""生产全路径权威验证（评委口径）：
① 端到端语义：原线路嵌入初始布局（RL 布局）的 Statevector
   == mapper 最终产物的 Statevector（同物理帧、N=max+1、闲置 |0⟩）
② 真机合法性：产物门集 ⊆ {rz,sx,x,cz}、cz 端点相邻、无禁用比特、
   比特 < spec.n、无残留 swap
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qtrail.config import Config, load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.models import QAPolicy
from qtrail.pure.mapper import PureMapper
from qtrail.pure.qasm import parse_qasm
from qtrail.pure.circuit import Inst

CKPT = "checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt"
CASES = [("h16", "data/benchpress/qasm/square-heisenberg/"
          "square_heisenberg_N16.qasm"),
         ("h25", "data/benchpress/qasm/square-heisenberg/"
          "square_heisenberg_N25.qasm")]


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


def check_legality(final_qc, spec):
    problems = []
    names = set()
    disabled = set(spec.disabled_qubits)
    for idx, inst in enumerate(final_qc.ops):
        names.add(inst.name)
        qs = [int(q) for q in inst.qubits]
        if inst.name == 'swap':
            problems.append(f'idx={idx} 残留 swap{qs}')
        if any(q >= spec.n for q in qs):
            problems.append(f'idx={idx} 比特越界 {qs}')
        if any(q in disabled for q in qs):
            problems.append(f'idx={idx} 禁用比特 {qs}')
        if inst.name == 'cz' and not spec.adj[qs[0], qs[1]]:
            problems.append(f'idx={idx} cz 非相邻 {qs}')
    if not names <= {'rz', 'sx', 'x', 'cz', 'measure'}:
        problems.append(f'非平台门集: {names}')
    return problems


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_tianyan287_spec(load_device_config())
    policy, ck = QAPolicy.load_checkpoint(CKPT, device_n=spec.n,
                                          map_location=dev)
    policy.eval()
    cfg = Config()
    cfg.device = load_device_config()
    cfg.model = ck.get("model_cfg", cfg.model)
    cfg.graph = policy.graph_cfg

    for tag, path in CASES:
        circ = parse_qasm(Path(path).read_text(encoding="utf-8"))
        torch.manual_seed(42)
        mapper = PureMapper(spec, policy=policy, cfg=cfg, dev=dev, seed=42,
                            selection_rule="fidelity", use_post=True,
                            pipeline="v5", routing_seeds=16)
        t0 = time.time()
        res = mapper.map_circuit(circ, circuit_id=tag)
        final_qc = res["routed_circuit"]
        layout = res["layout"]
        m = res["metrics"]
        print(f'{tag}: twoq={m["twoq_count"]} depth={m["depth"]} '
              f'fid={m["est_fidelity"]:.4e} ({time.time()-t0:.0f}s)',
              flush=True)

        # ① 真机合法性
        probs = check_legality(final_qc, spec)
        print(f'  合法性: {"通过 ✓" if not probs else "失败 ✗ " + str(probs[:5])}',
              flush=True)

        # ② 端到端语义（紧凑子空间法，2026-11-13）：最终产物只触碰
        # k 个不同物理比特——压缩映射到 0..k-1；原线路经
        # final_layout（逻辑→最终物理位）嵌入同一映射。全管线语义
        # 等价 ⟺ 两者在 k 比特子空间上的 Statevector 相等（其余
        # 比特恒等 |0⟩）。覆盖 RL 布局路径（布局跨度 89 不碍事）。
        fin = res["final_layout"]
        qubits = sorted({int(q) for i in final_qc.ops for q in i.qubits})
        all_q = sorted(set(qubits) | {int(fin.get(i, i))
                                      for i in range(circ.n)})
        cmap = {q: k for k, q in enumerate(all_q)}
        emb = [Inst(i.name, tuple(cmap[int(fin[int(q)])] for q in i.qubits),
                    i.params) for i in circ.ops]
        fin2 = [Inst(i.name, tuple(cmap[int(q)] for q in i.qubits),
                     i.params) for i in final_qc.ops]
        k = len(all_q)
        sv1 = np.array(Statevector(to_qc(emb, k)).data)
        sv2 = np.array(Statevector(to_qc(fin2, k)).data)
        fid = abs(np.vdot(sv1, sv2)) ** 2
        print(f'  端到端语义（紧凑子空间，k={k}）fid={fid:.6f}（应=1）',
              flush=True)


if __name__ == "__main__":
    main()
