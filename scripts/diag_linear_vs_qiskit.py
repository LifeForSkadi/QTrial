# -*- coding: utf-8 -*-
"""decompose_linear 差距解剖（2026-11-12）：同一 decompose 输入 dump 上
三条路径的结构对比——A=fixpoint（生产）、B=linear（补充四十新架构）、
Q=qiskit O3 后处理（补充二十六决胜实验的同款做法）。

解剖问题：B 比 A 多 3.6k 1Q 的来源（孤立 H 块？未合并 u3 对？）；
Q 的输出结构（游程是否跨 cz？rz 合并率？H 块残留数）——决定
linear 架构缺的机制。
"""
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.config import load_device_config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.post_v2 import (_decompose_ablation, _expand_to_u3cz,
                                 _merge_runs_u3cz, _U3_H)

DUMP_DIR = Path("scripts/tmp_decompose_ab")
TARGETS = ["qv32", "heis16", "qft20"]
_PLATFORM_1Q = {"rz", "sx", "x"}


def runs_of(ops):
    """逐比特游程统计（cz 分段内）：[(qubit, run_len), ...] + 跨段全景。"""
    segs = []      # cz 段内游程
    i = 0
    n = len(ops)
    while i < n:
        if ops[i].nq == 1:
            j = i
            while j < n and ops[j].nq == 1:
                j += 1
            by_q = Counter(g.qubits[0] for g in ops[i:j])
            segs.extend(by_q.values())
            i = j
        else:
            i += 1
    return segs


def h_residue(ops):
    """孤立 H 块（rz(π/2)·sx·rz(π/2) 三元组）计数：B 未吸收的 H。"""
    cnt = 0
    for i in range(len(ops) - 2):
        a, b, c = ops[i], ops[i + 1], ops[i + 2]
        if (a.name == "rz" and b.name == "sx" and c.name == "rz"
                and abs(a.params[0] - np.pi / 2) < 1e-6
                and abs(c.params[0] - np.pi / 2) < 1e-6
                and a.qubits == b.qubits == c.qubits):
            cnt += 1
    return cnt


def qiskit_post(dump, spec, label):
    """dump（物理坐标）→ 压缩映射 → qiskit O3 后处理（纯后处理：
    initial_layout 给定 → VF2 跳过、输入已相邻 → Sabre 零换）。"""
    from qiskit import QuantumCircuit, transpile
    from qtrail.pipeline.baselines import build_target
    touched = sorted({int(q) for i in dump.ops for q in i.qubits})
    remap = {q: i for i, q in enumerate(touched)}
    qc = QuantumCircuit(len(touched))
    for inst in dump.ops:
        if inst.name == "cx":
            qc.cx(remap[int(inst.qubits[0])], remap[int(inst.qubits[1])])
        elif inst.name == "u1":
            qc.rz(inst.params[0], remap[int(inst.qubits[0])])
        elif inst.name == "u3":
            qc.u(*inst.params, remap[int(inst.qubits[0])])
        elif inst.name == "swap":
            qc.swap(remap[int(inst.qubits[0])],
                    remap[int(inst.qubits[1])])
    target = build_target(spec)
    init = {qc.qubits[i]: int(touched[i]) for i in range(len(touched))}
    t0 = time.time()
    out = transpile(qc, target=target,
                    basis_gates=["rz", "sx", "x", "cz"],
                    optimization_level=3, initial_layout=init,
                    seed_transpiler=42)
    c = Counter(i.name for i in out.data)
    n1q = sum(v for k, v in c.items() if k in ("rz", "sx", "x"))
    ncz = c.get("cz", 0) + c.get("cx", 0) + c.get("swap", 0)
    ops = []
    for i in out.data:
        if i.operation.name == "measure":
            continue
        qs = tuple(out.find_bit(q).index for q in i.qubits)
        ops.append((i.operation.name, qs,
                    tuple(float(p) for p in i.operation.params)))
    segs = runs_of_q(ops)
    comp = Counter(o[0] for o in ops)
    hr = sum(1 for i in range(len(ops) - 2)
             if (ops[i][0] == "rz" and ops[i + 1][0] == "sx"
                 and ops[i + 2][0] == "rz"
                 and abs(ops[i][2][0] - np.pi / 2) < 1e-6
                 and abs(ops[i + 2][2][0] - np.pi / 2) < 1e-6
                 and ops[i][1] == ops[i + 1][1] == ops[i + 2][1]))
    print(f"[Q     ] {label}: 1Q={n1q} cz={ncz} wall={time.time()-t0:.1f}s "
          f"构成={dict(comp)} H残块={hr} "
          f"游程={len(segs)} 均值={np.mean(segs):.2f} 最大={max(segs)} "
          f"分布={dict(sorted(Counter(segs).items()))}", flush=True)
    return n1q, ncz, ops


def runs_of_q(ops):
    """qiskit 输出（(name, qubits, params) 元组）的游程统计。"""
    segs = []
    i = 0
    n = len(ops)
    while i < n:
        if len(ops[i][1]) == 1:
            j = i
            by_q = Counter()
            while j < n and len(ops[j][1]) == 1:
                by_q[ops[j][1][0]] += 1
                j += 1
            segs.extend(by_q.values())
            i = j
        else:
            i += 1
    return segs


def linear_instrumented(dump, spec, label):
    """B 路径 + 插桩：merge 前后结构。"""
    from qtrail.pure.circuit import Circuit
    from qtrail.pure.post_v2 import decompose_linear
    t0 = time.time()
    out = decompose_linear(dump, spec=spec)
    wall = time.time() - t0
    n1q = sum(1 for i in out.ops if i.name in _PLATFORM_1Q)
    nrz = sum(1 for i in out.ops if i.name == "rz")
    ncz = out.count("cz")
    segs = runs_of(out.ops)
    hr = h_residue(out.ops)
    print(f"[B+    ] {label}: 1Q={n1q}（rz={nrz}） cz={ncz} wall={wall:.2f}s "
          f"游程={len(segs)} 均值={np.mean(segs):.2f} 最大={max(segs)} "
          f"孤立H={hr} 分布={dict(sorted(Counter(segs).items()))}",
          flush=True)
    # 展开后 merge 前：u3H 总数与孤立 u3H（段内单元素）数
    ops = _expand_to_u3cz(list(dump.ops))
    i, n = 0, len(ops)
    n_h = 0
    while i < n:
        if ops[i].nq == 1:
            j = i
            seg = []
            while j < n and ops[j].nq == 1:
                seg.append(ops[j])
                j += 1
            by_q = {}
            for g in seg:
                by_q.setdefault(g.qubits[0], []).append(g)
            for q, gs in by_q.items():
                for g in gs:
                    if g.name == "u3" and g.params == _U3_H:
                        n_h += 1
            i = j
        else:
            i += 1
    print(f"[B+    ] 展开态 u3H 总数={n_h}", flush=True)
    return n1q, ncz


def main():
    spec = build_tianyan287_spec(load_device_config())
    for label in TARGETS:
        dump_path = DUMP_DIR / f"dump_{label}.pkl"
        if not dump_path.exists():
            print(f"[跳过] {label} 无 dump")
            continue
        with open(dump_path, "rb") as f:
            dump = pickle.load(f)
        print(f"\n== {label}（dump 门={len(dump.ops)}）", flush=True)
        linear_instrumented(dump, spec, label)
        qiskit_post(dump, spec, label)


if __name__ == "__main__":
    main()
