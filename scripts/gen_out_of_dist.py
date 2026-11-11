# -*- coding: utf-8 -*-
"""分布外测试线路生成器（2026-11-13，correctness_theorems.md §3 清单）。

7 类 587 之外的线路：Clifford 密集、T 密集、QAOA 变体、VQE 变体、
随机 SU(4) 块、含测量/经典控制、病态结构（超长 1Q 游程/全连通
子图/单双比特）。输出 QASM 到 data/ood/。
"""
import math
import random
from pathlib import Path

OUT = Path("data/ood")


def qasm(n, name, gates):
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{n}];",
             f"creg c[{n}];"]
    lines += gates
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(n)]
    (OUT / f"{name}.qasm").write_text("\n".join(lines), encoding="utf-8")
    return len(gates)


def gen_clifford(n=12, depth=3):
    """随机 Clifford（H/S/CX 密集）。"""
    rng = random.Random(42)
    gates = []
    for _ in range(depth * n):
        i, j = rng.randrange(n), rng.randrange(n)
        k = rng.randrange(3)
        if k == 0:
            gates.append(f"h q[{i}];")
        elif k == 1:
            gates.append(f"s q[{i}];")
        else:
            gates.append(f"cx q[{i}],q[{j}];")
    return qasm(n, "ood_clifford_n12", gates)


def gen_t_dense(n=12, depth=3):
    """T 密集（H/T/CNOT）。"""
    rng = random.Random(43)
    gates = []
    for _ in range(depth * n):
        i, j = rng.randrange(n), rng.randrange(n)
        k = rng.randrange(3)
        if k == 0:
            gates.append(f"h q[{i}];")
        elif k == 1:
            gates.append(f"t q[{i}];")
        else:
            gates.append(f"cx q[{i}],q[{j}];")
    return qasm(n, "ood_t_dense_n12", gates)


def gen_qaoa(n=12, p=3):
    """QAOA p 层（ring 图，随机 β/γ）。"""
    rng = random.Random(44)
    gates = [f"h q[{i}];" for i in range(n)]
    for _ in range(p):
        gamma, beta = rng.uniform(0, math.pi), rng.uniform(0, math.pi)
        for i in range(n):
            gates.append(f"rz({beta}) q[{i}];")
        for i in range(n):
            j = (i + 1) % n
            gates.append(f"cx q[{i}],q[{j}];")
            gates.append(f"rz({gamma}) q[{j}];")
            gates.append(f"cx q[{i}],q[{j}];")
    return qasm(n, "ood_qaoa_p3_n12", gates)


def gen_vqe(n=12, layers=4):
    """VQE 型 ansatz（RY 旋转 + 阶梯 CX 纠缠）。"""
    rng = random.Random(45)
    gates = []
    for _ in range(layers):
        for i in range(n):
            gates.append(f"ry({rng.uniform(0, 2*math.pi):.4f}) q[{i}];")
        for i in range(n - 1):
            gates.append(f"cx q[{i}],q[{i+1}];")
    return qasm(n, "ood_vqe_n12", gates)


def gen_su4_blocks(n=12, nblocks=12):
    """随机 SU(4) 块（KAK 生成：两侧 1Q + CNOT 骨架）。"""
    rng = random.Random(46)
    gates = []
    for _ in range(nblocks):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            j = (j + 1) % n
        for q in (i, j):
            gates.append(f"u3({rng.uniform(0,math.pi):.4f},"
                         f"{rng.uniform(0,2*math.pi):.4f},"
                         f"{rng.uniform(0,2*math.pi):.4f}) q[{q}];")
        gates.append(f"cx q[{i}],q[{j}];")
        gates.append(f"rz({rng.uniform(0,2*math.pi):.4f}) q[{j}];")
        gates.append(f"cx q[{i}],q[{j}];")
    return qasm(n, "ood_su4_blocks_n12", gates)


def gen_measured_feedback(n=8, rounds=6):
    """含中间测量与经典控制的线路（c_if 语义——QASM2 的 if）。"""
    rng = random.Random(47)
    gates = []
    for r in range(rounds):
        for i in range(n):
            gates.append(f"h q[{i}];")
        gates.append(f"measure q[0] -> c[0];")
        for i in range(1, n):
            gates.append(f"if (c==1) rz({rng.uniform(0,math.pi):.4f}) q[{i}];")
    return qasm(n, "ood_measured_feedback_n8", gates)


def gen_pathological():
    """病态结构：超长 1Q 游程、单比特、两比特全连通、长 cz 链。"""
    rng = random.Random(48)
    # 超长 1Q 游程（50 个 rz/sx 交替同比特）
    gates = []
    for k in range(50):
        gates.append(f"rz({rng.uniform(0,2*math.pi):.4f}) q[0];" if k % 2
                     else f"h q[0];")
    qasm(6, "ood_long_1q_run", gates)
    # 单比特线路
    gates = [f"u3({rng.uniform(0,math.pi):.4f},0,0) q[0];" for _ in range(20)]
    qasm(1, "ood_single_qubit", gates)
    # 两比特密集
    gates = []
    for _ in range(30):
        gates.append(f"cx q[0],q[1];")
        gates.append(f"u3({rng.uniform(0,math.pi):.4f},"
                     f"{rng.uniform(0,2*math.pi):.4f},0) q[0];")
    qasm(2, "ood_two_qubit_dense", gates)
    # 全连通子图（6 比特 K6 的 cz 密集）
    gates = []
    for i in range(6):
        for j in range(i + 1, 6):
            gates.append(f"cx q[{i}],q[{j}];")
            gates.append(f"rz({rng.uniform(0,2*math.pi):.4f}) q[{j}];")
    qasm(6, "ood_k6_dense", gates)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    total += gen_clifford()
    total += gen_t_dense()
    total += gen_qaoa()
    total += gen_vqe()
    total += gen_su4_blocks()
    total += gen_measured_feedback()
    gen_pathological()
    print(f"生成完成 → {OUT}（8 个文件）")


if __name__ == "__main__":
    main()
