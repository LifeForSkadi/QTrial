# -*- coding: utf-8 -*-
"""D1 验证实验：QCIS 发射器 sx/H 展开的矩阵语义核对 + cqlib 转换器对照。

1) 数值验证（numpy，RZ(a)=diag(e^{-ia/2}, e^{ia/2})，与项目自述平台约定一致）：
   - 代码展开 RZ(pi/2)·X·RZ(-pi/2) 到底等于什么（怀疑 = Y，非 SX）
   - 代码 H 展开 RZ(pi/2)·X·RZ(pi/2) 到底等于什么（怀疑 = X，非 H）
   - X2P（X 的 pi/2 脉冲 = exp(-i(pi/2)X/2)）是否恰为 SX
   - 正确恒等式 RZ(pi/2)·SX·RZ(pi/2) = -i·H 复核
2) cqlib 官方转换器对含 sx/x/h 线路的输出令牌（X2P = 平台的 SX 原生指令？）
"""
import math

import numpy as np


def RZ(a):
    return np.diag([math.e ** (-1j * a / 2), math.e ** (1j * a / 2)])


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
SX = np.array([[1, -1j], [-1j, 1]], dtype=complex) / math.sqrt(2)
X2P = np.cos(math.pi / 4) * np.eye(2, dtype=complex) - 1j * math.sin(math.pi / 4) * X


def eq(a, b):
    """模全局相位的酉等价性：min_gamma ||a - e^{ig} b||"""
    if a.shape != b.shape:
        return False
    ov = np.vdot(a, b)
    m = abs(ov)
    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)
    return (m / (n1 * n2)) > 1 - 1e-12


print("== 1) 矩阵验证 ==")
sx_expand = RZ(math.pi / 2) @ X @ RZ(-math.pi / 2)
print(f"RZ(pi/2)·X·RZ(-pi/2) == Y   : {eq(sx_expand, Y)}")
print(f"RZ(pi/2)·X·RZ(-pi/2) == SX  : {eq(sx_expand, SX)}")
print(f"RZ(pi/2)·X·RZ(-pi/2) == X2P : {eq(sx_expand, X2P)}")
print(f"X2P == SX（精确）            : {np.allclose(X2P, SX)}")

h_expand = RZ(math.pi / 2) @ X @ RZ(math.pi / 2)
print(f"RZ(pi/2)·X·RZ(pi/2)  == X   : {eq(h_expand, X)}")
print(f"RZ(pi/2)·X·RZ(pi/2)  == H   : {eq(h_expand, H)}")
print(f"RZ(pi/2)·SX·RZ(pi/2) == H   : {eq(RZ(math.pi / 2) @ SX @ RZ(math.pi / 2), H)}")

# 状态验证（最直白的实验）：|0> 演化
ket0 = np.array([1, 0], dtype=complex)
print(f"SX|0>              = {SX @ ket0}")
print(f"RZ(pi/2)X RZ(-pi/2)|0> = {sx_expand @ ket0}")
print(f"H|0>               = {H @ ket0}")
print(f"RZ(pi/2)X RZ(pi/2)|0>  = {h_expand @ ket0}")

print("\n== 2) cqlib 官方转换器对照 ==")
try:
    from qiskit import QuantumCircuit
    from qiskit import qasm2
    from cqlib.utils.qasm_to_qcis import QasmToQcis

    qc = QuantumCircuit(2)
    qc.sx(0)
    qc.x(1)
    qc.rz(0.5, 0)
    qc.cz(0, 1)
    qc.h(0)
    qc.s(0)
    qasm_str = qasm2.dumps(qc)
    print("--- 输入 QASM ---")
    print(qasm_str)
    qcis = QasmToQcis().convert_to_qcis(qasm_str)
    print("--- cqlib 输出 QCIS ---")
    print(qcis)
except Exception as e:
    print(f"cqlib converter 失败: {type(e).__name__}: {e}")
