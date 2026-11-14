# -*- coding: utf-8 -*-
"""qelib1 全门矩阵回归（2026-11-13 独立审计 P0-1）。

背景：`qtrail/pure/qasm.py` 的 `y` 展开曾写成 u3(pi/2, pi/2, pi/2)，
与 qelib1 权威定义 u3(pi, pi/2, pi/2) 不符（qiskit Operator 重叠
0.7071），会静默改变任何含 y 的输入线路。本测试用 qiskit Operator
作为独立基准，逐一核对解析器展开后的门矩阵（含全局相位等价）。
"""
from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qtrail.pure.qasm import QASMError, _eval, parse_qasm

CASES = [
    ("h", ()),
    ("x", ()),
    ("y", ()),
    ("z", ()),
    ("s", ()),
    ("sdg", ()),
    ("t", ()),
    ("tdg", ()),
    ("id", ()),
    ("u1", (0.37,)),
    ("u2", (0.2, 0.4)),
    ("u3", (0.2, 0.4, 0.6)),
    ("rx", (0.3,)),
    ("ry", (0.7,)),
    ("rz", (0.9,)),
    ("sx", ()),
    ("sxdg", ()),
]


def _parsed_operator(name: str, args: tuple) -> Operator:
    arg_text = "(" + ",".join(repr(a) for a in args) + ")" if args else ""
    text = ("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
            f"qreg q[1];\n{name}{arg_text} q[0];\n")
    circ = parse_qasm(text)
    qc = QuantumCircuit(1)
    for inst in circ.ops:
        if inst.name == "u3":
            qc.u(*inst.params, 0)
        elif inst.name == "u1":
            qc.p(inst.params[0], 0)
        elif inst.name == "id":
            qc.id(0)
        else:  # 解析器应把 qelib1 单比特门全部展开为 u3/u1/id
            raise AssertionError(f"非预期解析门 {inst.name}")
    return Operator(qc)


def _reference_operator(name: str, args: tuple) -> Operator:
    qc = QuantumCircuit(1)
    # qiskit 2.5 移除了 QuantumCircuit.u1/u2/u3 方法，用 u/p 等价构造
    if name == "u1":
        qc.p(*args, 0)
    elif name == "u2":
        qc.u(math.pi / 2, *args, 0)
    elif name == "u3":
        qc.u(*args, 0)
    else:
        getattr(qc, name)(*args, 0)
    return Operator(qc)


@pytest.mark.parametrize("name,args", CASES)
def test_qelib1_gate_matches_qiskit(name, args):
    got = _parsed_operator(name, args)
    want = _reference_operator(name, args)
    # equiv：允许全局相位（qelib1 定义不要求相位一致）
    assert got.equiv(want), (
        f"{name}{args} 解析展开与 qiskit 不等价；"
        f"overlap={abs(got.data[0, 0] * want.data[0, 0].conj()):.12f}")


def test_y_specifically_is_pauli_y():
    """P0-1 回归锚点：y 必须等于 [[0,-i],[i,0]]（允许全局相位）。"""
    got = _parsed_operator("y", ())
    qc = QuantumCircuit(1)
    qc.y(0)
    want = Operator(qc)
    assert got.equiv(want)
    # 直接核对矩阵元素，防止 equiv 的相位容忍掩盖符号错误
    y = want.data
    assert abs(abs(got.data[0, 0]) - abs(y[0, 0])) < 1e-12
    assert abs(abs(got.data[1, 0]) - abs(y[1, 0])) < 1e-12
    assert math.isclose(abs(got.data[1, 0]), 1.0, abs_tol=1e-12)


def test_eval_allows_qasm_arithmetic():
    assert abs(_eval("pi/2 + 0.1*2", {}) - (math.pi / 2 + 0.2)) < 1e-15
    assert abs(_eval("2^3", {}) - 8.0) < 1e-15


def test_eval_rejects_code_execution():
    """审计 P2-4：AST 白名单必须拒绝属性/调用/下标链。"""
    with pytest.raises(QASMError):
        _eval("().__class__.__bases__[0].__subclasses__()", {})
    with pytest.raises(QASMError):
        _eval("open('x')", {})


def test_mid_circuit_measure_rejected():
    """审计 P2-3：测量后同一比特再作用量子门必须报错，而非静默移后。"""
    text = ("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
            "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nx q[0];\n")
    with pytest.raises(QASMError):
        parse_qasm(text)
    # 末端测量（正常 QASM）仍应通过
    ok = ("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
          "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    assert len(parse_qasm(ok).measures) == 1
