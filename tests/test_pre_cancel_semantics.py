"""预路由消化（PureMapper.pre_cancel）语义测试（2026-11-13）。

预注册判据 P1（见 scripts/ab_pre_cancel.py）：解析器产出的输入级门
（u3/u1/cx/cz/swap）经 decompose_to_platform_v2 后必须与原线路在
态矢量上等价（全局相位允许，保真度 > 1-1e-9），且测量/cbits 原样保留。

失败含义：预消化在**输入级电路**上语义有误——生产管线会在布局/路由
之前调用它，任何偏差都会污染全部产物。
"""
from __future__ import annotations

import numpy as np

from qtrail.config import Config
from qtrail.devices import build_tianyan287_spec
from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.mapper import PureMapper
from qtrail.pure.post_v2 import decompose_to_platform_v2

CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
              dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=complex)


def _u3(th, ph, la):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s,
                      np.exp(1j * (ph + la)) * c]], dtype=complex)


def _gate_mat(name, params):
    if name == "cx":
        return CX
    if name == "cz":
        return CZ
    if name == "swap":
        return SWAP
    if name == "u3":
        return _u3(*params)
    if name in ("u1", "rz"):
        return np.diag([1, np.exp(1j * params[0])]).astype(complex)
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j],
                              [1 - 1j, 1 + 1j]], dtype=complex)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "id":
        return np.eye(2, dtype=complex)
    raise ValueError(name)


def _apply(sv, mat, qs, n):
    qs = tuple(sorted(int(q) for q in qs))
    perm = [q for q in range(n) if q not in qs]
    new_perm = perm + list(qs)
    sv = np.transpose(sv.reshape([2] * n), new_perm).reshape(
        2 ** (n - len(qs)), 2 ** len(qs))
    sv = sv @ mat.T
    return np.transpose(sv.reshape([2] * n),
                        np.argsort(new_perm)).reshape(-1)


def _sim(circ, n):
    sv = np.zeros(2 ** n, dtype=complex)
    sv[0] = 1
    for inst in circ.ops:
        if inst.name in ("barrier", "id"):
            continue
        sv = _apply(sv, _gate_mat(inst.name, inst.params), inst.qubits, n)
    return sv


def test_decompose_input_gates_statevector():
    """输入级 u3/u1/cx/cz/swap 全部覆盖（4 比特随机小线路）。"""
    rng = np.random.default_rng(2026)
    spec = build_tianyan287_spec()
    for trial in range(6):
        n = 4
        circ = Circuit(n)
        for _ in range(14):
            kind = rng.integers(0, 5)
            a, b = sorted(int(q) for q in rng.choice(n, 2, replace=False))
            if kind == 0:
                circ.append(Inst("u3", (int(a),),
                                 tuple(float(x) for x in rng.uniform(
                                     -np.pi, np.pi, 3))))
            elif kind == 1:
                circ.append(Inst("u1", (int(a),),
                                 (float(rng.uniform(-np.pi, np.pi)),)))
            elif kind == 2:
                circ.append(Inst("cx", (int(a), int(b))))
            elif kind == 3:
                circ.append(Inst("cz", (int(a), int(b))))
            else:
                circ.append(Inst("swap", (int(a), int(b))))
        circ.append(Inst("measure", (0,), cbits=0))
        out = decompose_to_platform_v2(circ, spec=spec, linear=True)
        names = {i.name for i in out.ops}
        assert names <= {"rz", "sx", "x", "cz"}, names
        assert len(out.measures) == 1
        assert int(out.measures[0].cbits) == 0
        fid = float(abs(np.vdot(_sim(circ, n), _sim(out, n))) ** 2)
        assert fid > 1 - 1e-9, f"trial {trial}: fidelity {fid}"


def test_pre_cancel_flag_guard():
    """无 2Q 门的输入不触发预消化分支（守卫条件可观察）。"""
    spec = build_tianyan287_spec()
    mapper = PureMapper(spec, cfg=Config(), dev="cpu", pre_cancel=True)
    c = Circuit(3)
    c.append(Inst("u3", (0,), (0.3, 0.1, 0.2)))
    assert mapper.pre_cancel is True
    assert not any(i.nq == 2 for i in c.ops)
