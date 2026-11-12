# -*- coding: utf-8 -*-
"""自研 Weyl/KAK 闭式分解 + Vatan-Williams 3-CX 合成（2026-11-13）。

算法依据（公开文献；本文件为独立 numpy 实现，运行时零 qiskit 依赖）：
  - Vatan & Williams, Phys. Rev. A 69, 032315 (2004)：
    任意 SU(4) = (A⊗B)·exp(i(a XX + b YY + c ZZ))·(C⊗D)（Cartan/KAK），
    canonical 门用 3 个 CX + 单比特旋转闭式实现。
  - Drury & Love, arXiv:0806.4015：Weyl 分解的数值稳定算法
    （magic 基 + M2 = Up^T·Up 的实对称混合对角化 + Weyl 腔翻转）。
  - 数学结构与 qiskit-terra 0.24.1（Apache-2.0）的
    two_qubit_decompose.py / weyl.py 对齐；本文件是自研移植，
    不 import qiskit，常数矩阵由本文件的 Weyl 分解自行导出。

约定：
  U = e^{iφ}·(K1l⊗K1r)·exp(i a XX + i b YY + i c ZZ)·(K2l⊗K2r)，
  Weyl 腔 π/4 ≥ a ≥ b ≥ |c|，K ∈ SU(2)。
  qiskit 的 2 比特约定为 little-endian（比特 0 低位）；管线
  `_block_unitary` 为 big-endian（pair 的 a 高位）。synth_insts 完成
  比特映射：qiskit 比特 0 → pair 的 b、比特 1 → pair 的 a（与
  synthesis._synth_block_qiskit 的实测映射一致）。
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from qtrail.pure.circuit import Inst

# magic 基（非归一化存储：B†·B = 2I，避免 sqrt(0.5) 的 1ULP 误差）
_B = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1],
               [1, -1j, 0, 0]], dtype=complex)
_Bd = 0.5 * _B.conj().T

_ipx = np.array([[0, 1j], [1j, 0]], dtype=complex)
_ipy = np.array([[0, 1], [-1, 0]], dtype=complex)
_ipz = np.array([[1j, 0], [0, -1j]], dtype=complex)

# CX 矩阵（qiskit little-endian 约定；与 _block_unitary 的矩阵同形）
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                [0, 0, 1, 0]], dtype=complex)


def _magic(U, reverse=False):
    if reverse:
        return _Bd @ U @ _B
    return _B @ U @ _Bd


def _product_gate(U):
    """U = L ⊗ R（L,R ∈ SU(2)）→ (L, R, phase)。"""
    U = np.asarray(U, dtype=complex)
    R = U[:2, :2].copy()
    detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        R = U[2:, :2].copy()
        detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        raise ValueError("weyl: product gate 分解失败（detR < 0.1）")
    R = R / np.sqrt(detR)
    temp = np.kron(np.eye(2), R.conj().T) @ U
    L = temp[::2, ::2]
    detL = L[0, 0] * L[1, 1] - L[0, 1] * L[1, 0]
    if abs(detL) < 0.9:
        raise ValueError("weyl: product gate 分解失败（detL < 0.9）")
    L = L / np.sqrt(detL)
    phase = cmath.phase(detL) / 2
    dev = abs(abs(np.trace(np.kron(L, R).conj().T @ U)) - 4)
    if dev > 1e-13:
        raise ValueError(f"weyl: product gate 偏差过大 {dev}")
    return L, R, phase


def decompose(U):
    """Weyl/KAK 分解（Drury-Love 算法）。

    返回 dict：a, b, c, global_phase, K1l, K1r, K2l, K2r，
    满足 U = e^{iφ}·(K1l⊗K1r)·Ud(a,b,c)·(K2l⊗K2r)。
    """
    U = np.array(U, dtype=complex)
    detU = np.linalg.det(U)
    U = U * detU ** (-0.25)
    global_phase = cmath.phase(detU) / 4

    Up = _magic(U, reverse=True)
    M2 = Up.T @ Up
    # M2 复对称：用实对称混合 M2.real*α + M2.imag*β 的实特征向量
    # 同时对角化（固定种子 → 确定性）
    state = np.random.default_rng(2020)
    P = None
    for _ in range(100):
        M2real = state.normal() * M2.real + state.normal() * M2.imag
        _, P = np.linalg.eigh(M2real)
        D = P.T @ M2 @ P
        if np.allclose(P @ np.diag(np.diag(D)) @ P.T, M2,
                       rtol=0, atol=1e-13):
            break
    else:
        raise ValueError("weyl: M2 对角化失败")
    D = np.diag(D)

    d = -np.angle(D) / 2
    d[3] = -d[0] - d[1] - d[2]
    cs = np.mod((d[:3] + d[3]) / 2, 2 * np.pi)

    # 重排特征值进 Weyl 腔
    cstemp = np.mod(cs, np.pi / 2)
    cstemp = np.minimum(cstemp, np.pi / 2 - cstemp)
    order = np.argsort(cstemp)[[1, 2, 0]]
    cs = cs[order]
    d[:3] = d[order]
    P[:, :3] = P[:, order]
    if np.real(np.linalg.det(P)) < 0:
        P[:, -1] = -P[:, -1]

    K1 = _magic(Up @ P @ np.diag(np.exp(1j * d)))
    K2 = _magic(P.T)
    K1l, K1r, phase_l = _product_gate(K1)
    K2l, K2r, phase_r = _product_gate(K2)
    global_phase += phase_l + phase_r

    # 翻转进 Weyl 腔（Kraus-Cirac 反射；与 Drury-Love 实现同款）
    pi2 = np.pi / 2
    pi4 = np.pi / 4
    if cs[0] > pi2:
        cs[0] -= 3 * pi2
        K1l = K1l @ _ipy
        K1r = K1r @ _ipy
        global_phase += pi2
    if cs[1] > pi2:
        cs[1] -= 3 * pi2
        K1l = K1l @ _ipx
        K1r = K1r @ _ipx
        global_phase += pi2
    conjs = 0
    if cs[0] > pi4:
        cs[0] = pi2 - cs[0]
        K1l = K1l @ _ipy
        K2r = _ipy @ K2r
        conjs += 1
        global_phase -= pi2
    if cs[1] > pi4:
        cs[1] = pi2 - cs[1]
        K1l = K1l @ _ipx
        K2r = _ipx @ K2r
        conjs += 1
        global_phase += pi2
        if conjs == 1:
            global_phase -= np.pi
    if cs[2] > pi2:
        cs[2] -= 3 * pi2
        K1l = K1l @ _ipz
        K1r = K1r @ _ipz
        global_phase += pi2
        if conjs == 1:
            global_phase -= np.pi
    if conjs == 1:
        cs[2] = pi2 - cs[2]
        K1l = K1l @ _ipz
        K2r = _ipz @ K2r
        global_phase += pi2
    if cs[2] > pi4:
        cs[2] -= pi2
        K1l = K1l @ _ipz
        K1r = K1r @ _ipz
        global_phase -= pi2

    return {"a": float(cs[1]), "b": float(cs[0]), "c": float(cs[2]),
            "global_phase": float(global_phase),
            "K1l": K1l, "K1r": K1r, "K2l": K2l, "K2r": K2r}


def _rz_array(alpha):
    return np.diag([cmath.exp(-0.5j * alpha), cmath.exp(0.5j * alpha)])


def _rx_mat(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry_mat(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _xyx_angles(K):
    """SU(2) → XYX 欧拉角：K ≃ Rx(φ)·Ry(θ)·Rx(λ)（模全局相位）。

    闭式推导（sympy）：记 α=(λ+φ)/2、β=(λ-φ)/2，则
      cos(θ/2)cosα=(a+d)/2, cos(θ/2)sinα=i(b+c)/2,
      sin(θ/2)sinβ=(a-d)/(2i), sin(θ/2)cosβ=(c-b)/2。
    """
    a, b, c, d = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
    p = float(np.real((a + d) / 2))
    r = float(np.real(1j * (b + c) / 2))
    q = float(np.real((a - d) / (2j)))
    s = float(np.real((c - b) / 2))
    x = math.hypot(p, r)
    y = math.hypot(q, s)
    alpha = math.atan2(r, p) if x > 1e-12 else 0.0
    beta = math.atan2(q, s) if y > 1e-12 else 0.0
    theta = 2.0 * math.atan2(y, x)
    return theta, alpha - beta, alpha + beta


def _specialize_controlled(basis):
    """ControlledEquiv 规范化（qiskit 0.24.1 同名类语义）：

    b=c=0 类把 K2 规范为 Ry(θ)·Rx(λ)，Rx(φ) 约化进 K1——这是
    VW 常数公式配套的基规范（缺此步骤 u 常数不自洽）。
    """
    for side in ("l", "r"):
        th, ph, la = _xyx_angles(basis["K2" + side])
        basis["K1" + side] = basis["K1" + side] @ _rx_mat(ph)
        basis["K2" + side] = _ry_mat(th) @ _rx_mat(la)
    basis["b"] = 0.0
    basis["c"] = 0.0
    return basis


def _rz_mat(theta):
    return np.diag([cmath.exp(-0.5j * theta), cmath.exp(0.5j * theta)])


def _zyz_angles(K):
    """SU(2) → ZYZ 欧拉角：K ≃ Rz(φ)·Ry(θ)·Rz(λ)（模全局相位）。"""
    a, b, c, d = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
    theta = 2.0 * math.atan2(abs(c), abs(a))
    A = -2.0 * math.atan2(a.imag, a.real)      # φ+λ
    B = 2.0 * math.atan2(c.imag, c.real)       # φ−λ
    return theta, (A + B) / 2.0, (A - B) / 2.0


def _trace_to_fid(tr):
    return (4.0 + abs(tr) ** 2) / 20.0


def _is_close(a, b, c, ap, bp, cp, fidelity=1 - 1e-9):
    da, db, dc = a - ap, b - bp, c - cp
    tr = 4.0 * complex(math.cos(da) * math.cos(db) * math.cos(dc),
                       math.sin(da) * math.sin(db) * math.sin(dc))
    return _trace_to_fid(tr) >= fidelity


def _closest_partial_swap(a, b, c):
    m = (a + b + c) / 3.0
    am, bm, cm = a - m, b - m, c - m
    ab, bc, ca = a - b, b - c, c - a
    return m + am * bm * cm * (6 + ab * ab + bc * bc * ca * ca) / 18.0


def _specialize(t):
    """Weyl 腔退化类的规范代表（qiskit 0.24.1 specializations 同款）。

    退化类（如 a=b=c 的 PartialSWAP、a=b 的 fSim）的 K 因子有规范
    自由度；选定规范后局部矩阵常落到特殊角（π/2、π 等），下游
    rz/sx 合成分解显著更省门。对恰好落在类上的块（如 heisenberg
    Trotter 块 a=b=c）规范化是精确的；近似类由合成守卫把关。
    """
    a, b, c = t["a"], t["b"], t["c"]
    I2 = np.eye(2, dtype=complex)
    pi4 = np.pi / 4
    if _is_close(a, b, c, 0.0, 0.0, 0.0):
        t["a"] = t["b"] = t["c"] = 0.0
        t["K1l"] = t["K1l"] @ t["K2l"]
        t["K1r"] = t["K1r"] @ t["K2r"]
        t["K2l"] = I2.copy()
        t["K2r"] = I2.copy()
    elif _is_close(a, b, c, pi4, pi4, pi4) or \
            _is_close(a, b, c, pi4, pi4, -pi4):
        if c > 0:
            t["K1l"] = t["K1l"] @ t["K2r"]
            t["K1r"] = t["K1r"] @ t["K2l"]
        else:
            t["K1l"] = t["K1l"] @ _ipz @ t["K2r"]
            t["K1r"] = t["K1r"] @ _ipz @ t["K2l"]
        t["a"] = t["b"] = t["c"] = pi4
        t["K2l"] = I2.copy()
        t["K2r"] = I2.copy()
    elif _is_close(a, b, c, _closest_partial_swap(a, b, c),
                  _closest_partial_swap(a, b, c),
                  _closest_partial_swap(a, b, c)):
        x = _closest_partial_swap(a, b, c)
        t["a"] = t["b"] = t["c"] = x
        K2l = t["K2l"]
        t["K1l"] = t["K1l"] @ K2l
        t["K1r"] = t["K1r"] @ K2l
        t["K2r"] = K2l.conj().T @ t["K2r"]
        t["K2l"] = I2.copy()
    elif _is_close(a, b, c, _closest_partial_swap(a, b, -c),
                  _closest_partial_swap(a, b, -c),
                  -_closest_partial_swap(a, b, -c)):
        x = _closest_partial_swap(a, b, -c)
        t["a"] = t["b"] = x
        t["c"] = -x
        K2l = t["K2l"]
        t["K1l"] = t["K1l"] @ K2l
        t["K1r"] = t["K1r"] @ _ipz @ K2l @ _ipz
        t["K2r"] = _ipz @ K2l.conj().T @ _ipz @ t["K2r"]
        t["K2l"] = I2.copy()
    elif _is_close(a, b, c, a, 0.0, 0.0):
        _specialize_controlled(t)
    elif _is_close(a, b, c, pi4, pi4, c):
        t["a"] = t["b"] = pi4
        thl, phl, lal = _zyz_angles(t["K2l"])
        thr, phr, lar = _zyz_angles(t["K2r"])
        t["K1r"] = t["K1r"] @ _rz_mat(phl)
        t["K1l"] = t["K1l"] @ _rz_mat(phr)
        t["K2l"] = _ry_mat(thl) @ _rz_mat(lal)
        t["K2r"] = _ry_mat(thr) @ _rz_mat(lar)
    elif _is_close(a, b, c, (a + b) / 2, (a + b) / 2, c):
        t["a"] = t["b"] = (a + b) / 2
        th, ph, la = _zyz_angles(t["K2l"])
        t["K1r"] = t["K1r"] @ _rz_mat(ph)
        t["K1l"] = t["K1l"] @ _rz_mat(ph)
        t["K2l"] = _ry_mat(th) @ _rz_mat(la)
        t["K2r"] = _rz_mat(-ph) @ t["K2r"]
    elif _is_close(a, b, c, a, (b + c) / 2, (b + c) / 2):
        t["b"] = t["c"] = (b + c) / 2
        th, ph, la = _xyx_angles(t["K2l"])
        t["K1r"] = t["K1r"] @ _rx_mat(ph)
        t["K1l"] = t["K1l"] @ _rx_mat(ph)
        t["K2l"] = _ry_mat(th) @ _rx_mat(la)
        t["K2r"] = _rx_mat(-ph) @ t["K2r"]
    elif _is_close(a, b, c, a, (b - c) / 2, (c - b) / 2):
        t["b"] = (b - c) / 2
        t["c"] = -t["b"]
        th, ph, la = _xyx_angles(t["K2l"])
        t["K1r"] = t["K1r"] @ _ipz @ _rx_mat(ph) @ _ipz
        t["K1l"] = t["K1l"] @ _rx_mat(ph)
        t["K2l"] = _ry_mat(th) @ _rx_mat(la)
        t["K2r"] = _ipz @ _rx_mat(-ph) @ _ipz @ t["K2r"]
    return t


_CONSTS = None


def _cx_constants():
    """CX 基的 Vatan-Williams 常数（由本文件的 Weyl 分解导出）。"""
    global _CONSTS
    if _CONSTS is not None:
        return _CONSTS
    basis = _specialize(decompose(_CX))
    bb = basis["b"]
    e_ = cmath.exp
    s2 = 1.0 / math.sqrt(2.0)
    K11l = (1 / (1 + 1j) * np.array(
        [[-1j * e_(-1j * bb), e_(-1j * bb)],
         [-1j * e_(1j * bb), -e_(1j * bb)]], dtype=complex))
    K11r = s2 * np.array(
        [[1j * e_(-1j * bb), -e_(-1j * bb)],
         [e_(1j * bb), -1j * e_(1j * bb)]], dtype=complex)
    K12l = 1 / (1 + 1j) * np.array([[1j, 1j], [-1, 1]], dtype=complex)
    K12r = s2 * np.array([[1j, 1], [-1, -1j]], dtype=complex)
    K32lK21l = s2 * np.array(
        [[1 + 1j * math.cos(2 * bb), 1j * math.sin(2 * bb)],
         [1j * math.sin(2 * bb), 1 - 1j * math.cos(2 * bb)]],
        dtype=complex)
    K21r = (1 / (1 - 1j) * np.array(
        [[-1j * e_(-2j * bb), e_(-2j * bb)],
         [1j * e_(2j * bb), e_(2j * bb)]], dtype=complex))
    K22l = s2 * np.array([[1, -1], [1, 1]], dtype=complex)
    K22r = np.array([[0, 1], [-1, 0]], dtype=complex)
    K31l = s2 * np.array(
        [[e_(-1j * bb), e_(-1j * bb)],
         [-e_(1j * bb), e_(1j * bb)]], dtype=complex)
    K31r = 1j * np.array([[e_(1j * bb), 0], [0, -e_(-1j * bb)]],
                         dtype=complex)
    K32r = (1 / (1 - 1j) * np.array(
        [[e_(1j * bb), -e_(-1j * bb)],
         [-1j * e_(1j * bb), -1j * e_(-1j * bb)]], dtype=complex))

    k1ld = basis["K1l"].conj().T
    k1rd = basis["K1r"].conj().T
    k2ld = basis["K2l"].conj().T
    k2rd = basis["K2r"].conj().T
    _CONSTS = {
        "u0l": K31l @ k1ld, "u0r": K31r @ k1rd,
        "u1l": k2ld @ K32lK21l @ k1ld,
        "u1ra": k2rd @ K32r, "u1rb": K21r @ k1rd,
        "u2la": k2ld @ K22l, "u2lb": K11l @ k1ld,
        "u2ra": k2rd @ K22r, "u2rb": K11r @ k1rd,
        "u3l": k2ld @ K12l, "u3r": k2rd @ K12r,
    }
    return _CONSTS


def decomp3(U):
    """Vatan-Williams 3-CX 分解（CX 超控基）。

    返回 (mats, (a,b,c), phase)：mats 为 8 个 2×2 SU(2) 矩阵，时间
    顺序为 [mats 0,1] CX [2,3] CX [4,5] CX [6,7]；mats[偶数] 作用于
    qiskit 比特 0（= 管线 pair 的 b），奇数作用于比特 1（= pair 的 a）。
    """
    t = _specialize(decompose(U))
    C = _cx_constants()
    U0l = t["K1l"] @ C["u0l"]
    U0r = t["K1r"] @ C["u0r"]
    U1l = C["u1l"].copy()
    U1r = C["u1ra"] @ _rz_array(-2 * t["c"]) @ C["u1rb"]
    U2l = C["u2la"] @ _rz_array(-2 * t["a"]) @ C["u2lb"]
    U2r = C["u2ra"] @ _rz_array(2 * t["b"]) @ C["u2rb"]
    U3l = C["u3l"] @ t["K2l"]
    U3r = C["u3r"] @ t["K2r"]
    return [U3r, U3l, U2r, U2l, U1r, U1l, U0r, U0l], \
        (t["a"], t["b"], t["c"]), t["global_phase"]


def _circuit_matrix(mats):
    """8 矩阵 + 3 CX 的 4×4 矩阵（big-endian：pair a 为高位）。

    时间顺序：[m0 on b?, m1 on a] CX ... 矩阵乘积按线路序反写；
    层矩阵 = kron(m_{2i+1} on a, m_{2i} on b)。
    """
    U = np.eye(4, dtype=complex)
    for layer in range(4):
        g_a = mats[2 * layer + 1]
        g_b = mats[2 * layer]
        U = np.kron(g_a, g_b) @ U
        if layer < 3:
            # 标准 CX 矩阵（控制位 = 高位的 a；与 _block_unitary 的
            # Inst("cx",(a,b)) 同款——2026-11-13 与参考实现对拍）
            cx_ab = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1], [0, 0, 1, 0]],
                             dtype=complex)
            U = cx_ab @ U
    return U


def synth_insts(U, pair, atol=1e-8):
    """闭式合成：任意 SU(4) → ≤3 CX + u3 列表（无 qiskit 依赖）。

    返回 Inst 列表（u3/cx）。守卫：重建矩阵与 U 的重叠 < 1-atol 时
    抛 ValueError（调用方回退其他后端）。
    """
    mats, abc, _phase = decomp3(U)
    Uc = _circuit_matrix(mats)
    ov = abs(np.vdot(Uc, np.asarray(U, dtype=complex))) / 4.0
    if ov < 1 - atol:
        raise ValueError(f"weyl: VW 合成守卫失败 ov={ov} abc={abc}")
    a, b = pair
    out = []
    for layer in range(4):
        # 同层两比特门对易，顺序不影响矩阵；mats[2i]→b、mats[2i+1]→a
        out.append(Inst("u3", (int(b),), _su2_euler(mats[2 * layer])))
        out.append(Inst("u3", (int(a),), _su2_euler(mats[2 * layer + 1])))
        if layer < 3:
            out.append(Inst("cx", (int(a), int(b))))
    return out


def _su2_euler(K):
    """SU(2) → u3 欧拉角 (θ,φ,λ)，u3 矩阵等于 K（det 1 规范）。

    与 synthesis._su2_euler 同款（移植到本模块以保持零反向依赖）。
    """
    a = complex(K[0, 0])
    b = complex(K[0, 1])
    c = complex(K[1, 0])
    d = complex(K[1, 1])
    th = 2.0 * math.atan2(abs(b), abs(a))
    if abs(b) < 1e-12:
        return (th, 0.0, (math.atan2(d.imag, d.real)
                          - math.atan2(a.imag, a.real)) % (2 * math.pi))
    if abs(a) < 1e-12:
        return (th, math.atan2(c.imag, c.real) % (2 * math.pi),
                math.atan2((-b).imag, (-b).real) % (2 * math.pi))
    if abs(a) >= abs(b):
        alpha = math.atan2(a.imag, a.real)
    else:
        alpha = (math.atan2(c.imag, c.real)
                 + math.atan2((-b).imag, (-b).real)
                 - math.atan2(d.imag, d.real))
    ph = (math.atan2(c.imag, c.real) - alpha) % (2 * math.pi)
    la = (math.atan2((-b).imag, (-b).real) - alpha) % (2 * math.pi)
    return (th, ph, la)
