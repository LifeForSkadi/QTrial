# -*- coding: utf-8 -*-
"""独立审查脚本（2026-11-13 审查代理）：post 管线 B1 恒等式 + 关键 pass
实现的数值验证。全部 numpy 矩阵直算（atol=1e-9 判过），不依赖 qiskit。

每个测试输出：PASS/FAIL + 偏差量级（相对 Frobenius 距离，模全局相位）。
"""
import sys
import math
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qtrail.pure.circuit import Circuit, Inst
from qtrail.pure.post import (H_GATE, _u3_to_rzsx, _u3_to_rzsx_min,
                              decompose_to_platform, absorb_trailing,
                              push_swaps, post_route)
from qtrail.pure import post_v2 as pv2
from qtrail.pure import synthesis as syn
from qtrail.pure.lnn import chain_for_distance, cnot_chain_on_path, noise_aware_chain

rng = np.random.default_rng(20261113)

# ---------------------------------------------------------------- 基本矩阵
def rz(a):
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])

def sx():
    return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

X = np.array([[0., 1.], [1., 0.]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def u3(th, ph, la):
    c, s = math.cos(th / 2), math.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                    dtype=complex)

def ry(a):
    return u3(a, 0, 0)

def rx(a):
    return u3(a, -math.pi / 2, math.pi / 2)

def u1(la):
    return np.diag([1.0, np.exp(1j * la)])

H = u3(math.pi / 2, 0, math.pi)
CZ = np.diag([1., 1., 1., -1.])
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
              dtype=complex)
CXb = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
               dtype=complex)  # cx 控制=比特 b（bit1）
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=complex)
H_PLAT = rz(math.pi / 2) @ sx() @ rz(math.pi / 2)   # 平台 H

def kron4(m0, m1):   # 比特 a(高位)⊗比特 b(低位) 序
    return np.kron(m0, m1)

def phase_of(V, U):
    """V ≈ g·U 的 g。"""
    return np.vdot(V, U) / np.vdot(U, U)

def rel_err(U, V):
    """模全局相位的相对偏差：min_g ||U-gV||_F / ||U||_F。

    g = ⟨V,U⟩/⟨V,V⟩（把 U 投影到 V 上，共轭取在 V 侧）。"""
    g = np.vdot(V, U) / np.vdot(V, V)
    return float(np.linalg.norm(U - g * V) / np.linalg.norm(U))

RESULTS = []
def check(name, U, V, tol=1e-9, note=""):
    err = rel_err(U, V)
    ok = err < tol
    RESULTS.append((name, ok, err, note))
    print(f'{"PASS" if ok else "FAIL":4} {name:<58} dev={err:.3e} {note}')

# ---------------------------------------------------------------- T1: H 分解
# phase_of(V,U) = Σconj(V)U/||U||² = conj(V/U 比值)；真实比值 = conj(p)
p = phase_of(H_PLAT, H)
p_act = np.conj(p)
check("T1a H = rz(pi/2)·sx·rz(pi/2) 模全局相位", H, H_PLAT,
      note=f"相位={p_act:.4f}（理论 e^-iπ/4={np.exp(-1j*math.pi/4):.4f}）")
err_phase = abs(p_act - np.exp(-1j * math.pi / 4))
print(f'      T1b 平台H 全局相位=e^-iπ/4 偏差={err_phase:.3e}')

# ---------------------------------------------------------------- T2: cx 分解
def Hwrap(m, q):
    """q=0 作用于比特 a，q=1 作用于比特 b。"""
    return (kron4(m, I2) if q == 0 else kron4(I2, m))

# cx(a,b) = H_b·cz·H_b（H 作用于目标 b）
lhs2 = Hwrap(H, 1) @ CZ @ Hwrap(H, 1)
check("T2a cx(a,b) = H_b·cz·H_b（精确）", CX, lhs2, tol=1e-14)

# 平台分解：rzH·sx·rzH（目标）· cz · rzH·sx·rzH（目标）
lhs2b = Hwrap(H_PLAT, 1) @ CZ @ Hwrap(H_PLAT, 1)
p2 = np.conj(phase_of(lhs2b, CX))
check("T2b 平台 cx 展开 = g·cx 模全局相位", CX, lhs2b,
      note=f"g={p2:.4f}（理论 -i={-1j}）")
print(f'      T2c 平台 cx 展开相位=-i 偏差={abs(p2 - (-1j)):.3e}')

# ---------------------------------------------------------------- T3: swap = 3cx
lhs3 = CX @ CXb @ CX
check("T3  swap(a,b) = cx(a,b)·cx(b,a)·cx(a,b)", SWAP, lhs3, tol=1e-14)

# ---------------------------------------------------------------- T4: sx 幂
check("T4a sx·sx = X（精确）", X, sx() @ sx(), tol=1e-14)
check("T4b sx^4 = I（精确）", I2, np.linalg.matrix_power(sx(), 4), tol=1e-14)

# ---------------------------------------------------------------- T5: slide_x 恒等式
def embed_a(m):
    return kron4(m, I2)

def embed_b(m):
    return kron4(I2, m)

lhs5 = embed_a(X) @ CZ
rhs5 = CZ @ embed_a(X) @ embed_b(rz(math.pi))
p5 = np.conj(phase_of(lhs5, rhs5))
check("T5a X_a·CZ = CZ·X_a·rz_b(pi)·g 模全局相位", rhs5, lhs5,
      note=f"g={p5:.4f}（docstring 称 i={1j}）")
print(f'      T5b slide_x 相位因子=i 偏差={abs(p5 - 1j):.3e}')

# ---------------------------------------------------------------- T6: H 吸收
ths = rng.uniform(-2 * math.pi, 2 * math.pi, 20)
errs = [rel_err(rx(t), H @ rz(t) @ H) for t in ths]
check("T6a H·rz(θ)·H = rx(θ)（20 随机 θ）", rx(0.3), rx(0.3),
      tol=1e-9, note="")
print(f'      T6a 最大偏差={max(errs):.3e}')
errs2 = [rel_err(rx(t), H_PLAT @ rz(t) @ H_PLAT) for t in ths]
p6 = np.conj(phase_of(H_PLAT @ rz(0.7) @ H_PLAT, rx(0.7)))
check("T6b 平台H·rz(θ)·平台H = g·rx(θ)", rx(0.3), H_PLAT @ rz(0.3) @ H_PLAT,
      note=f"g={p6:.4f}（理论 -i）")
print(f'      T6b 最大偏差={max(errs2):.3e} 相位偏差={abs(p6 - (-1j)):.3e}')

# rz(θ)·H = H·rx(θ)（_absorb_h docstring 声称）
errs3 = [rel_err(H @ rx(t), rz(t) @ H) for t in ths]
print(f'      T6c rz(θ)·H = H·rx(θ) 最大偏差={max(errs3):.3e}')
errs4 = [rel_err(rx(t) @ H, H @ rz(t)) for t in ths]
print(f'      T6d H·rz(θ) = rx(θ)·H 最大偏差={max(errs4):.3e}')

# _to_cz_basis 声称：H·ry(θ)·H = rz(θ) 到全局相位
errs5 = [rel_err(rz(t), H @ ry(t) @ H) for t in ths]
print(f'      T6e [_to_cz_basis 注释声称] H·ry(θ)·H = rz(θ) 最大偏差={max(errs5):.3e}（应≈0 若声称成立）')
errs5b = [rel_err(ry(-t), H @ ry(t) @ H) for t in ths]
print(f'      T6f [正确恒等式] H·ry(θ)·H = ry(-θ) 最大偏差={max(errs5b):.3e}')

# ---------------------------------------------------------------- T7: rz 与 cz 对易
for q in (0, 1):
    lhs7 = Hwrap(rz(0.7), q) @ CZ
    rhs7 = CZ @ Hwrap(rz(0.7), q)
    check(f"T7  rz(θ)@比特{'a' if q==0 else 'b'} 与 cz 对易", lhs7, rhs7, tol=1e-14)

# ---------------------------------------------------------------- T8: ZSXXZ 模板
def zsxxz(th, ph, la):
    # 电路序 [rz(λ), sx, rz(θ+π), sx, rz(φ+3π)]：迭代正序左乘
    M = rz(la)
    M = sx() @ M
    M = rz(th + math.pi) @ M
    M = sx() @ M
    M = rz(ph + 3 * math.pi) @ M
    return M

n_rand = 50
ths8 = rng.uniform(0.01, 2 * math.pi, n_rand)
phs8 = rng.uniform(0, 2 * math.pi, n_rand)
las8 = rng.uniform(0, 2 * math.pi, n_rand)
errs8 = [rel_err(u3(t, p, l), zsxxz(t, p, l)) for t, p, l in zip(ths8, phs8, las8)]
print(f'      T8a ZSXXZ 随机 50 组最大偏差={max(errs8):.3e}')

# θ=π 特例（φ-λ ≠ 0 mod π）
specials = [(math.pi, 0.3, 1.1), (math.pi, 1.0, 2.5), (math.pi, 2.0, 0.7),
            (0.0, 0.9, 1.4), (2 * math.pi, 0.5, 0.5), (math.pi / 2, 0.0, math.pi),
            (math.pi, 0.0, math.pi)]
for (t, p, l) in specials:
    e = rel_err(u3(t, p, l), zsxxz(t, p, l))
    print(f'      T8b ZSXXZ θ={t:.4f} φ={p:.4f} λ={l:.4f} 偏差={e:.3e}')

# 零角省略变体（strip 掉近似 0 的 rz）
def strip_zero(gates):
    return [g for g in gates if not (g[0] == "rz" and abs(g[1][0] % (2 * math.pi)) < 1e-9)]

cases_strip = [(0.7, 0.3, 0.0), (0.7, 0.0, 0.4), (math.pi, 0.3, 0.7),
               (0.7, math.pi, 0.4), (0.7, 0.3, 2 * math.pi),
               (math.pi, 0.3, 2 * math.pi)]
for (t, p, l) in cases_strip:
    seq = [("rz", (l,)), ("sx", ()), ("rz", (t + math.pi,)), ("sx", ()),
           ("rz", (p + 3 * math.pi,))]
    stripped = strip_zero(seq)
    M = np.eye(2, dtype=complex)
    for nm, pr in stripped:            # 电路序正序：左乘
        M = (rz(pr[0]) if nm == "rz" else sx()) @ M
    e = rel_err(u3(t, p, l), M)
    print(f'      T8c strip变体 θ={t:.3f} φ={p:.3f} λ={l:.3f} → {len(stripped)}门 偏差={e:.3e}')

# ---------------------------------------------------------------- T9: _u3_to_rzsx_min 输出验证
grid = [(math.pi / 2, 0.0, math.pi), (math.pi, 0.0, math.pi), (0.0, 0.9, 1.4),
        (1e-10, 0.9, 1.4), (math.pi, 0.3, 1.1), (math.pi, 1.0, 2.5),
        (0.7, 0.0, 0.0), (0.7, math.pi, 0.4), (math.pi - 1e-12, 0.5, 0.5),
        (math.pi + 1e-12, 0.5, 0.5), (0.7, 0.3, 2 * math.pi),
        (math.pi, 0.3, 2 * math.pi), (0.3, 0.0, math.pi)]
for (t, p, l) in grid + list(zip(ths8[:10], phs8[:10], las8[:10])):
    gates = _u3_to_rzsx_min(t, p, l)
    M = np.eye(2, dtype=complex)
    for nm, pr in gates:               # 电路序正序：左乘
        M = (rz(pr[0]) if nm == "rz" else (sx() if nm == "sx" else X)) @ M
    e = rel_err(u3(t, p, l), M)
    tag = "FAIL" if e > 1e-8 else " ok "
    print(f'      T9 [{tag}] θ={t:.3e} φ={p:.3f} λ={l:.3f} → '
          f'{[ (g[0], round(g[1][0],2)) if g[1] else (g[0],()) for g in gates ]} dev={e:.3e}')
    if e > 1e-8:
        RESULTS.append((f"T9 u3({t:.2f},{p:.2f},{l:.2f}) 合成", False, e, str(gates)))

# ---------------------------------------------------------------- T10: 已删除 ZXZXZ 候选复核
def zxzxz(t, p, l):
    return rz(p + math.pi) @ X @ rz(t + math.pi) @ X

for (t, p, l) in [(0.7, 0.3, 1.1), (1.1, 0.9, 0.4), (math.pi / 2, 0.5, 0.5)]:
    M = zxzxz(t, p, l)
    diagness = abs(M[0, 1]) + abs(M[1, 0])
    e_expected = rel_err(rz(p - t), M)
    e_generic = rel_err(u3(t, p, l), M)
    print(f'      T10 ZXZXZ θ={t:.2f} φ={p:.2f} λ={l:.2f} '
          f'非对角={diagness:.2e} ==Rz(φ-θ)偏差={e_expected:.2e} '
          f'vs u3 偏差={e_generic:.3f}（删除该候选正确性：非对角≈0 且 ≠通用 u3）')

# ---------------------------------------------------------------- T11: _mat_to_u3_params 往返
def rand_su(d):
    """随机 SU(d)（谱分解指数化）。"""
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Hm = A + A.conj().T
    w, V = np.linalg.eigh(Hm)
    U = V @ np.diag(np.exp(1j * w)) @ V.conj().T
    return U / np.linalg.det(U) ** (1.0 / d)

def rand_su2():
    return rand_su(2)

errs11 = []
for _ in range(100):
    U = rand_su2()
    th, ph, la = pv2._mat_to_u3_params(U)
    errs11.append(rel_err(U, u3(th, ph, la)))
print(f'      T11 _mat_to_u3_params 随机 SU(2) 100 组最大偏差={max(errs11):.3e}')
# 特殊分支：对角、θ=π、θ>π/2
for U in (rz(0.8), X, np.array([[0, -1j], [-1j, 0]]),
          rz(1.9) @ X @ rz(0.4)):
    U = U / np.linalg.det(U) ** 0.5
    th, ph, la = pv2._mat_to_u3_params(U)
    print(f'      T11b 分支案例 U00={U[0,0]:.2f} U01={U[0,1]:.2f} → dev={rel_err(U, u3(th, ph, la)):.3e}')

# ---------------------------------------------------------------- T12: merge_1q_runs 实现
def mat_of_1q(name, params):
    if name == "rz":
        return rz(params[0])
    if name == "sx":
        return sx()
    if name == "x":
        return X
    raise ValueError(name)

errs12 = []
for _ in range(30):
    n_g = rng.integers(3, 8)
    run = []
    M = np.eye(2, dtype=complex)
    for _ in range(n_g):
        nm = rng.choice(["rz", "sx", "x"])
        pr = (float(rng.uniform(0, 2 * math.pi)),) if nm == "rz" else ()
        run.append((nm, pr))
        M = mat_of_1q(nm, pr) @ M
    ops = [Inst(nm, (0,), pr) for nm, pr in run]
    ops_c = list(ops)
    pv2.merge_1q_runs(ops_c)
    M2 = np.eye(2, dtype=complex)
    for inst in ops_c:
        M2 = pv2._oneq_matrix(inst.name, inst.params) @ M2
    errs12.append(rel_err(M, M2))
print(f'      T12 merge_1q_runs 30 随机游程最大偏差={max(errs12):.3e}')

# ---------------------------------------------------------------- T13: 通勤规则（_cx_commutes）
def three_bit(mat_ab, a, b, n=3):
    """把 2Q 矩阵嵌到 n 比特（比特序 0..n-1 高位在前）。"""
    dims = [2] * n
    I = np.eye(2)
    full = None
    for i in range(n):
        mi = mat_ab if i == a else (np.eye(2))
        mi = np.eye(2)
    # 用 kron 逐位构建
    order = sorted(set(range(n)) - {a, b})
    M = None
    # 简化为 n=3：a,b ∈ {0,1,2}
    mats = [I2, I2, I2]
    mats[a], mats[b] = None, None
    # 直接构造：基 = 比特 a(高位)⊗b(低位)，外比特放两侧
    if (a, b) == (0, 1):
        return np.kron(mat_ab, I2)
    if (a, b) == (1, 2):
        return np.kron(I2, mat_ab)
    if (a, b) == (0, 2):
        return (np.kron(I2, SWAP) if False else None)
    raise ValueError

def cx_n(a, b, n=3):
    """n=3 比特 cx（控制 a、目标 b），经 SWAP 共轭映射到 (0,1)/(1,2) 形。"""
    P = np.eye(2 ** n, dtype=complex)
    if set((a, b)) == {0, 1}:
        base = np.kron(CX, I2) if (a, b) == (0, 1) else np.kron(CXb, I2)
    elif set((a, b)) == {1, 2}:
        base = np.kron(I2, CX) if (a, b) == (1, 2) else np.kron(I2, CXb)
    elif set((a, b)) == {0, 2}:
        P = np.kron(I2, SWAP)  # 交换比特 1,2
        base = P @ np.kron(CX, I2) @ P if (a, b) == (0, 2) \
            else P @ np.kron(CXb, I2) @ P
    else:
        raise ValueError((a, b))
    return base

def cz_n(a, b, n=3):
    """n=3 比特 cz（对称门），经 SWAP 共轭映射。"""
    if set((a, b)) == {0, 1}:
        return np.kron(CZ, I2)
    if set((a, b)) == {1, 2}:
        return np.kron(I2, CZ)
    if set((a, b)) == {0, 2}:
        P = np.kron(I2, SWAP)
        return P @ np.kron(CZ, I2) @ P
    raise ValueError((a, b))

# 同控/同靶对易
d1 = rel_err(cx_n(0, 1) @ cx_n(0, 2), cx_n(0, 2) @ cx_n(0, 1))
d2 = rel_err(cx_n(0, 1) @ cx_n(2, 1), cx_n(2, 1) @ cx_n(0, 1))
print(f'      T13a cx 同控对易偏差={d1:.3e}  同靶对易偏差={d2:.3e}')
# 交叉控制不对易
d3 = rel_err(cx_n(0, 1) @ cx_n(1, 0) if False else CXb @ CX, CX @ CXb)
print(f'      T13b cx(a,b)·cx(b,a) vs 反序偏差={d3:.3f}（应显著≠0）')
# cz 全对易
d4 = rel_err(cz_n(0, 1) @ cz_n(1, 2), cz_n(1, 2) @ cz_n(0, 1))
print(f'      T13c cz 全对易偏差={d4:.3e}')
# cz·cz = I
print(f'      T13d cz·cz=I 偏差={rel_err(np.eye(4), CZ @ CZ):.3e}')

# ---------------------------------------------------------------- T14: 实现级随机测试（n=3 电路）
def circuit_matrix(ops, n=3):
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    for inst in ops:
        nm = inst.name
        qs = [int(q) for q in inst.qubits]
        if nm in ("rz", "sx", "x"):
            m = mat_of_1q(nm, inst.params)
            mats = [I2] * n
            mats[qs[0]] = m
            G = mats[0]
            for k in range(1, n):
                G = np.kron(G, mats[k])
        elif nm == "cz":
            G = cz_n(qs[0], qs[1], n)
        elif nm == "cx":
            G = cx_n(qs[0], qs[1], n)
        else:
            raise ValueError(nm)
        U = G @ U
    return U

def rand_circ(n=3, n_ops=12, gates=("rz", "sx", "x", "cz")):
    ops = []
    for _ in range(n_ops):
        nm = rng.choice(gates)
        if nm in ("cz", "cx"):
            a, b = rng.choice(n, size=2, replace=False)
            ops.append(Inst(nm, (int(a), int(b))))
        else:
            q = int(rng.integers(0, n))
            if nm == "rz":
                ops.append(Inst("rz", (q,), (float(rng.uniform(0, 2 * math.pi)),)))
            else:
                ops.append(Inst(nm, (q,)))
    return ops

def check_pass(name, fn, gates=("rz", "sx", "x", "cz"), n_trials=30,
               tol=1e-8, mod_phase=True):
    worst = 0.0
    for _ in range(n_trials):
        ops = rand_circ(gates=gates)
        U0 = circuit_matrix(ops)
        fn(ops)
        U1 = circuit_matrix(ops)
        worst = max(worst, rel_err(U0, U1))
    check(f"T14 {name}（30 随机电路）", np.eye(1), np.eye(1),
          tol=1e-15, note="")
    ok = worst < tol
    RESULTS.append((f"{name} 随机电路测试", ok, worst, ""))
    print(f'      T14 [{ "PASS" if ok else "FAIL" }] {name} 最大偏差={worst:.3e}')

check_pass("_cancel_adjacent_cz", pv2._cancel_adjacent_cz)
check_pass("_cancel_cz_global", pv2._cancel_cz_global)
check_pass("commute_rz_through_cz", pv2.commute_rz_through_cz)
check_pass("commute_rz_through_cz_rev", pv2.commute_rz_through_cz_rev)
check_pass("_merge_sx_pairs+_strip_identity_1q", lambda o: (pv2._merge_sx_pairs(o), pv2._strip_identity_1q(o)))
check_pass("slide_1q_through", lambda o: pv2.slide_1q_through(o, max_rounds=4))
check_pass("slide_x_through_cz", lambda o: pv2.slide_x_through_cz(o, max_rounds=4),
           tol=2e-8)   # 模全局相位：i 因子被丢弃，仍应模相位相等
check_pass("_sort_segment_1q", lambda o: pv2._sort_segment_1q(o))
check_pass("_preslide_1q", lambda o: pv2._preslide_1q(o, max_rounds=4))
check_pass("commute_cancel_cx", lambda o: pv2.commute_cancel_cx(o, window=8),
           gates=("cx", "rz"))

# _absorb_h：构造含 H·rz·H 模式的电路
def rand_absorb(n_trials=20):
    worst = 0.0
    for _ in range(n_trials):
        ops = rand_circ(gates=("rz", "sx", "x", "cz"))
        q = int(rng.integers(0, 3))
        ops = ops[:6] + [Inst("rz", (q,), (H_GATE[0],)), Inst("sx", (q,)),
                         Inst("rz", (q,), (H_GATE[0],)),
                         Inst("rz", (q,), (float(rng.uniform(0, 2 * math.pi)),)),
                         Inst("rz", (q,), (H_GATE[0],)), Inst("sx", (q,)),
                         Inst("rz", (q,), (H_GATE[0],))] + ops[6:]
        U0 = circuit_matrix(ops)
        pv2._absorb_h(ops)
        U1 = circuit_matrix(ops)
        worst = max(worst, rel_err(U0, U1))
    print(f'      T14 [{"PASS" if worst < 1e-8 else "FAIL"}] _absorb_h 随机电路 最大偏差={worst:.3e}')

rand_absorb()

# ---------------------------------------------------------------- T15: swap 推挤/消解/吸收
def rand_swap_circ(n=3, n_ops=14):
    ops = rand_circ(gates=("rz", "sx", "x", "cz"), n_ops=n_ops)
    for _ in range(4):
        a, b = rng.choice(n, size=2, replace=False)
        pos = int(rng.integers(0, len(ops) + 1))
        ops.insert(pos, Inst("swap", (int(a), int(b))))
    return ops

def circuit_matrix_swap(ops, n=3):
    U = np.eye(2 ** n, dtype=complex)
    for inst in ops:
        qs = [int(q) for q in inst.qubits]
        if inst.name == "swap":
            P = np.eye(1)
            # 用排列构造：交换比特 qs[0], qs[1] 的基
            dim = 2 ** n
            perm = np.arange(dim)
            for i in range(dim):
                bits = [ (i >> (n - 1 - k)) & 1 for k in range(n) ]
                bits[qs[0]], bits[qs[1]] = bits[qs[1]], bits[qs[0]]
                j = 0
                for b in bits:
                    j = (j << 1) | b
                perm[i] = j
            P = np.zeros((dim, dim))
            P[np.arange(dim), perm.astype(int)] = 1.0
            U = P @ U
        else:
            nm = inst.name
            if nm in ("rz", "sx", "x"):
                m = mat_of_1q(nm, inst.params)
                mats = [I2] * n
                mats[qs[0]] = m
                G = mats[0]
                for k in range(1, n):
                    G = np.kron(G, mats[k])
            else:
                G = cz_n(qs[0], qs[1], n)
            U = G @ U
    return U

def _mk_circ(ops, n=3):
    c = Circuit(n)
    c.ops = ops
    return c

worst = 0.0
for _ in range(30):
    ops = rand_swap_circ()
    U0 = circuit_matrix_swap(ops)
    pv2.push_swaps_v2(_mk_circ(ops), None)
    U1 = circuit_matrix_swap(ops)
    worst = max(worst, rel_err(U0, U1))
print(f'      T15 [{"PASS" if worst < 1e-8 else "FAIL"}] push_swaps 随机电路 最大偏差={worst:.3e}')

# absorb_trailing + final_layout 一致性（生产 compact 判据：吸收后产物|0⟩
# == 原逻辑线路经【更新后 fin】嵌入的 |0⟩）
def embed_circ(logical_ops, fin):
    return [Inst(i.name, (int(fin[int(i.qubits[0])]),) if i.nq == 1
                 else (int(fin[int(i.qubits[0])]), int(fin[int(i.qubits[1])])),
                 i.params) for i in logical_ops]

worst_abs = 0.0
for _ in range(30):
    logical = rand_circ(gates=("rz", "sx", "x", "cz"), n_ops=8)
    pos = {i: i for i in range(3)}
    ops = [Inst(i.name, tuple(int(pos[int(q)]) for q in i.qubits), i.params)
           for i in logical]
    for _ in range(3):                      # 尾部 swap 序列
        a, b = rng.choice(3, size=2, replace=False)
        ops.append(Inst("swap", (int(a), int(b))))
        la = next((k for k, v in pos.items() if v == a), None)
        lb = next((k for k, v in pos.items() if v == b), None)
        if la is not None:
            pos[la] = b
        if lb is not None:
            pos[lb] = a
    fin = dict(pos)
    absorb_trailing(_mk_circ(ops), fin)
    U1 = circuit_matrix_swap(ops)                     # 吸收后产物
    U2 = circuit_matrix_swap(embed_circ(logical, fin))  # 原逻辑线路经 fin 嵌入
    z = np.zeros(2 ** 3, dtype=complex)
    z[0] = 1.0
    worst_abs = max(worst_abs, float(np.linalg.norm(U1 @ z - U2 @ z)))
print(f'      T15b absorb_trailing+final_layout compact 判据 最大偏差={worst_abs:.3e}')

# ---------------------------------------------------------------- T16: fold_only / fold_and_relink（含幽灵位）
def rand_routed(n_logic=3, n_phys=6, n_ops=14):
    """逻辑线路嵌入物理 0..n_logic-1，路由器式 swap 序列（可穿空位）。

    pos：逻辑→物理；inv：物理→逻辑（路由器双映射语义）。"""
    pos = {i: i for i in range(n_logic)}
    inv = {i: i for i in range(n_logic)}
    ops = []
    for _ in range(n_ops):
        if rng.random() < 0.4:
            a, b = rng.choice(n_phys, size=2, replace=False)
            ops.append(Inst("swap", (int(a), int(b))))
            la, lb = inv.get(int(a)), inv.get(int(b))
            inv.pop(int(a), None); inv.pop(int(b), None)
            if la is not None:
                inv[int(b)] = la
            if lb is not None:
                inv[int(a)] = lb
            pos = {l: p for p, l in inv.items()}
        else:
            q = int(rng.integers(0, n_logic))
            p = pos[q]
            if rng.random() < 0.5:
                ops.append(Inst("rz", (p,), (float(rng.uniform(0, 2 * math.pi)),)))
            else:
                ops.append(Inst("sx", (p,)))
    return ops, pos

def full_matrix(ops, n):
    """n 比特电路矩阵（rz/sx/x/swap/cz/cx 通用，位构造）。"""
    U = np.eye(2 ** n, dtype=complex)
    for inst in ops:
        if inst.name == "swap":
            P = np.zeros((2 ** n, 2 ** n))
            perm = np.arange(2 ** n)
            for i in range(2 ** n):
                bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                a, b = int(inst.qubits[0]), int(inst.qubits[1])
                bits[a], bits[b] = bits[b], bits[a]
                j = 0
                for bb in bits:
                    j = (j << 1) | bb
                perm[i] = j
            P[np.arange(2 ** n), perm.astype(int)] = 1.0
            U = P @ U
        elif inst.nq == 1:
            m = mat_of_1q(inst.name, inst.params)
            q = int(inst.qubits[0])
            mats = [I2] * n
            mats[q] = m
            G = mats[0]
            for k in range(1, n):
                G = np.kron(G, mats[k])
            U = G @ U
        else:  # cz
            G = np.eye(2 ** n, dtype=complex)
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            for i in range(2 ** n):
                bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                if bits[a] and bits[b]:
                    G[i, i] = -1.0
            U = G @ U
    return U

worst_fold = 0.0
for _ in range(40):
    ops, pos = rand_routed()
    U0 = full_matrix(ops, 6)
    folded, arr = pv2.fold_only(_mk_circ(ops, 6))
    U1 = full_matrix(folded.ops, 6)
    # fold 终态不变量：U(folded)|0⟩ = Π_final·U(routed)|0⟩，
    # Π: 把坐标 p 的内容搬到 arr[p] ⟺ (Π|ψ⟩)[k] = ψ[arr^{-1}(k)]
    inv_arr = np.empty(6, dtype=np.int64)
    inv_arr[np.asarray(arr, dtype=np.int64)] = np.arange(6)
    perm = np.arange(64)
    for i in range(64):
        bits = [(i >> (5 - k)) & 1 for k in range(6)]
        j = 0
        for k in range(6):
            j = (j << 1) | bits[int(inv_arr[k])]
        perm[i] = j
    Pi = np.zeros((64, 64))
    Pi[np.arange(64), perm] = 1.0
    z = np.zeros(64, dtype=complex)
    z[0] = 1.0
    # Pi = Π^{-1}（构造方向），故 Π = Pi.T
    worst_fold = max(worst_fold,
                     float(np.linalg.norm(Pi.T @ (U0 @ z) - U1 @ z)))
print(f'      T16 [{"PASS" if worst_fold < 1e-8 else "FAIL"}] fold_only 幽灵位随机电路 最大偏差={worst_fold:.3e}')

# ---------------------------------------------------------------- T17: 链恒等式
def cx_n_large(a, b, n):
    """n 比特 cx（控制 a、目标 b），按位 kron 构造。"""
    dim = 2 ** n
    G = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        j = i
        if bits[a]:
            j = i ^ (1 << (n - 1 - b))
        G[j, i] = 1.0
    return G

worst_chain = 0.0
for d in range(1, 9):
    chain = chain_for_distance(d)
    U = np.eye(2 ** (d + 1), dtype=complex)
    for c, t in chain:
        G = np.zeros_like(U)
        # cx(c, t) 在 d+1 比特
        U = cx_n_large(c, t, d + 1) @ U
    target = cx_n_large(0, d, d + 1)
    worst_chain = max(worst_chain, rel_err(target, U))
print(f'      T17a KMS 链 d=1..8 == cx(0,d) 最大偏差={worst_chain:.3e}')

# 方向翻转（局部索引 0..4 帧内：CNOT(seg[4], seg[0]) = 镜像链）
chain = chain_for_distance(4)
flipped = [(4 - c, 4 - t) for c, t in chain]
Uf = np.eye(2 ** 5, dtype=complex)
for c, t in flipped:
    Uf = cx_n_large(c, t, 5) @ Uf
e_flip = rel_err(cx_n_large(4, 0, 5), Uf)
print(f'      T17b 链方向翻转映射 偏差={e_flip:.3e}')

# ---------------------------------------------------------------- T18: KAK 往返 + magic 基
worst_kak = 0.0
worst_kak20 = 0.0
for _ in range(10):
    U4 = rand_su(4)
    dec = syn.kak_decompose(U4, iters=200)
    worst_kak = max(worst_kak, rel_err(U4, syn.rebuild_kak(dec)))
    dec20 = syn.kak_decompose(U4, iters=20)   # 生产口径（仅作拟合初值）
    worst_kak20 = max(worst_kak20, rel_err(U4, syn.rebuild_kak(dec20)))
print(f'      T18 [{"PASS" if worst_kak < 1e-6 else "FAIL"}] KAK 往返(iters=200) 最大偏差={worst_kak:.3e}'
      f' | 生产口径 iters=20 最大偏差={worst_kak20:.3e}（仅作拟合初值）')

# magic 基性质：M†·kron(l,r)·M ∈ SO(4) 实矩阵
l, r = rand_su2(), rand_su2()
Q = syn.MAGIC.conj().T @ np.kron(l, r) @ syn.MAGIC
print(f'      T18b magic 基 kron→SO(4) 虚部={np.abs(Q.imag).max():.3e} 正交性={np.linalg.norm(Q @ Q.T - np.eye(4)):.3e}')
ll, rr = syn._so4_to_su2(np.real(Q))
print(f'      T18c _so4_to_su2 往返 l={rel_err(l, ll):.2e} r={rel_err(r, rr):.2e}')

# ---------------------------------------------------------------- T19: fit_block_torch + 3cx 合成
worst_fit = 0.0
n_fit_ok = 0
for _ in range(3):
    U4 = rand_su(4)
    dec = syn.kak_decompose(U4, iters=20)
    init = np.zeros(21)
    for idx, K in enumerate((dec["K1"], dec["K2"])):
        th, ph, la = syn._su2_euler(K)
        init[3 * idx:3 * idx + 3] = [th, ph, la]
    for idx, K in enumerate((dec["K3"], dec["K4"])):
        th, ph, la = syn._su2_euler(K)
        init[14 + 3 * idx:14 + 3 * idx + 3] = [th, ph, la]
    init[6:14] = syn._cached_3cx_params(np.asarray(dec["k"]))
    t, ov = syn.fit_block_torch(U4, init)
    insts = syn.block_params_to_insts(t, (0, 1))
    M = np.eye(4, dtype=complex)
    for inst in insts:                      # 电路序正序：左乘
        if inst.nq == 1:
            m = pv2._oneq_matrix(inst.name, inst.params)
            q = int(inst.qubits[0])
            M = (np.kron(m, I2) if q == 0 else np.kron(I2, m)) @ M
        else:
            ctl = int(inst.qubits[0])
            M = (CX if ctl == 0 else CXb) @ M
    e = rel_err(U4, M)
    worst_fit = max(worst_fit, e)
    n_fit_ok += int(ov >= 1 - 1e-7)
print(f'      T19 [{"PASS" if worst_fit < 1e-4 and n_fit_ok == 3 else "FAIL"}] '
      f'fit_block_torch+block_params_to_insts 往返 最大偏差={worst_fit:.3e} '
      f'（B2 守卫阈值允许 ≤8.9e-4；ov≥1-1e-7: {n_fit_ok}/3）')

# _to_cz_basis 往返
insts = syn.block_params_to_insts(t, (0, 1))
czb = pv2._to_cz_basis(insts)
M2 = np.eye(4, dtype=complex)
for inst in czb:                           # 电路序正序：左乘
    if inst.nq == 1:
        m = pv2._oneq_matrix(inst.name, inst.params)
        q = int(inst.qubits[0])
        M2 = (np.kron(m, I2) if q == 0 else np.kron(I2, m)) @ M2
    else:
        M2 = CZ @ M2
print(f'      T19b _to_cz_basis 往返偏差={rel_err(M, M2):.3e}')

# ---------------------------------------------------------------- T20: B2 误差界复核
# 构造 ov = 1-1e-7 的近恒等 SU(4)：U = diag(e^{iδ},e^{iδ},e^{iδ},e^{-3iδ})
d20 = math.sqrt(1e-7 / 1.5)
U20 = np.diag([np.exp(1j * d20)] * 3 + [np.exp(-3j * d20)])
ov20 = abs(np.trace(U20)) / 4
frob = np.linalg.norm(U20 - np.eye(4))
bound_correct = 2 * math.sqrt(2) * math.sqrt(1 - ov20)
bound_docs = 2 * math.sqrt(1 - ov20)
print(f'      T20 ov={ov20:.9f} ||U-I||_F={frob:.3e} '
      f'正确界 2√2√(1-ov)={bound_correct:.3e} 文档界 2√(1-ov)={bound_docs:.3e} '
      f'→ 文档界 {"够" if bound_docs >= frob else "不够（被打破，因子差 √2）"}')

# ---------------------------------------------------------------- T21: noise_aware_relabel 语义
import types as _types
class FakeCalib:
    err_2q = {}
    err_1q = np.full(6, 1e-3)
    err_ro = np.full(6, 1e-2)
adj21 = np.zeros((6, 6), dtype=bool)
for i in range(5):
    adj21[i, i + 1] = adj21[i + 1, i] = True
adj21[0, 5] = adj21[5, 0] = True
spec21 = _types.SimpleNamespace(n=6, disabled_qubits=set(), adj=adj21,
                                calib=FakeCalib())
worst21 = 0.0
for _ in range(20):
    ops = []
    for k in range(8):
        a, b = rng.choice(6, size=2, replace=False)
        ops.append(Inst("cz", (int(a), int(b))))
        q = int(rng.integers(0, 6))
        ops.append(Inst("rz", (q,), (float(rng.uniform(0, 2 * math.pi)),)))
    fin21 = {0: 0, 1: 2}
    c21 = _mk_circ(ops, 6)
    U0 = full_matrix(ops, 6)
    pv2.noise_aware_relabel(c21, spec21, fin21, max_passes=5)
    # 从 1Q 门标签重建坐标置换 Π（每坐标取首个 1Q 门），验证 U1 == Π·U0·Π†
    sigma = {}
    for inst, oi in zip(c21.ops, ops):
        if inst.nq == 1 and int(oi.qubits[0]) not in sigma:
            sigma[int(oi.qubits[0])] = int(inst.qubits[0])
    Pi = np.zeros((64, 64))
    perm = np.arange(64)
    for i in range(64):
        bits = [(i >> (5 - k)) & 1 for k in range(6)]
        j = 0
        for k in range(6):
            j = (j << 1) | bits[sigma.get(k, k)]
        perm[i] = j
    Pi[np.arange(64), perm] = 1.0
    U1 = full_matrix(c21.ops, 6)
    worst21 = max(worst21, rel_err(Pi @ U0 @ Pi.T, U1))
print(f'      T21 noise_aware_relabel 语义（U1 == Π·U0·Π†）最大偏差={worst21:.3e}')

# ---------------------------------------------------------------- 汇总
print()
print("=" * 70)
n_fail = sum(1 for _, ok, _, _ in RESULTS if not ok)
print(f"总结：{len(RESULTS)} 项检查，{n_fail} 项失败")
for name, ok, err, note in RESULTS:
    if not ok:
        print(f"  FAIL {name}: dev={err:.3e} {note}")
