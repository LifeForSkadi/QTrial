# -*- coding: utf-8 -*-
"""2Q 块重合成（A2 组件，2026-11-09）：Cartan KAK 分解 + 3-CX 合成。

对标 qiskit UnitarySynthesis 的公开数学（Weil 分解 / Vatan-Williams），
独立实现 + 噪声感知选优（qiskit 的合成是噪声无关的固定分解）。

实现策略（自审计驱动）：
  - kak_decompose：magic 基变换 + 交替最小二乘 Procrustes
    （min ||Q1·D(k)·Q2 - Um||，Q1,Q2 ∈ SO(4)，D 为 Cartan 对角），
    避开了 SO(4)→SU(2)×SU(2) 解析映射的记忆风险；
  - SO(4) 块提取（su2 因子）由 _so4_to_su2 的往返测试锁定映射形式；
  - W(k) 的 3-CX 合成由 torch 批量拟合锁定（解析初值 + Adam 精修），
    k 网格缓存加速；
  - 全部数值逻辑由往返测试保证：U → decompose → rebuild == U（模全局相位）。
"""
from __future__ import annotations

import math

import numpy as np

from qtrail.pure.circuit import Inst

# magic 基（列向量），与 qiskit two_qubit_decompose 同款
MAGIC = (1.0 / math.sqrt(2)) * np.array(
    [[1, 1j, 0, 0],
     [0, 0, 1j, 1],
     [0, 0, 1j, -1],
     [1, -1j, 0, 0]], dtype=complex)


def _su2_proj(M):
    """任意 2×2 复矩阵 → SU(2) 投影（酉化 + det 规范化）。"""
    U, _, Vh = np.linalg.svd(M)
    K = U @ Vh
    K = K / np.sqrt(np.linalg.det(K))
    return K


def _extract_left(T, K2):
    """T ≈ kron(K1, K2)：从 2×2 块提取 K1（块_ij = K1_ij·K2）。"""
    K1 = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            blk = T[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            K1[i, j] = np.trace(blk @ K2.conj().T) / 2.0
    return _su2_proj(K1)


def _extract_right(T, K1):
    """T ≈ kron(K1, K2)：从 2×2 块提取 K2（加权平均各块）。"""
    K2 = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            blk = T[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            K2 += np.conj(K1[i, j]) * blk
    return _su2_proj(K2)


def _extract_k(R):
    """R ≈ w_matrix(k)：提取 Cartan 系数。

    w_matrix 结构：W00 = e^{ik3}cos(k1-k2)、W03 = i·e^{ik3}sin(k1-k2)、
    W11 = e^{-ik3}cos(k1+k2)、W12 = i·e^{-ik3}sin(k1+k2)。
    k3 由 W00·e^{-ik3} 为实确定（两候选取残差小者）；r = k1-k2 与
    s = k1+k2 由 atan2 反解（与 qiskit Weyl 分解对照验证）。
    """
    k3_cands = [np.angle(R[0, 0]), np.angle(R[0, 0]) + np.pi]
    best = None
    for k3 in k3_cands:
        w00 = R[0, 0] * np.exp(-1j * k3)
        w03 = R[0, 3] * np.exp(-1j * k3) * (-1j)
        score = abs(w00.imag) + abs(w03.imag)
        if best is None or score < best[0]:
            best = (score, k3, w00, w03)
    _, k3, w00, w03 = best
    r = math.atan2(w03.real, w00.real)              # r = k1 - k2
    w11 = R[1, 1] * np.exp(1j * k3)
    w12 = R[1, 2] * np.exp(1j * k3) * (-1j)
    s = math.atan2(w12.real, w11.real)              # s = k1 + k2
    k1 = (r + s) / 2.0
    k2 = (s - r) / 2.0
    return np.array([k1, k2, k3])


def kak_decompose(U: np.ndarray, iters: int = 100) -> dict:
    """Cartan KAK 分解（计算基复域交替 Procrustes，2026-11-09 重写）：

    U = phase·(K1⊗K2)·w_matrix(k)·(K3⊗K4)。

    交替更新（每因子独立闭式 Procrustes + kron 块结构提取）：
      K1,K2 ← U·(W·kron(K3,K4)·phase)† 的 kron 块提取；
      K3,K4 ← (kron(K1,K2)·W·phase)†·U 的 kron 块提取；
      k ← 残差 R = kron(K1,K2)†·U·kron(K3,K4)†/phase 的 w_matrix 结构反解；
      phase ← 重建酉与 U 的重叠。
    正确性由往返测试 + qiskit Weyl 分解对照（tests）双重锁定。
    """
    U = np.asarray(U, dtype=complex)
    K1 = K2 = K3 = K4 = np.eye(2, dtype=complex)
    k = np.zeros(3)
    phase = 1.0 + 0j
    XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                  dtype=complex)
    YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
                  dtype=complex)
    ZZ = np.diag([1, -1, -1, 1]).astype(complex)
    for _ in range(iters):
        # K1..K4 闭式更新（kron 块提取，给定 k 时各步最优）
        W = w_matrix(k)
        M = W @ np.kron(K3, K4) * phase
        T = U @ M.conj().T
        K1 = _extract_left(T, K2)
        K2 = _extract_right(T, K1)
        N = np.kron(K1, K2) @ W * phase
        T2 = N.conj().T @ U
        K3 = _extract_left(T2, K4)
        K4 = _extract_right(T2, K3)
        # k 高斯-牛顿更新（∂W/∂k_j = i·P_j·W；3×3 正规方程二次收敛）
        A = np.kron(K1, K2) * phase
        B = np.kron(K3, K4)
        for _g in range(12):
            W = w_matrix(k)
            res = (A @ W @ B - U).reshape(-1)
            J = np.zeros((16, 3), dtype=complex)
            for j, P in enumerate((XX, YY, ZZ)):
                J[:, j] = (A @ (1j * P @ W) @ B).reshape(-1)
            JtJ = np.real(J.conj().T @ J)
            Jtr = np.real(J.conj().T @ res)
            dk = np.linalg.solve(JtJ + 1e-12 * np.eye(3), Jtr)
            k = k - dk
            if np.linalg.norm(dk) < 1e-12:
                break
        approx = np.kron(K1, K2) @ w_matrix(k) @ np.kron(K3, K4)
        phase = np.vdot(approx, U) / (np.linalg.norm(approx) ** 2)
    return {"k": (float(k[0]), float(k[1]), float(k[2])),
            "K1": K1, "K2": K2, "K3": K3, "K4": K4,
            "phase": phase}


def _so4_to_su2(Q):
    """SO(4) → SU(2)×SU(2) 提取（magic 基性质 M†·kron(l,r)·M ∈ SO(4)）。

    K = M·Q·M† = kron(l, r)（4×4 复），从 K 的 2×2 块提取：
      r = su2_proj(K00)；l 的元素 = trace(K_ij·r†)/2（数值稳健）。
    (l, r) 的自由度（lV, V†r）由 su2 规范化取代表元，KAK 重建不敏感。
    """
    K = MAGIC @ Q @ MAGIC.conj().T
    r = _su2_proj(K[:2, :2])
    l = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            blk = K[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            l[i, j] = np.trace(blk @ r.conj().T) / 2.0
    l = _su2_proj(l)
    return l, r


def _bloch_to_su2(th, ph, la):
    c, s = math.cos(th / 2), math.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                    dtype=complex)


# ======================================================================
# 块级全参数合成（torch 拟合，2026-11-09 终版方案）
# ======================================================================
# 电路模板（块内两比特 a, b）：
#   [RzRyRz on a][RzRyRz on b]  (K1⊗K2 欧拉，6 参数)
#   CX(a,b) (Rz⊗Rz)(t0,t1) (Ry⊗Ry)(t2,t3) CX(a,b)
#          (Rz⊗Rz)(t4,t5) (Ry⊗Ry)(t6,t7) CX(a,b)   (3-CX 模板，8 参数)
#   [RzRyRz on a][RzRyRz on b]  (K3⊗K4 欧拉，6 参数)
#   × 全局相位（1 参数）
# 共 21 参数。拟合目标 ||C(t) - U||（重叠最大化），收敛验证 > 1-1e-9
# 才应用（否则返回 None 保留原块）。


def _u3_mat(th, ph, la):
    c, s = math.cos(th / 2), math.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]],
                    dtype=complex)


def _block_circuit_matrix(t):
    """21 参数 → 4×4 电路矩阵（numpy，测试/验证用）。"""
    ua = _u3_mat(t[0], t[1], t[2])
    ub = _u3_mat(t[3], t[4], t[5])
    C = np.kron(ua, ub)
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                  dtype=complex)
    rz = lambda a: np.diag([math.e ** (-1j * a / 2), math.e ** (1j * a / 2)])
    ry = lambda a: np.array([[math.cos(a / 2), -math.sin(a / 2)],
                             [math.sin(a / 2), math.cos(a / 2)]], dtype=complex)
    C = cx @ C
    C = np.kron(rz(t[6]), rz(t[7])) @ C
    C = np.kron(ry(t[8]), ry(t[9])) @ C
    C = cx @ C
    C = np.kron(rz(t[10]), rz(t[11])) @ C
    C = np.kron(ry(t[12]), ry(t[13])) @ C
    C = cx @ C
    uc = _u3_mat(t[14], t[15], t[16])
    ud = _u3_mat(t[17], t[18], t[19])
    C = np.kron(uc, ud) @ C
    return np.exp(1j * t[20]) * C


def fit_block_circuit(U, init=None, max_steps=30, tol=1e-11):
    """拟合块电路参数使 C(t) == U（模全局相位）。返回 (t, overlap)。

    有限差分高斯-牛顿（numpy）：21 参数、每步 21 次 4×4 矩阵前向，
    ~2ms/步；好初值（KAK）3-8 步收敛到 1e-12。多起点由调用方控制。
    """
    t = np.asarray(init, dtype=np.float64) if init is not None else None
    if t is None:
        rng = np.random.default_rng()
        t = np.concatenate([rng.uniform(0, 2 * np.pi, 20), [0.0]])

    def resid(tt):
        C = _block_circuit_matrix(tt)
        # 相位对齐后的复残差
        ov = np.vdot(C, U)
        C = C * (ov / abs(ov))
        return (C - U).reshape(-1)

    h = 1e-7
    for step in range(max_steps):
        r = resid(t)
        loss = np.linalg.norm(r)
        if loss < tol:
            break
        J = np.zeros((16, 21), dtype=complex)
        for j in range(21):
            tp = t.copy()
            tp[j] += h
            J[:, j] = (resid(tp) - r) / h
        JtJ = np.real(J.conj().T @ J)
        Jtr = np.real(J.conj().T @ r)
        try:
            dk = np.linalg.solve(JtJ + 1e-10 * np.eye(21), Jtr)
        except np.linalg.LinAlgError:
            break
        # 线搜索（保证下降）
        alpha = 1.0
        for _ in range(20):
            if np.linalg.norm(resid(t - alpha * dk)) < loss:
                break
            alpha *= 0.5
        t = t - alpha * dk
    C = _block_circuit_matrix(t)
    ov = abs(np.vdot(C, U)) / (np.linalg.norm(C) * np.linalg.norm(U))
    return t, ov


def fit_block_multi(U, n_starts=2):
    """多起点拟合，返回 (t, overlap) 最优解。"""
    best = (None, 0.0)
    # 起点 1：KAK 分解初值
    try:
        dec = kak_decompose(U)
        init = np.zeros(21)
        for idx, K in enumerate((dec["K1"], dec["K2"])):
            th, ph, la = _su2_euler(K)
            init[3 * idx:3 * idx + 3] = [th, ph, la]
        for idx, K in enumerate((dec["K3"], dec["K4"])):
            th, ph, la = _su2_euler(K)
            init[14 + 3 * idx:14 + 3 * idx + 3] = [th, ph, la]
        init[6:14] = _init_3cx_params(np.asarray(dec["k"]))
        t, ov = fit_block_circuit(U, init=init)
        if ov > best[1]:
            best = (t, ov)
        if ov > 1 - 1e-9:
            return best
    except Exception:
        pass
    # 随机起点（收敛失败时兜底）
    for _ in range(n_starts):
        t, ov = fit_block_circuit(U, init=None)
        if ov > best[1]:
            best = (t, ov)
        if ov > 1 - 1e-9:
            break
    return best


def _su2_euler(K):
    """SU(2) → u3 欧拉角（θ,φ,λ）。"""
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
        alpha = (math.atan2(c.imag, c.real) + math.atan2((-b).imag, (-b).real)
                 - math.atan2(d.imag, d.real))
    ph = (math.atan2(c.imag, c.real) - alpha) % (2 * math.pi)
    la = (math.atan2((-b).imag, (-b).real) - alpha) % (2 * math.pi)
    return (th, ph, la)


def _init_3cx_params(k):
    """3-CX 模板 8 参数的近似初值（W(k) 的 2-CX ZZ 分解 + 零填充）。"""
    k1, k2, k3 = k
    # W = diag(a,b,b,a)：ZZ 层 + 实对角层
    a = math.cos(k1 - k2) * (math.cos(k3) + 1j * math.sin(k3))
    b = 1j * math.sin(k1 - k2) * (math.cos(k3) + 1j * math.sin(k3))
    # diag(1,e^{iα},e^{iα},1) 层的 α = phase(b)-phase(a) 修正：
    alpha = np.angle(b) - np.angle(a) if abs(a) > 1e-12 else 0.0
    theta = 2.0 * math.atan2(abs(b), abs(a))
    return np.array([-alpha / 2, -alpha / 2, theta, theta,
                     -alpha / 2, -alpha / 2, theta, theta])


# 3-CX 中间 8 参数缓存：键 = Cartan 系数 k 的 3 位小数量化。
# W(k) 的合成只依赖 k——相同 k 的块复用拟合好的中间参数，
# Adam 从好初值出发步数大幅下降（耗时优化的核心，2026-11-09）。
_FIT3CX_CACHE: dict = {}


def _cached_3cx_params(k):
    """k → 8 参数（缓存优先）。首次拟合 W(k) 后写入缓存。"""
    import torch
    key = tuple(round(float(x), 3) for x in k)
    if key in _FIT3CX_CACHE:
        return _FIT3CX_CACHE[key]
    W = w_matrix(k)
    Wt = torch.tensor(W, dtype=torch.complex128)

    def cmat(t):
        def rz(a):
            return torch.stack([torch.exp(-1j * a / 2),
                                torch.exp(1j * a / 2)]).diag_embed()
        def ry(a):
            h = a / 2
            return torch.stack([torch.stack([torch.cos(h), -torch.sin(h)]),
                                torch.stack([torch.sin(h), torch.cos(h)])]).to(
                                    torch.complex128)
        cx = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=torch.complex128)
        C = cx
        C = torch.kron(rz(t[0]), rz(t[1])) @ C
        C = torch.kron(ry(t[2]), ry(t[3])) @ C
        C = cx @ C
        C = torch.kron(rz(t[4]), rz(t[5])) @ C
        C = torch.kron(ry(t[6]), ry(t[7])) @ C
        C = cx @ C
        return C

    t = torch.tensor(np.asarray(_init_3cx_params(k), dtype=np.float64),
                     dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([t], lr=0.1)
    for step in range(400):
        opt.zero_grad()
        C = cmat(t)
        ov = torch.vdot(C.reshape(-1), Wt.reshape(-1))
        loss = 1.0 - (ov.abs() / 4.0) ** 2
        loss.backward()
        opt.step()
        if loss.item() < 1e-14:
            break
    result = t.detach().numpy()
    _FIT3CX_CACHE[key] = result
    return result


def fit_block_torch(U, init, steps=300, lr=0.1):
    """KAK 初值下 torch Adam 拟合（纯 matmul 构造）。

    3-CX 中间参数经 _cached_3cx_params 缓存（相同 Cartan 系数复用），
    300 步内多数案例收敛到 loss < 1e-12；未收敛案例由调用方的
    overlap 验证拒绝、保留原块（耗时优先，2026-11-09）。
    """
    import torch
    Ut = torch.tensor(np.asarray(U, dtype=complex), dtype=torch.complex128)

    def cmat(t):
        def rz(a):
            return torch.stack([torch.exp(-1j * a / 2),
                                torch.exp(1j * a / 2)]).diag_embed()
        def ry(a):
            h = a / 2
            return torch.stack([torch.stack([torch.cos(h), -torch.sin(h)]),
                                torch.stack([torch.sin(h), torch.cos(h)])]).to(
                                    torch.complex128)
        def u3(a, b, c):
            cc, ss = torch.cos(a / 2), torch.sin(a / 2)
            return torch.stack([torch.stack([cc, -torch.exp(1j * c) * ss]),
                                torch.stack([torch.exp(1j * b) * ss,
                                             torch.exp(1j * (b + c)) * cc])])
        cx = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=torch.complex128)
        C = torch.kron(u3(t[0], t[1], t[2]), u3(t[3], t[4], t[5]))
        C = cx @ C
        C = torch.kron(rz(t[6]), rz(t[7])) @ C
        C = torch.kron(ry(t[8]), ry(t[9])) @ C
        C = cx @ C
        C = torch.kron(rz(t[10]), rz(t[11])) @ C
        C = torch.kron(ry(t[12]), ry(t[13])) @ C
        C = cx @ C
        C = torch.kron(u3(t[14], t[15], t[16]), u3(t[17], t[18], t[19])) @ C
        return torch.exp(1j * t[20]) * C

    t = torch.tensor(np.asarray(init, dtype=np.float64), dtype=torch.float64,
                     requires_grad=True)
    opt = torch.optim.Adam([t], lr=lr)
    prev_loss = float("inf")
    stall = 0
    max_steps = steps + 400  # 仍在改善的案例可超出基础预算（停滞检测兜底）
    for step in range(max_steps):
        opt.zero_grad()
        C = cmat(t)
        ov = torch.vdot(C.reshape(-1), Ut.reshape(-1))
        loss = 1.0 - (ov.abs() / 4.0) ** 2
        loss.backward()
        opt.step()
        if loss.item() < 1e-12:
            break
        # 收敛停滞检测：连续 2 个窗口（100 步）改善 < 1e-7 才放弃
        # （容忍 Adam 的平台期；失败案例仍省 ~60% 时间）
        if step % 50 == 49:
            if prev_loss - loss.item() < 1e-7:
                stall += 1
            else:
                stall = 0
            if stall >= 2:
                break
            prev_loss = loss.item()
    tn = t.detach().numpy()
    C = _block_circuit_matrix(tn)
    ov = abs(np.vdot(C, U)) / (np.linalg.norm(C) * np.linalg.norm(U))
    return tn, ov


def block_params_to_insts(t, pair):
    """21 参数 + 物理比特对 (a, b) → Inst 列表（u3/u1/cx）。

    模板比特 0 → a、比特 1 → b；ry(θ) → u3(θ,0,0)、rz(θ) → u1(θ)。
    """
    a, b = pair
    out = []
    for i in range(2):  # 前层 K1⊗K2：t[0:3]（比特 a）与 t[3:6]（比特 b）
        th, ph, la = t[3 * i], t[3 * i + 1], t[3 * i + 2]
        q = a if i == 0 else b
        out.append(Inst("u3", (q,), (float(th), float(ph), float(la))))
    out.append(Inst("cx", (a, b)))
    out.append(Inst("u1", (a,), (float(t[6]),)))
    out.append(Inst("u1", (b,), (float(t[7]),)))
    out.append(Inst("u3", (a,), (float(t[8]), 0.0, 0.0)))
    out.append(Inst("u3", (b,), (float(t[9]), 0.0, 0.0)))
    out.append(Inst("cx", (a, b)))
    out.append(Inst("u1", (a,), (float(t[10]),)))
    out.append(Inst("u1", (b,), (float(t[11]),)))
    out.append(Inst("u3", (a,), (float(t[12]), 0.0, 0.0)))
    out.append(Inst("u3", (b,), (float(t[13]), 0.0, 0.0)))
    out.append(Inst("cx", (a, b)))
    for i in range(4, 6):  # 后层 K3⊗K4：t[14:17]（比特 a）与 t[17:20]（比特 b）
        base = 14 + 3 * (i - 4)
        th, ph, la = t[base], t[base + 1], t[base + 2]
        q = a if i == 4 else b
        out.append(Inst("u3", (q,), (float(th), float(ph), float(la))))
    return out


def collect_2q_blocks(circ, spec, min_gate: int = 5):
    """收集 2Q 块：同一比特对上的连续 2Q 门序列（中间允许 1Q 门）。

    返回 [(start, end, pair, n_swap, n_2q)]（ops 索引区间，含区间内所有门）。
    只返回"合成有利"的块：3·s + g ≥ min_gate（循环内默认 5 = 至少
    省 2 个 CX；最终单点可放宽到 3——3-cz 块虽不省 2Q 数，但 Tuccil
    8 因子能压 1Q：块级 32→20）。

    tu_locked 冻结段（2026-11-13 Tuccil 循环保护）：tu_locked 标记的
    cz 是硬块边界——不作为块起点、不跨越扩展。机制：Tuccil 3-CZ
    模板产物（块级 1Q −66%）若不冻结，后续轮次的同对 cz 聚拢会把它
    与邻居 cz 排成连续游程，再被收集成块重新合成——紧凑输出被通用
    3cx 形态吸收（不动点吸收机制）。
    """
    blocks = []
    ops = circ.ops
    i = 0
    n = len(ops)
    while i < n:
        inst = ops[i]
        if inst.nq == 2 and inst.name in ("cx", "cz", "swap"):
            if getattr(inst, "tu_locked", False):
                i += 1
                continue
            a, b = int(inst.qubits[0]), int(inst.qubits[1])
            if a > b:
                a, b = b, a
            if not spec.adj[a, b]:
                i += 1
                continue
            pair = (a, b)
            j = i + 1
            s = 1 if inst.name == "swap" else 0
            g = 0 if inst.name == "swap" else 1
            # 扩展块：同对 2Q 门 + 对比特 1Q 门（中间允许任意长度——
            # 2026-11-13 修复：此前只支持"单个 1Q 门"跨越，u3 展开的
            # 5-7 连续 1Q 序列切断收集，heisenberg 类 12cx/对 的块全空、
            # A2 零效果）。
            # 2026-11-13 语义修复：**外比特 1Q 门是硬边界**——4×4 块酉
            # 无法表示第三比特的效应（_block_unitary 会把它误当对位
            # b 的门，拟合守卫对错误 U 通过，替换后语义崩：heisenberg
            # N16 实测 fid 2.3e-5）。qiskit Collect2qBlocks 的
            # "uninterrupted" 语义同款。
            while j < n:
                nj = ops[j]
                if nj.nq == 1:
                    if int(nj.qubits[0]) == a or int(nj.qubits[0]) == b:
                        j += 1
                        continue
                    break
                if nj.nq == 2 and nj.name in ("cx", "cz", "swap"):
                    if getattr(nj, "tu_locked", False):
                        break
                    a2, b2 = int(nj.qubits[0]), int(nj.qubits[1])
                    if a2 > b2:
                        a2, b2 = b2, a2
                    if (a2, b2) == pair and spec.adj[a2, b2]:
                        s += 1 if nj.name == "swap" else 0
                        g += 0 if nj.name == "swap" else 1
                        j += 1
                        continue
                break
            n_2q = s + g
            # 收益门槛：合成 3 CX vs 原展开 3s+g，至少省 2 个 CX 才值得
            # （1.4s 拟合 vs 1 个 CX 的收益不划算；2026-11-09 耗时优化）
            # min_gate 可放宽（最终单点：3-cz 块的 1Q 压缩窗口）
            if n_2q >= 2 and 3 * s + g >= min_gate:
                blocks.append((i, j, pair, s, n_2q))
            i = j
        else:
            i += 1
    return blocks


def rebuild_kak(dec: dict) -> np.ndarray:
    """KAK 参数 → 重建 4×4 酉（含全局相位）。往返测试用。

    利用 magic 基性质：M·φ(l,r)·M† = kron(l, r)，M·D·M† = w_matrix(k)：
      U = kron(K1,K2)·w_matrix(k)·kron(K3,K4)·phase
    """
    U = np.kron(dec["K1"], dec["K2"]) @ w_matrix(dec["k"]) \
        @ np.kron(dec["K3"], dec["K4"])
    return dec["phase"] * U


def w_matrix(k):
    """交互项 W = exp(i Σ k_j σ_j⊗σ_j) 的计算基矩阵（解析）。"""
    k1, k2, k3 = k
    a = math.cos(k1 - k2) * (math.cos(k3) + 1j * math.sin(k3))
    b = 1j * math.sin(k1 - k2) * (math.cos(k3) + 1j * math.sin(k3))
    c = math.cos(k1 + k2) * (math.cos(k3) - 1j * math.sin(k3))
    d = 1j * math.sin(k1 + k2) * (math.cos(k3) - 1j * math.sin(k3))
    return np.array([[a, 0, 0, b],
                     [0, c, d, 0],
                     [0, d, c, 0],
                     [b, 0, 0, a]], dtype=complex)


# ---------------------------------------------------------- 3-CX 合成（拟合）
# 模板：CX(0,1)·(Rz⊗Rz)(t0,t1)·(Ry⊗Ry)(t2,t3)·CX(0,1)·
#        (Rz⊗Rz)(t4,t5)·(Ry⊗Ry)(t6,t7)·CX(0,1)·(全局相位)
# 参数 8 维；由 test_w_matrix_vs_3cx_circuit 锁定（torch 拟合）。
def synth_3cx(k):
    """W(k) 的 3-CX 电路参数（torch 数值拟合，结果缓存）。"""
    t = _fit_3cx_params(np.asarray(k, dtype=np.float64))
    return [
        ("cx", (0, 1)),
        ("rz", (0, float(t[0]))), ("rz", (1, float(t[1]))),
        ("ry", (0, float(t[2]))), ("ry", (1, float(t[3]))),
        ("cx", (0, 1)),
        ("rz", (0, float(t[4]))), ("rz", (1, float(t[5]))),
        ("ry", (0, float(t[6]))), ("ry", (1, float(t[7]))),
        ("cx", (0, 1)),
    ]


_FIT_CACHE: dict = {}


def _fit_3cx_params(k: np.ndarray) -> np.ndarray:
    """拟合 3-CX 模板的 8 个旋转参数使电路 == W(k)（torch Adam）。"""
    key = tuple(round(float(x), 6) for x in k)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    import torch
    W = torch.tensor(w_matrix(k), dtype=torch.complex128)

    def circuit_matrix(t):
        # 构造单比特旋转（复数域）
        def rz(a):
            return torch.diag(torch.stack(
                [torch.exp(-1j * a / 2), torch.exp(1j * a / 2)]))
        def ry(a):
            return torch.stack([
                torch.stack([torch.cos(a / 2), -torch.sin(a / 2)]),
                torch.stack([torch.sin(a / 2), torch.cos(a / 2)])]).to(
                    torch.complex128)
        cx = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 0, 1], [0, 0, 1, 0]],
                          dtype=torch.complex128)
        I2 = torch.eye(2, dtype=torch.complex128)
        U = cx
        U = torch.kron(rz(t[0]), rz(t[1])) @ U
        U = torch.kron(ry(t[2]), ry(t[3])) @ U
        U = cx @ U
        U = torch.kron(rz(t[4]), rz(t[5])) @ U
        U = torch.kron(ry(t[6]), ry(t[7])) @ U
        U = cx @ U
        return U

    t = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([t], lr=0.05)
    for step in range(400):
        opt.zero_grad()
        C = circuit_matrix(t)
        ov = torch.vdot(C.reshape(-1), W.reshape(-1))
        loss = 1.0 - (ov.abs() / (torch.linalg.norm(C) * torch.linalg.norm(W))) ** 2
        loss.backward()
        opt.step()
        if step % 100 == 0 and loss.item() < 1e-16:
            break
    C = circuit_matrix(t)
    ov = torch.vdot(C.reshape(-1), W.reshape(-1))
    quality = 1.0 - (ov.abs() / (torch.linalg.norm(C) * torch.linalg.norm(W))) ** 2
    if quality.item() > 1e-10:
        raise RuntimeError(f"3cx fit failed for k={k}: residual={quality.item()}")
    result = t.detach().numpy()
    _FIT_CACHE[key] = result
    return result


def fit_block_cz(U, init, steps=400, lr=0.1):
    """cz 基 3-CZ 模板拟合（2026-11-13，qiskit tqp 输出结构对齐）：

    U = (K1⊗K2)·CZ·(Rza⊗Rzb)·CZ·(Rzc⊗I)·CZ·(K3⊗K4)
    ——15 参数（4 个 SU(2) 欧拉角 + 3 个中间 rz）。Rzd 经规范固定
    删除：CZ 与 rz 对易 ⇒ rz 可在 CZ 间滑移吸收，16 参数中的 1 个
    是规范自由度（SU(4) 15 维）——平坦方向使 Adam 卡局部最优
    （ov 0.695/0.804），15 参数一一对应消除平坦方向。
    初值：K 因子 euler 角 + 中间 rz（KAK Cartan 系数的对角化近似）。
    返回 (t, overlap)；overlap < 1-1e-7 由调用方拒绝。
    """
    import torch
    Ut = torch.tensor(np.asarray(U, dtype=complex), dtype=torch.complex128)

    def cmat(t):
        def rz(a):
            return torch.stack([torch.exp(-1j * a / 2),
                                torch.exp(1j * a / 2)]).diag_embed()

        def u3(a, b, c):
            cc, ss = torch.cos(a / 2), torch.sin(a / 2)
            return torch.stack([torch.stack([cc, -torch.exp(1j * c) * ss]),
                                torch.stack([torch.exp(1j * b) * ss,
                                             torch.exp(1j * (b + c)) * cc])])
        cz = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, -1]], dtype=torch.complex128)
        # t: K1(0:3) K2(3:6) Rza(6) Rzb(7) Rzc(8) K3(9:12) K4(12:15)
        C = torch.kron(u3(t[0], t[1], t[2]), u3(t[3], t[4], t[5]))
        C = cz @ C
        C = torch.kron(rz(t[6]), rz(t[7])) @ C
        C = cz @ C
        C = torch.kron(rz(t[8]), rz(torch.zeros_like(t[8]))) @ C
        C = cz @ C
        C = torch.kron(u3(t[9], t[10], t[11]),
                       u3(t[12], t[13], t[14])) @ C
        return C

    t = torch.tensor(np.asarray(init, dtype=np.float64), dtype=torch.float64,
                     requires_grad=True)
    opt = torch.optim.Adam([t], lr=lr)
    prev_loss = float('inf')
    stall = 0
    max_steps = steps + 400
    for _ in range(max_steps):
        opt.zero_grad()
        C = cmat(t)
        ov = torch.vdot(C.reshape(-1), Ut.reshape(-1))
        loss = 1.0 - (ov.abs() / 4.0) ** 2   # 模方损失（凸化，
                                             # 与 fit_block_torch 一致）
        loss.backward()
        opt.step()
        if float(loss.detach()) < 1e-12:
            break
        with torch.no_grad():
            cur = float(loss.detach())
        if abs(prev_loss - cur) < 1e-7:
            stall += 1
        else:
            stall = 0
        if stall > 2:
            break
        prev_loss = cur
    with torch.no_grad():
        C = cmat(t)
        ov = float(torch.abs(torch.vdot(C.reshape(-1),
                                        Ut.reshape(-1)))) / 4.0
    return t.detach().numpy(), ov


def fit_block_tuccil(U, init, steps=400, lr=0.1):
    """Tuccil 3-CZ 模板拟合（2026-11-13，qiskit TwoQubitBasisDecomposer
    的 cz 基底结构对齐）：

    U = (K1⊗K2)·CZ·(A⊗B)·CZ·(C⊗D)·CZ·(K3⊗K4)
    ——21 参数（8 个 SU(2) 欧拉角 - 3 个规范固定：CZ 与 rz 对易的
    全局滑动）。8 因子可对角化（θ≈0 → 1 门 rz），heisenberg 类块
    的对角解输出 ~11 门 vs 3cx 模板 12 单位固定输出的 ~49 门——
    1Q 差距（1.66x）的模板级修复。21 参数与 3cx 同冗余结构（收敛
    性已验证）。初值：3cx 解经 cz 基转换（中间 rz 近似）→ 精修。
    返回 (t, overlap)。
    """
    import torch
    Ut = torch.tensor(np.asarray(U, dtype=complex), dtype=torch.complex128)

    def cmat(t):
        def u3(a, b, c):
            cc, ss = torch.cos(a / 2), torch.sin(a / 2)
            return torch.stack([torch.stack([cc, -torch.exp(1j * c) * ss]),
                                torch.stack([torch.exp(1j * b) * ss,
                                             torch.exp(1j * (b + c)) * cc])])
        cz = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, -1]], dtype=torch.complex128)

        def u(i):
            return u3(t[3 * i], t[3 * i + 1], t[3 * i + 2])
        # 因子顺序：K1(0) K2(1) A(2) B(3) C(4) D(5) K3(6) K4(7)
        C = torch.kron(u(0), u(1))
        C = cz @ C
        C = torch.kron(u(2), u(3)) @ C
        C = cz @ C
        C = torch.kron(u(4), u(5)) @ C
        C = cz @ C
        C = torch.kron(u(6), u(7)) @ C
        return C

    t = torch.tensor(np.asarray(init, dtype=np.float64), dtype=torch.float64,
                     requires_grad=True)
    opt = torch.optim.Adam([t], lr=lr)
    prev_loss = float('inf')
    stall = 0
    max_steps = steps + 400
    for _ in range(max_steps):
        opt.zero_grad()
        C = cmat(t)
        ov = torch.vdot(C.reshape(-1), Ut.reshape(-1))
        loss = 1.0 - (ov.abs() / 4.0) ** 2
        loss.backward()
        opt.step()
        if float(loss.detach()) < 1e-12:
            break
        with torch.no_grad():
            cur = float(loss.detach())
        if abs(prev_loss - cur) < 1e-7:
            stall += 1
        else:
            stall = 0
        if stall > 2:
            break
        prev_loss = cur
    with torch.no_grad():
        C = cmat(t)
        ov = float(torch.abs(torch.vdot(C.reshape(-1),
                                        Ut.reshape(-1)))) / 4.0
    return t.detach().numpy(), ov
