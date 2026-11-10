# -*- coding: utf-8 -*-
"""3-CX 中间 8 参数的高斯-牛顿拟合（替换 _cached_3cx_params 的
400 步 torch Adam——trotter4 超时的直接卡点，2026-11-13）。

数值方法：有限差分雅可比 + 实部正规方程（与 fit_block_circuit
同款——"KAK 初值 3-8 步收敛到 1e-12"）。模板与 _cached_3cx_params
的 cmat 完全一致（cx + RzRz + RyRy + cx + RzRz + RyRy + cx）。
"""
import numpy as np

from qtrail.pure.synthesis import _init_3cx_params, w_matrix


def _cmat_np(t):
    def rz(a):
        return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])

    def ry(a):
        h = a / 2
        return np.array([[np.cos(h), -np.sin(h)],
                         [np.sin(h), np.cos(h)]], dtype=complex)

    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=complex)
    C = cx
    C = np.kron(rz(t[0]), rz(t[1])) @ C
    C = np.kron(ry(t[2]), ry(t[3])) @ C
    C = cx @ C
    C = np.kron(rz(t[4]), rz(t[5])) @ C
    C = np.kron(ry(t[6]), ry(t[7])) @ C
    C = cx @ C
    return C


def fit_3cx_params_gn(k, max_steps=30, tol=1e-12):
    """k（Cartan 系数三元组）→ 8 参数（高斯-牛顿，numpy）。

    返回 (t, overlap)；overlap = |⟨C(t), W(k)⟩|/4。
    """
    W = w_matrix(k)
    t = _init_3cx_params(np.asarray(k, dtype=np.float64)).copy()

    def resid(tt):
        C = _cmat_np(tt)
        ov = np.vdot(C.reshape(-1), W.reshape(-1))
        C = C * (ov / abs(ov))
        return (C - W).reshape(-1)

    h = 1e-7
    for _ in range(max_steps):
        r = resid(t)
        loss = np.linalg.norm(r)
        if loss < tol:
            break
        J = np.zeros((16, 8), dtype=complex)
        for j in range(8):
            tp = t.copy()
            tp[j] += h
            J[:, j] = (resid(tp) - r) / h
        JtJ = np.real(J.conj().T @ J)
        Jtr = np.real(J.conj().T @ r)
        try:
            step = np.linalg.solve(JtJ + 1e-12 * np.eye(8), Jtr)
        except np.linalg.LinAlgError:
            break
        # 阻尼：loss 不降则减半步长（GN 在小模板上几乎总是满步收敛）
        t_new = t - step
        if np.linalg.norm(resid(t_new)) > loss:
            t_new = t - 0.5 * step
        t = t_new
    C = _cmat_np(t)
    ov = abs(np.vdot(C.reshape(-1), W.reshape(-1))) / 4.0
    return t, ov
