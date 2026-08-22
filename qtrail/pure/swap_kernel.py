"""交换评分内核（Numba 编译）：候选×前沿距离矩阵的融合计算。

与 numpy 版逐位等价：D[i,f] = dist_flat[after(p_a)*N + after(p_b)]，
其中 after(·) 为交换 (ea[i], eb[i]) 后的位置。numpy 版经 after_vec
（np.where 广播）+ 花式索引分两步完成，本内核一次循环直接产出
最终矩阵，消除中间 [C,F] 数组分配——纯工程加速，数值逐位一致
（同一 dist 矩阵的同一元素读取，int64 索引运算无舍入）。

可选编译：无 Numba 时回退纯 Python 循环（功能等价、速度略降），
保证交付管线零硬依赖。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # noqa: BLE001
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(cache=True)
    def _after_dist_flat(ea, eb, q1, q2, dist_flat, N):
        C = ea.shape[0]
        F = q1.shape[0]
        D = np.empty((C, F), dtype=np.int64)
        for i in range(C):
            a = ea[i]
            b = eb[i]
            for f in range(F):
                pa = q1[f]
                pb = q2[f]
                if pa == a:
                    pa = b
                elif pa == b:
                    pa = a
                if pb == a:
                    pb = b
                elif pb == b:
                    pb = a
                D[i, f] = dist_flat[pa * N + pb]
        return D
else:
    def _after_dist_flat(ea, eb, q1, q2, dist_flat, N):
        C = ea.shape[0]
        F = q1.shape[0]
        D = np.empty((C, F), dtype=np.int64)
        for i in range(C):
            a, b = int(ea[i]), int(eb[i])
            for f in range(F):
                pa, pb = int(q1[f]), int(q2[f])
                if pa == a:
                    pa = b
                elif pa == b:
                    pa = a
                if pb == a:
                    pb = b
                elif pb == b:
                    pb = a
                D[i, f] = dist_flat[pa * N + pb]
        return D


def after_dist_matrix(dist_flat, N, q1, q2, ea, eb):
    """[C,F] 距离矩阵：每个候选交换 × 每个前沿/扩展门后的距离。

    与 numpy after_vec + dist[d1,d2_] 逐位一致。
    """
    return _after_dist_flat(ea, eb, q1, q2, dist_flat, N)


def has_numba() -> bool:
    return _HAS_NUMBA
