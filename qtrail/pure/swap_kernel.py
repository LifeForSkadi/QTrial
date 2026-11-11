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
        D = np.empty((C, F), dtype=dist_flat.dtype)
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
        D = np.empty((C, F), dtype=dist_flat.dtype)
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


if _HAS_NUMBA:
    @njit(cache=True)
    def _h2_batch(ea, eb, P1, P2, dist_flat, N):
        """前瞻层批量内核（2026-11-12）：对每候选 c 计算换位后的
        二次前瞻 h2[c] = min_f Σ_f' dist(after_c(f'))（排除 c 自身）。

        替代 router.py 的 `for c in range(len(cand))` Python 循环
        （20 次 numba 边界调用/门）——一次调用内循环全部候选。
        P1/P2 形状 [C,F]：候选 c 交换后前沿门的新位置。"""
        C = ea.shape[0]
        F = P1.shape[1]
        h2 = np.empty(C, dtype=np.float64)
        for c in range(C):
            best = np.inf
            for s in range(C):
                if s == c:
                    continue
                total = 0.0
                a = ea[s]
                b = eb[s]
                for f in range(F):
                    pa = P1[c, f]
                    pb = P2[c, f]
                    if pa == a:
                        pa = b
                    elif pa == b:
                        pa = a
                    if pb == a:
                        pb = b
                    elif pb == b:
                        pb = a
                    total += dist_flat[pa * N + pb]
                if total < best:
                    best = total
            h2[c] = best
        return h2
else:
    def _h2_batch(ea, eb, P1, P2, dist_flat, N):
        C = ea.shape[0]
        F = P1.shape[1]
        h2 = np.empty(C, dtype=np.float64)
        for c in range(C):
            best = np.inf
            for s in range(C):
                if s == c:
                    continue
                total = 0.0
                a, b = int(ea[s]), int(eb[s])
                for f in range(F):
                    pa, pb = int(P1[c, f]), int(P2[c, f])
                    if pa == a:
                        pa = b
                    elif pa == b:
                        pa = a
                    if pb == a:
                        pb = b
                    elif pb == b:
                        pb = a
                    total += dist_flat[pa * N + pb]
                if total < best:
                    best = total
            h2[c] = best
        return h2


def h2_lookahead_batch(ea, eb, P1, P2, dist_flat, N):
    """h2 前瞻批量入口（2026-11-12）：语义与 router.py 的
    Python 循环逐位一致（排除自身、取 min 总和）。"""
    return _h2_batch(ea, eb, P1, P2, dist_flat, N)


def has_numba() -> bool:
    return _HAS_NUMBA
