"""后处理栈热循环 Numba 编译版（push_swaps 流式压实重写）。

与 post.py 的 Python 版**逐位等价**：同一套重写规则、同一轮次结构、
同一扫描顺序；线路表示为 (name 码, q0, q1) 并行数组 + 源索引映射
（重写不改变门参数，参数按源索引对齐回填）。无 Numba 时回退
Python 原版（功能等价）。

规则（与 _swap_past 一致）：
  SWAP(a,b); SWAP(a,b) → 抵消（同边无序对）
  SWAP(a,b); CX(a,c)   → CX(b,c);  SWAP(a,b)（单边触碰：重标记）
  SWAP(a,b); CX(a,b)   → CX(b,a);  SWAP(a,b)（双边触碰：控/靶翻转）
  SWAP(a,b); CZ 单边触碰 → 重标记；双边/不相交 → 不变
  单比特门触碰 → 重标记；不相交 → 对易
  其余（未知名）→ 阻塞
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # noqa: BLE001
    _HAS_NUMBA = False

# 门名编码（与 qtrail.pure.circuit.Inst 一一对应）
U3, U1, CX, CZ, SWAP, UNK = 0, 1, 2, 3, 4, 5
_NAME_TO_CODE = {"u3": U3, "u1": U1, "cx": CX, "cz": CZ, "swap": SWAP}
_CODE_TO_NAME = {U3: "u3", U1: "u1", CX: "cx", CZ: "cz", SWAP: "swap"}


if _HAS_NUMBA:
    @njit(cache=True)
    def _swap_past_arr(a, b, gname, gq0, gq1):
        """swap(a,b) 穿过门 g 的重写。返回 (ok, name', q0', q1')。"""
        if gname == CX:
            if (gq0 != a and gq0 != b and gq1 != a and gq1 != b):
                return True, CX, gq0, gq1
            if (gq0 == a or gq0 == b) and (gq1 == a or gq1 == b):
                return True, CX, gq1, gq0
            q0 = b if gq0 == a else (a if gq0 == b else gq0)
            q1 = b if gq1 == a else (a if gq1 == b else gq1)
            return True, CX, q0, q1
        if gname == CZ or gname == SWAP:
            # 对称 2Q 门（cz / 异边 swap）：单边触碰重标记；双边（同边已被
            # 消解检查拦截）与不相交 → 不变。Python 参考版同款通用分支。
            if (gq0 == a or gq0 == b) and (gq1 == a or gq1 == b):
                return True, gname, gq0, gq1
            if gq0 != a and gq0 != b and gq1 != a and gq1 != b:
                return True, gname, gq0, gq1
            q0 = b if gq0 == a else (a if gq0 == b else gq0)
            q1 = b if gq1 == a else (a if gq1 == b else gq1)
            return True, gname, q0, q1
        if gname == U3 or gname == U1:
            q = gq0
            if q == a:
                return True, gname, b, gq1
            if q == b:
                return True, gname, a, gq1
            return True, gname, gq0, gq1
        return False, gname, gq0, gq1

    @njit(cache=True)
    def _push_round(nm, a0, a1, bnm, ba0, ba1, bsrc, src, n):
        """一轮单向推挤 + 消解（流式压实到 b* 缓冲）。
        返回 (新长度, 是否变化)。与 Python 版 push_swaps 单轮逐位等价。

        循环结构对照 Python：指针 i 指向「待比较对」的第一个元素；
        pending=已登记的待推交换（Python 中位于 ops[i]），此时待穿门
        在源位置 i。末对 (n-2, n-1) 在 pending 建立后也必须处理，
        故条件用 i < n 而非 i < n-1。
        """
        w = 0
        i = 0
        changed = False
        pending = False
        pa = pb = pa_src = -1
        while i < n:
            if not pending:
                if i == n - 1:
                    # 仅剩最后一个元素（Python 循环条件 i < len-1 直接
                    # 结束、该元素原地保留）：落位收尾
                    bnm[w] = nm[i]
                    ba0[w] = a0[i]
                    ba1[w] = a1[i]
                    bsrc[w] = src[i]
                    w += 1
                    i += 1
                    break
                if nm[i] != SWAP:
                    bnm[w] = nm[i]
                    ba0[w] = a0[i]
                    ba1[w] = a1[i]
                    bsrc[w] = src[i]
                    w += 1
                    i += 1
                    continue
                pa = a0[i]
                pb = a1[i]
                pa_src = src[i]
                pending = True
                i += 1  # 待推交换已登记；待穿门在 i（原 i+1）
                continue
            # ---- pending：处理 (swap(pa,pb), nm[i])
            if i >= n:
                break  # 交换位于列表末尾（Python：i<len-1 条件退出、原地保留）
            a, b = pa, pb
            gname = nm[i]
            gq0 = a0[i]
            gq1 = a1[i]
            if gname == SWAP and ((gq0 == a and gq1 == b)
                                  or (gq0 == b and gq1 == a)):
                pending = False
                i += 1
                changed = True
                continue
            ok, rname, rq0, rq1 = _swap_past_arr(a, b, gname, gq0, gq1)
            if ok:
                bnm[w] = rname
                ba0[w] = rq0
                ba1[w] = rq1
                bsrc[w] = src[i]
                w += 1
                i += 1
                changed = True
                continue
            # 阻塞：swap 落位，下一元素（阻塞门）留待下轮比较
            bnm[w] = SWAP
            ba0[w] = a
            ba1[w] = b
            bsrc[w] = pa_src
            w += 1
            pending = False
        if pending:
            bnm[w] = SWAP
            ba0[w] = pa
            ba1[w] = pb
            bsrc[w] = pa_src
            w += 1
        while i < n:
            bnm[w] = nm[i]
            ba0[w] = a0[i]
            ba1[w] = a1[i]
            bsrc[w] = src[i]
            w += 1
            i += 1
        return w, changed

    @njit(cache=True)
    def _push_swaps_kernel(names, q0, q1, max_rounds):
        """多轮推挤至不动点。返回 (最终名字数组, q0, q1, 源索引)。"""
        L = names.shape[0]
        nm = names.copy()
        a0 = q0.copy()
        a1 = q1.copy()
        src = np.arange(L, dtype=np.int64)
        bnm = np.empty(L, dtype=np.int64)
        ba0 = np.empty(L, dtype=np.int64)
        ba1 = np.empty(L, dtype=np.int64)
        bsrc = np.empty(L, dtype=np.int64)
        n = L
        changed = True
        rounds = 0
        while changed and rounds < max_rounds:
            rounds += 1
            n, changed = _push_round(nm, a0, a1, bnm, ba0, ba1, bsrc, src, n)
            nm, bnm = bnm, nm
            a0, ba0 = ba0, a0
            a1, ba1 = ba1, a1
            src, bsrc = bsrc, src
        return nm[:n], a0[:n], a1[:n], src[:n]


def _circ_to_arrays(ops):
    L = len(ops)
    names = np.empty(L, dtype=np.int64)
    q0 = np.zeros(L, dtype=np.int64)
    q1 = np.zeros(L, dtype=np.int64)
    for i, inst in enumerate(ops):
        names[i] = _NAME_TO_CODE.get(inst.name, UNK)
        q0[i] = inst.qubits[0]
        q1[i] = inst.qubits[1] if inst.nq == 2 else -1
    return names, q0, q1


def numba_push_swaps(circ, max_rounds: int = 200):
    """push_swaps 的 Numba 版：返回重写后的 ops 列表（Inst）。"""
    from qtrail.pure.circuit import Inst
    ops = circ.ops
    if not ops:
        return list(ops)
    names, q0, q1 = _circ_to_arrays(ops)
    nm, nq0, nq1, src = _push_swaps_kernel(names, q0, q1, max_rounds)
    return _rebuild(ops, nm, nq0, nq1, src)


def _rebuild(ops, nm, nq0, nq1, src):
    from qtrail.pure.circuit import Inst
    out = []
    for k in range(nm.shape[0]):
        s = int(src[k])
        inst = ops[s]
        code = int(nm[k])
        name = _CODE_TO_NAME.get(code, inst.name)
        out.append(Inst(name,
                        (int(nq0[k]),) if code in (U3, U1) else
                        (int(nq0[k]), int(nq1[k])),
                        inst.params))
    return out


if _HAS_NUMBA:
    @njit(cache=True)
    def _swap_past_arr_safe(a, b, gname, gq0, gq1, amat, N):
        """连通性感知版 _swap_past_arr：触碰 2Q 门的重标记仅当新比特对
        仍在耦合图上（amat 校验）时返回 ok=True，否则阻塞。"""
        if gname == CX:
            if (gq0 != a and gq0 != b and gq1 != a and gq1 != b):
                return True, CX, gq0, gq1
            if (gq0 == a or gq0 == b) and (gq1 == a or gq1 == b):
                return True, CX, gq1, gq0          # 同对翻转：恒合法
            q0 = b if gq0 == a else (a if gq0 == b else gq0)
            q1 = b if gq1 == a else (a if gq1 == b else gq1)
            if amat[q0 * N + q1]:
                return True, CX, q0, q1
            return False, gname, gq0, gq1
        if gname == CZ:
            if (gq0 == a or gq0 == b) and (gq1 == a or gq1 == b):
                return True, CZ, gq0, gq1
            if gq0 != a and gq0 != b and gq1 != a and gq1 != b:
                return True, CZ, gq0, gq1
            q0 = b if gq0 == a else (a if gq0 == b else gq0)
            q1 = b if gq1 == a else (a if gq1 == b else gq1)
            if amat[q0 * N + q1]:
                return True, CZ, q0, q1
            return False, gname, gq0, gq1
        if gname == SWAP:
            if (gq0 == a or gq0 == b) and (gq1 == a or gq1 == b):
                return True, gname, gq0, gq1
            if gq0 != a and gq0 != b and gq1 != a and gq1 != b:
                return True, gname, gq0, gq1
            q0 = b if gq0 == a else (a if gq0 == b else gq0)
            q1 = b if gq1 == a else (a if gq1 == b else gq1)
            return True, gname, q0, q1          # swap 无邻近约束
        if gname == U3 or gname == U1:
            q = gq0
            if q == a:
                return True, gname, b, gq1
            if q == b:
                return True, gname, a, gq1
            return True, gname, gq0, gq1
        return False, gname, gq0, gq1

    @njit(cache=True)
    def _push_round_safe(nm, a0, a1, bnm, ba0, ba1, bsrc, src, n, amat, N):
        """与 _push_round 同结构，_swap_past_arr 换连通性感知版。"""
        w = 0
        i = 0
        changed = False
        pending = False
        pa = pb = pa_src = -1
        while i < n:
            if not pending:
                if i == n - 1:
                    bnm[w] = nm[i]
                    ba0[w] = a0[i]
                    ba1[w] = a1[i]
                    bsrc[w] = src[i]
                    w += 1
                    i += 1
                    break
                if nm[i] != SWAP:
                    bnm[w] = nm[i]
                    ba0[w] = a0[i]
                    ba1[w] = a1[i]
                    bsrc[w] = src[i]
                    w += 1
                    i += 1
                    continue
                pa = a0[i]
                pb = a1[i]
                pa_src = src[i]
                pending = True
                i += 1
                continue
            if i >= n:
                break
            a, b = pa, pb
            gname = nm[i]
            gq0 = a0[i]
            gq1 = a1[i]
            if gname == SWAP and ((gq0 == a and gq1 == b)
                                  or (gq0 == b and gq1 == a)):
                pending = False
                i += 1
                changed = True
                continue
            ok, rname, rq0, rq1 = _swap_past_arr_safe(a, b, gname, gq0, gq1,
                                                      amat, N)
            if ok:
                bnm[w] = rname
                ba0[w] = rq0
                ba1[w] = rq1
                bsrc[w] = src[i]
                w += 1
                i += 1
                changed = True
                continue
            bnm[w] = SWAP
            ba0[w] = a
            ba1[w] = b
            bsrc[w] = pa_src
            w += 1
            pending = False
        if pending:
            bnm[w] = SWAP
            ba0[w] = pa
            ba1[w] = pb
            bsrc[w] = pa_src
            w += 1
        while i < n:
            bnm[w] = nm[i]
            ba0[w] = a0[i]
            ba1[w] = a1[i]
            bsrc[w] = src[i]
            w += 1
            i += 1
        return w, changed

    @njit(cache=True)
    def _push_swaps_safe_kernel(names, q0, q1, amat, N, max_rounds):
        L = names.shape[0]
        nm = names.copy()
        a0 = q0.copy()
        a1 = q1.copy()
        src = np.arange(L, dtype=np.int64)
        bnm = np.empty(L, dtype=np.int64)
        ba0 = np.empty(L, dtype=np.int64)
        ba1 = np.empty(L, dtype=np.int64)
        bsrc = np.empty(L, dtype=np.int64)
        n = L
        changed = True
        rounds = 0
        while changed and rounds < max_rounds:
            rounds += 1
            n, changed = _push_round_safe(nm, a0, a1, bnm, ba0, ba1, bsrc,
                                          src, n, amat, N)
            nm, bnm = bnm, nm
            a0, ba0 = ba0, a0
            a1, ba1 = ba1, a1
            src, bsrc = bsrc, src
        return nm[:n], a0[:n], a1[:n], src[:n]


def numba_push_swaps_safe(circ, amat, max_rounds: int = 200):
    """push_swaps_safe 的 Numba 版（连通性感知）。amat: [N,N] bool。"""
    import numpy as np
    ops = circ.ops
    if not ops:
        return list(ops)
    names, q0, q1 = _circ_to_arrays(ops)
    amat = np.asarray(amat, dtype=bool)
    N = amat.shape[0]
    flat = amat.ravel()
    nm, nq0, nq1, src = _push_swaps_safe_kernel(names, q0, q1, flat, N,
                                                max_rounds)
    return _rebuild(ops, nm, nq0, nq1, src)


def has_numba() -> bool:
    return _HAS_NUMBA
