"""路由主循环 Numba 内核（2026-11-12）：router.sabre_route 的 Python
主循环 → 数组驱动的单函数内核。

语义与 Python 路径逐位一致，依赖以下不变式（已在 router.py 注释中
逐条证明/核对）：
  1. 评分和全为整数（dist 无噪声缩放时 int64），求和顺序无关；
     执行段按门号排序、候选边排序去重、PCG 平局裁决序列逐次一致
     → 两条路径的交换序列与输出逐位相同。
  2. deps[s] ≤ 2（依赖构建只记每比特前一个 2Q 门）→ 双槽表示。
  3. 链成员判定 O(1)：chain_head_of[k] == i ⟺ k 是链 i 的成员。
     注意不可用 chain_prev[k] ≠ -1 判定——链 i 的区间内可夹着
     异对子链（如 {c,d} 链）的成员，其 chain_prev 也非 -1；
     归属表（一次 O(m) 构建）才是正确判据。
  4. ext/l1 成员均为 2Q 门（加入点全部带 _nq==2 守卫）。

PCG64Mcg 为 128 位状态乘性 PRNG（numba 无 int128）→ 内核不持有
PRNG 状态，改为消费 Python 侧预生成的 u64 缓冲并在内核内复刻
gen_range 的 rejection sampling（zone = 2^64 − 2^64 mod n）——
缓冲耗尽返回 action=5 由 Python 续灌，序列与 Python 路径的逐步
调用完全一致。

action 码：0 = 正常完成；1 = 需要防卡死逃逸（Python 回卷+拉近）；
2 = swap 触顶（Python 抛 RuntimeError）；5 = rng 缓冲耗尽（续灌）；
6 = out 数组容量不足（Python 扩容后重入）。

out 采用预分配数组 + 计数器而非 numba typed.List：typed.List 的
tuple 从 Python 侧逐字段读取要走 numba boxing（typeof 派发 ~2μs/
字段——hwb12 44 万门 × 7 字段 ≈ 7s，实测转换段占内核路径 70%）。
数组跨界读取无此开销，逃逸回卷 = 计数器递减，语义不变。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # noqa: BLE001
    _HAS_NUMBA = False


if _HAS_NUMBA:
    from qtrail.pure.swap_kernel import _after_dist_flat, _h2_batch

    _ST_DT = np.dtype([("f_cnt", np.int64), ("e_cnt", np.int64),
                       ("l1_cnt", np.int64), ("b_cnt", np.int64),
                       ("swap_cnt", np.int64), ("since_exec", np.int64),
                       ("decay_steps", np.int64), ("rng_ptr", np.int64),
                       ("out_cnt", np.int64)])

    @njit(cache=True)
    def _sabre_loop(N, m, _q0, _q1, _nq, op_ids, p0, p1, p2, amat,
                    nbrs_head, nbrs_list, dist_flat,
                    chain_next, chain_prev, chain_1q_owner, chain_head_of,
                    deps_a, deps_b, n2q_deps, cnt_af,
                    succ_head, succ_list, in_allowed,
                    wext, wl, lam_ms, lookahead_w,
                    use_decay, decay_inc, decay_reset, has_l1, cand_fext,
                    dbg, attempt_limit, max_swaps, swap_id,
                    f_arr, f_pos, in_f, e_arr, e_pos, in_e,
                    l1_arr, l1_pos, in_l1,
                    exec_mask, pos, inv, decay, last_edge, last_qubit,
                    batch_a, batch_b, rng_buf, st,
                    out_op, out_q0, out_q1, out_p0, out_p1, out_p2,
                    out_nq, out_cap,
                    fq1, fq2, eq1, eq2, lq1, lq2,
                    pkey, ea_, eb_, e_tmp):
        K = rng_buf.shape[0]
        while st[0]["f_cnt"] > 0:
            # ---- 执行段：收集可执行门（快照），按门号排序执行
            ecnt = 0
            fcnt = st[0]["f_cnt"]
            for idx in range(fcnt):
                i = f_arr[idx]
                if _nq[i] == 1 or amat[pos[_q0[i]], pos[_q1[i]]]:
                    e_tmp[ecnt] = i
                    ecnt += 1
            if ecnt > 0:
                st[0]["since_exec"] = 0
                st[0]["b_cnt"] = 0
                st[0]["decay_steps"] = 0
                decay[:] = 1.0
                e_sorted = np.sort(e_tmp[:ecnt])
                for x in range(ecnt):
                    i = e_sorted[x]
                    # 移出 frontier（swap-with-last）
                    fp = f_pos[i]
                    fcnt2 = st[0]["f_cnt"]
                    last_i = f_arr[fcnt2 - 1]
                    f_arr[fp] = last_i
                    f_pos[last_i] = fp
                    st[0]["f_cnt"] = fcnt2 - 1
                    in_f[i] = False
                    exec_mask[i] = True
                    # 链头执行 = 整链按序输出（含链区间内并入的 1Q 门）
                    tail = i
                    while chain_next[tail] != -1:
                        tail = chain_next[tail]
                    for k in range(i, tail + 1):
                        if k == i or chain_head_of[k] == i \
                                or chain_1q_owner[k] == i:
                            if k != i:
                                exec_mask[k] = True
                            oc = st[0]["out_cnt"]
                            if oc >= out_cap:
                                return 6
                            q0p = pos[_q0[k]]
                            out_op[oc] = op_ids[k]
                            out_q0[oc] = q0p
                            out_p0[oc] = p0[k]
                            out_p1[oc] = p1[k]
                            out_p2[oc] = p2[k]
                            if _nq[k] == 1:
                                out_q1[oc] = -1
                                out_nq[oc] = 1
                            else:
                                out_q1[oc] = pos[_q1[k]]
                                out_nq[oc] = 2
                            st[0]["out_cnt"] = oc + 1
                    # 后继更新（qiskit SabreDAG 语义）
                    for j in range(succ_head[i], succ_head[i + 1]):
                        s = succ_list[j]
                        if deps_a[s] == i:
                            deps_a[s] = -1
                        elif deps_b[s] == i:
                            deps_b[s] = -1
                        if deps_a[s] == -1 and deps_b[s] == -1 \
                                and chain_1q_owner[s] == -1:
                            if not in_f[s]:
                                in_f[s] = True
                                f_pos[s] = st[0]["f_cnt"]
                                f_arr[st[0]["f_cnt"]] = s
                                st[0]["f_cnt"] = st[0]["f_cnt"] + 1
                            in_allowed[s] = True
                            if in_e[s]:
                                ep = e_pos[s]
                                ec2 = st[0]["e_cnt"]
                                last_e = e_arr[ec2 - 1]
                                e_arr[ep] = last_e
                                e_pos[last_e] = ep
                                st[0]["e_cnt"] = ec2 - 1
                                in_e[s] = False
                            if in_l1[s]:
                                lp = l1_pos[s]
                                lc = st[0]["l1_cnt"]
                                last_l = l1_arr[lc - 1]
                                l1_arr[lp] = last_l
                                l1_pos[last_l] = lp
                                st[0]["l1_cnt"] = lc - 1
                                in_l1[s] = False
                            for j2 in range(succ_head[s],
                                            succ_head[s + 1]):
                                t = succ_list[j2]
                                dok = (deps_a[t] == -1
                                       or in_allowed[deps_a[t]]) \
                                    and (deps_b[t] == -1
                                         or in_allowed[deps_b[t]])
                                if not in_f[t] and dok \
                                        and chain_prev[t] == -1 \
                                        and _nq[t] == 2:
                                    if not in_e[t]:
                                        in_e[t] = True
                                        e_pos[t] = st[0]["e_cnt"]
                                        e_arr[st[0]["e_cnt"]] = t
                                        st[0]["e_cnt"] = st[0]["e_cnt"] + 1
                                if has_l1 and _nq[s] == 2:
                                    cnt_af[t] += 1
                                    if not in_f[t] and not exec_mask[t] \
                                            and chain_prev[t] == -1 \
                                            and _nq[t] == 2 \
                                            and cnt_af[t] == n2q_deps[t]:
                                        if not in_l1[t]:
                                            in_l1[t] = True
                                            l1_pos[t] = st[0]["l1_cnt"]
                                            l1_arr[st[0]["l1_cnt"]] = t
                                            st[0]["l1_cnt"] = \
                                                st[0]["l1_cnt"] + 1
                        else:
                            dok2 = (deps_a[s] == -1
                                    or in_allowed[deps_a[s]]) \
                                and (deps_b[s] == -1
                                     or in_allowed[deps_b[s]])
                            if dok2 and _nq[s] == 2:
                                if not in_e[s]:
                                    in_e[s] = True
                                    e_pos[s] = st[0]["e_cnt"]
                                    e_arr[st[0]["e_cnt"]] = s
                                    st[0]["e_cnt"] = st[0]["e_cnt"] + 1
                continue

            # ---- 评分段：无门可执行，向量化评分选交换
            if st[0]["since_exec"] >= attempt_limit:
                return 1
            F = 0
            fcnt = st[0]["f_cnt"]
            for idx in range(fcnt):
                i = f_arr[idx]
                if _nq[i] == 2:
                    fq1[F] = pos[_q0[i]]
                    fq2[F] = pos[_q1[i]]
                    F += 1
            if F == 0:
                return 0
            E = 0
            ecnt = st[0]["e_cnt"]
            for idx in range(ecnt):
                i = e_arr[idx]
                if _nq[i] == 2:
                    eq1[E] = pos[_q0[i]]
                    eq2[E] = pos[_q1[i]]
                    E += 1
            # 候选边：前沿门触碰位置的邻接边（排序去重 → 全量扫描同序）
            mark = np.zeros(N, np.bool_)
            for f in range(F):
                mark[fq1[f]] = True
                mark[fq2[f]] = True
            if cand_fext:
                for e in range(E):
                    mark[eq1[e]] = True
                    mark[eq2[e]] = True
            # 与 Python 路径的 pairs 归一化逐位一致：eps 任一端点
            # 触碰的边都加入（(a,b) if b>a else (b,a)），可重复——
            # 排序去重后与 set 语义相同。不可只扫 b>a 半边：
            # 会丢"邻居在 eps、自身不在"的边（heis N16 实测 (3,4)）。
            pcnt = 0
            for a in range(N):
                if mark[a]:
                    for j in range(nbrs_head[a], nbrs_head[a + 1]):
                        b = nbrs_list[j]
                        if b > a:
                            pkey[pcnt] = a * N + b
                        else:
                            pkey[pcnt] = b * N + a
                        pcnt += 1
            if pcnt == 0:
                return 0
            keys = np.sort(pkey[:pcnt])
            C = 0
            prev = -1
            for x in range(pcnt):
                kk = keys[x]
                if kk != prev:
                    prev = kk
                    ea_[C] = kk // N
                    eb_[C] = kk % N
                    C += 1
            Df = _after_dist_flat(ea_[:C], eb_[:C], fq1[:F], fq2[:F],
                                  dist_flat, N)
            basic = Df.sum(axis=1)
            unlocked = (Df == 1).sum(axis=1)
            if E > 0:
                De = _after_dist_flat(ea_[:C], eb_[:C], eq1[:E], eq2[:E],
                                      dist_flat, N)
                ext_cost = De.sum(axis=1)
            else:
                # 与 Python 路径一致：空扩展集时 De 为空矩阵，
                # sum 得 int64 零（保持整数和不变式）
                ext_cost = np.zeros(C, np.int64)
            ms_delta = 1.0 - unlocked / max(F, 1)
            l1_cost = None
            if has_l1 and st[0]["l1_cnt"] > 0:
                L = 0
                lcnt = st[0]["l1_cnt"]
                for idx in range(lcnt):
                    i = l1_arr[idx]
                    lq1[L] = pos[_q0[i]]
                    lq2[L] = pos[_q1[i]]
                    L += 1
                Dl = _after_dist_flat(ea_[:C], eb_[:C], lq1[:L], lq2[:L],
                                      dist_flat, N)
                l1_cost = Dl.sum(axis=1)
            h2 = None
            if lookahead_w > 0.0:
                P1 = np.where(fq1[None, :F] == ea_[:C, None],
                              eb_[:C, None],
                              np.where(fq1[None, :F] == eb_[:C, None],
                                       ea_[:C, None], fq1[None, :F]))
                P2 = np.where(fq2[None, :F] == ea_[:C, None],
                              eb_[:C, None],
                              np.where(fq2[None, :F] == eb_[:C, None],
                                       ea_[:C, None], fq2[None, :F]))
                h2 = _h2_batch(ea_[:C], eb_[:C], P1, P2, dist_flat, N)
            scores = basic + wext * ext_cost
            if h2 is not None:
                scores = scores + lookahead_w * h2
            if l1_cost is not None:
                scores = scores + wl * l1_cost
            if use_decay:
                scores = scores * np.maximum(decay[ea_[:C]],
                                             decay[eb_[:C]])
            scores = scores + lam_ms * ms_delta
            sbest = scores.min()
            tied = np.where(scores <= sbest + 1e-10)[0]
            if dbg:
                print("kb r", st[0]["swap_cnt"], "F", F, "E", E,
                      "Lcnt", st[0]["l1_cnt"], "C", C,
                      "sbest", sbest, "ntied", tied.shape[0])
                print("kb fq1", fq1[:F], "fq2", fq2[:F],
                      "eq1", eq1[:E], "eq2", eq2[:E])
                print("kb cand", ea_[:C], eb_[:C])
                print("kb basic", basic)
                print("kb ext", ext_cost)
                print("kb l1c", l1_cost)
                print("kb scores", scores)
                print("kb tied", tied)
            nt = tied.shape[0]
            # PCG gen_range（rejection sampling，逐位复刻 qtrail.pure.pcg）
            MAXU = np.uint64(0xFFFFFFFFFFFFFFFF)
            r = ((MAXU % np.uint64(nt)) + np.uint64(1)) % np.uint64(nt)
            while True:
                ptr = st[0]["rng_ptr"]
                if ptr >= K:
                    return 5
                v = rng_buf[ptr]
                st[0]["rng_ptr"] = ptr + 1
                if r == np.uint64(0) or v < (np.uint64(0) - r):
                    break
            p = int(v % np.uint64(nt))
            pick = tied[p]
            a = ea_[pick]
            b = eb_[pick]
            oc = st[0]["out_cnt"]
            if oc >= out_cap:
                return 6
            out_op[oc] = swap_id
            out_q0[oc] = a
            out_q1[oc] = b
            out_p0[oc] = 0.0
            out_p1[oc] = 0.0
            out_p2[oc] = 0.0
            out_nq[oc] = 2
            st[0]["out_cnt"] = oc + 1
            st[0]["swap_cnt"] = st[0]["swap_cnt"] + 1
            st[0]["since_exec"] = st[0]["since_exec"] + 1
            batch_a[st[0]["b_cnt"]] = a
            batch_b[st[0]["b_cnt"]] = b
            st[0]["b_cnt"] = st[0]["b_cnt"] + 1
            last_edge[a * N + b] = st[0]["swap_cnt"]
            last_qubit[a] = st[0]["swap_cnt"]
            last_qubit[b] = st[0]["swap_cnt"]
            if use_decay:
                st[0]["decay_steps"] = st[0]["decay_steps"] + 1
                if st[0]["decay_steps"] >= decay_reset:
                    decay[:] = 1.0
                    st[0]["decay_steps"] = 0
                else:
                    decay[a] += decay_inc
                    decay[b] += decay_inc
            la = inv[a]
            lb = inv[b]
            if la >= 0:
                inv[a] = -1
                pos[la] = b
            if lb >= 0:
                inv[b] = -1
                pos[lb] = a
                inv[a] = lb
            if la >= 0:
                inv[b] = la
            if st[0]["swap_cnt"] > max_swaps:
                return 2
        return 0


def has_numba() -> bool:
    return _HAS_NUMBA


def run_numba_route(*, circ, spec, seed, ops, m, _q0, _q1, _nq,
                    amat, nbrs, dist, dist_flat,
                    chain_next, chain_prev, chain_1q_owner, chain_head_of,
                    deps, succ, frontier, allowed, ext, l1,
                    cnt_af, n2q_deps, wext, wl, lam_ms, lookahead_w,
                    decay_inc, decay_reset, cand_scope, attempt_limit,
                    max_swaps, pos, inv, decay, last_edge, last_qubit):
    """内核路径入口（router.sabre_route 调用）：共享结构 → 数组转换 →
    内核循环（逃逸/续灌协作）→ 重建 Circuit。语义与 Python 路径逐位一致。"""
    from qtrail.pure.circuit import Circuit, Inst
    from qtrail.pure.pcg import Pcg64Mcg
    N = spec.n
    # 传入的 Python 列表 → 数组（reflected list 的每元素访问会跨 C
    # 边界回调 typeof——hwb8 实测每门 6 次 typeof ≈ 7s/44 万门，
    # 是内核路径第一大隐藏开销；asarray 后为原生 int64 访问）
    chain_next = np.asarray(chain_next, np.int64)
    chain_prev = np.asarray(chain_prev, np.int64)
    chain_1q_owner = np.asarray(chain_1q_owner, np.int64)
    # 名字表 / 参数元数据
    id_of = {}
    names = []
    npc_of = {}
    for inst in ops:
        nm = inst.name
        if nm not in id_of:
            id_of[nm] = len(names)
            names.append(nm)
            npc_of[nm] = len(inst.params)
    if "swap" not in id_of:
        id_of["swap"] = len(names)
        names.append("swap")
        npc_of["swap"] = 0
    swap_id = id_of["swap"]
    op_ids = np.empty(m, np.int64)
    p0 = np.zeros(m, np.float64)
    p1 = np.zeros(m, np.float64)
    p2 = np.zeros(m, np.float64)
    for i, inst in enumerate(ops):
        op_ids[i] = id_of[inst.name]
        pr = inst.params
        if len(pr) > 0:
            p0[i] = pr[0]
        if len(pr) > 1:
            p1[i] = pr[1]
        if len(pr) > 2:
            p2[i] = pr[2]
    # deps（≤2）→ 双槽
    deps_a = np.full(m, -1, np.int64)
    deps_b = np.full(m, -1, np.int64)
    for i, d in enumerate(deps):
        for j, x in enumerate(d):
            if j == 0:
                deps_a[i] = x
            elif j == 1:
                deps_b[i] = x
            else:
                raise AssertionError(f"deps[{i}] 超过 2 项——内核表示不兼容")
    # succ CSR
    succ_head = np.zeros(m + 1, np.int64)
    for i in range(m):
        succ_head[i + 1] = succ_head[i] + len(succ[i])
    succ_list = np.empty(int(succ_head[m]), np.int64)
    k = 0
    for i in range(m):
        for x in succ[i]:
            succ_list[k] = x
            k += 1
    # 邻接 CSR
    nbrs_head = np.zeros(N + 1, np.int64)
    for a in range(N):
        nbrs_head[a + 1] = nbrs_head[a] + len(nbrs[a])
    nbrs_list = np.empty(int(nbrs_head[N]), np.int64)
    k = 0
    for a in range(N):
        for b in nbrs[a]:
            nbrs_list[k] = b
            k += 1
    # 集合 → 数组+掩码（成员序无关：执行段排序、评分和顺序无关）
    f_arr = np.empty(m, np.int64)
    f_pos = np.full(m, -1, np.int64)
    in_f = np.zeros(m, np.bool_)
    k = 0
    for i in frontier:
        f_arr[k] = i
        f_pos[i] = k
        in_f[i] = True
        k += 1
    e_arr = np.empty(m, np.int64)
    e_pos = np.full(m, -1, np.int64)
    in_e = np.zeros(m, np.bool_)
    k = 0
    for i in ext:
        e_arr[k] = i
        e_pos[i] = k
        in_e[i] = True
        k += 1
    l1_arr = np.empty(m, np.int64)
    l1_pos = np.full(m, -1, np.int64)
    in_l1 = np.zeros(m, np.bool_)
    k = 0
    for i in l1:
        l1_arr[k] = i
        l1_pos[i] = k
        in_l1[i] = True
        k += 1
    in_allowed = np.zeros(m, np.bool_)
    for i in allowed:
        in_allowed[i] = True
    exec_mask = np.zeros(m, np.bool_)
    cnt_af_arr = np.asarray(cnt_af, np.int64)
    n2q_arr = np.asarray(n2q_deps, np.int64)
    chain_head_arr = np.asarray(chain_head_of, np.int64)
    batch_a = np.empty(attempt_limit + 8, np.int64)
    batch_b = np.empty(attempt_limit + 8, np.int64)
    # PCG 缓冲（惰性：首次抽取时续灌）
    pcg = Pcg64Mcg(seed)
    K = 8192
    rng_buf = np.empty(K, np.uint64)
    st = np.zeros(1, dtype=_ST_DT)
    st[0]["f_cnt"] = len(frontier)
    st[0]["e_cnt"] = len(ext)
    st[0]["l1_cnt"] = len(l1)
    st[0]["rng_ptr"] = K
    # out 数组（容量不足时 action=6 扩容重入；门 ≤ m、交换 ≤ max_swaps，
    # 初始余量 64k 覆盖绝大多数线路的交换数）
    out_cap = m + 65536
    out_op = np.empty(out_cap, np.int64)
    out_q0 = np.empty(out_cap, np.int64)
    out_q1 = np.empty(out_cap, np.int64)
    out_p0 = np.empty(out_cap, np.float64)
    out_p1 = np.empty(out_cap, np.float64)
    out_p2 = np.empty(out_cap, np.float64)
    out_nq = np.empty(out_cap, np.int64)
    # 评分段工作区（复用，避免逐次分配）
    fq1 = np.empty(m, np.int64)
    fq2 = np.empty(m, np.int64)
    eq1 = np.empty(m, np.int64)
    eq2 = np.empty(m, np.int64)
    lq1 = np.empty(m, np.int64)
    lq2 = np.empty(m, np.int64)
    ne = int(nbrs_head[N])
    pkey = np.empty(ne, np.int64)
    ea_ = np.empty(ne, np.int64)
    eb_ = np.empty(ne, np.int64)
    e_tmp = np.empty(m, np.int64)
    lam_ms_f = 0.0 if lam_ms is None else float(lam_ms)
    has_l1 = wl != 0.0
    use_decay = decay_inc > 0.0
    cand_fext = (cand_scope == "fext")
    import os as _os
    dbg = bool(_os.environ.get("QTRAIL_RDBG"))

    def grow_out():
        """out 数组双倍扩容（内核返回 6 或逃逸追加前容量不足时）。"""
        nonlocal out_cap, out_op, out_q0, out_q1, out_p0, out_p1, out_p2, out_nq
        new_cap = out_cap * 2
        _grow = np.empty(new_cap - out_cap)
        out_op = np.concatenate([out_op, _grow.astype(np.int64)])
        out_q0 = np.concatenate([out_q0, _grow.astype(np.int64)])
        out_q1 = np.concatenate([out_q1, _grow.astype(np.int64)])
        out_p0 = np.concatenate([out_p0, _grow.astype(np.float64)])
        out_p1 = np.concatenate([out_p1, _grow.astype(np.float64)])
        out_p2 = np.concatenate([out_p2, _grow.astype(np.float64)])
        out_nq = np.concatenate([out_nq, _grow.astype(np.int64)])
        out_cap = new_cap

    def escape_once():
        """防卡死逃逸（qiskit force_enable_closest_node 语义）：
        回卷交换批 + 最近门沿最短路径拉近（与 Python 路径逐行同构）。"""
        bcnt = st[0]["b_cnt"]
        for kk in range(bcnt - 1, -1, -1):
            ra = int(batch_a[kk])
            rb = int(batch_b[kk])
            st[0]["out_cnt"] = st[0]["out_cnt"] - 1
            st[0]["swap_cnt"] = st[0]["swap_cnt"] - 1
            la = int(inv[ra])
            lb = int(inv[rb])
            if la >= 0:
                inv[ra] = -1
                pos[la] = rb
            if lb >= 0:
                inv[rb] = -1
                pos[lb] = ra
            if la >= 0:
                inv[rb] = la
            if lb >= 0:
                inv[ra] = lb
        st[0]["b_cnt"] = 0
        st[0]["since_exec"] = 0
        best_gate = -1
        best_d = 10 ** 9
        for idx in range(st[0]["f_cnt"]):
            i = int(f_arr[idx])
            if _nq[i] == 2:
                d = int(dist[pos[_q0[i]], pos[_q1[i]]])
                if d < best_d:
                    best_d = d
                    best_gate = i
        if best_gate != -1 and best_d > 1:
            q0 = int(pos[_q0[best_gate]])
            q1 = int(pos[_q1[best_gate]])
            from collections import deque
            prev = {q0: None}
            dq = deque([q0])
            while dq:
                u = dq.popleft()
                if u == q1:
                    break
                for v in nbrs[u]:
                    if v not in prev:
                        prev[v] = u
                        dq.append(v)
            if q1 in prev:
                path = []
                u = q1
                while u is not None:
                    path.append(u)
                    u = prev[u]
                path.reverse()
                split = len(path) // 2
                moves = [(path[i], path[i + 1]) for i in range(split)]
                for i in range(split - 1):
                    e = len(path) - 1 - i
                    moves.append((path[e], path[e - 1]))
                for a, b in moves:
                    if st[0]["out_cnt"] >= out_cap:
                        grow_out()
                    oc = st[0]["out_cnt"]
                    out_op[oc] = swap_id
                    out_q0[oc] = a
                    out_q1[oc] = b
                    out_p0[oc] = 0.0
                    out_p1[oc] = 0.0
                    out_p2[oc] = 0.0
                    out_nq[oc] = 2
                    st[0]["out_cnt"] = oc + 1
                    st[0]["swap_cnt"] = st[0]["swap_cnt"] + 1
                    batch_a[st[0]["b_cnt"]] = a
                    batch_b[st[0]["b_cnt"]] = b
                    st[0]["b_cnt"] = st[0]["b_cnt"] + 1
                    la = int(inv[a])
                    lb = int(inv[b])
                    if la >= 0:
                        inv[a] = -1
                        pos[la] = b
                    if lb >= 0:
                        inv[b] = -1
                        pos[lb] = a
                    if la >= 0:
                        inv[b] = la
                    if lb >= 0:
                        inv[a] = lb

    while True:
        act = _sabre_loop(N, m, _q0, _q1, _nq, op_ids, p0, p1, p2, amat,
                          nbrs_head, nbrs_list, dist_flat,
                          chain_next, chain_prev, chain_1q_owner,
                          chain_head_arr,
                          deps_a, deps_b, n2q_arr, cnt_af_arr,
                          succ_head, succ_list, in_allowed,
                          wext, wl, lam_ms_f, lookahead_w,
                          use_decay, decay_inc, decay_reset,
                          has_l1, cand_fext, dbg,
                          attempt_limit, max_swaps, swap_id,
                          f_arr, f_pos, in_f, e_arr, e_pos, in_e,
                          l1_arr, l1_pos, in_l1,
                          exec_mask, pos, inv, decay,
                          last_edge, last_qubit,
                          batch_a, batch_b, rng_buf, st,
                          out_op, out_q0, out_q1, out_p0, out_p1, out_p2,
                          out_nq, out_cap,
                          fq1, fq2, eq1, eq2, lq1, lq2,
                          pkey, ea_, eb_, e_tmp)
        if act == 1:
            escape_once()
        elif act == 5:
            for j in range(K):
                rng_buf[j] = pcg.next_u64()
            st[0]["rng_ptr"] = 0
        elif act == 2:
            raise RuntimeError(
                f"sabre_route swap 触顶 {max_swaps}——线路需要更多"
                f"路由预算，调用方须提高 max_swaps 或降级线路")
        elif act == 6:
            grow_out()
        else:
            break

    routed = Circuit(spec.n, name=circ.name + "_routed")
    n_out = int(st[0]["out_cnt"])
    for idx in range(n_out):
        oid = int(out_op[idx])
        q0 = int(out_q0[idx])
        q1 = int(out_q1[idx])
        nm = names[oid]
        npc = npc_of[nm]
        if npc == 0:
            params = ()
        elif npc == 1:
            params = (float(out_p0[idx]),)
        else:
            params = (float(out_p0[idx]), float(out_p1[idx]),
                      float(out_p2[idx]))
        if int(out_nq[idx]) == 1:
            routed.append(Inst(nm, (q0,), params))
        else:
            routed.append(Inst(nm, (q0, q1), params))
    for inst in circ.measures:
        routed.append(Inst("measure",
                           (int(pos[int(inst.qubits[0])]),),
                           cbits=inst.cbits))
    routed.strip_ids()
    pos_dict = {i: int(pos[i]) for i in range(circ.n)}
    return routed, int(st[0]["swap_cnt"]), pos_dict
