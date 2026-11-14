"""自研映射管线（除可选的 2Q 块合成后端外零第三方运行时依赖）：
纯 Circuit 输入 → RL 布局 → 自研 SABRE 路由 → 后处理栈 → CZ 基输出
（qtrail/pure 包顶层入口）。

后端披露（2026-11-12 学术审查 P1-2）：A2 2Q 块合成默认调用
qiskit 的 Weyl/KAK 闭式分解器（快路径，可选）；qiskit 缺失时自动
回退自研 torch 合成。除该数值组件外全部决策逻辑与优化 pass 自研。

复用同包组件：problems（程序图）、search（多起点/LS）、
models（RL 策略，纯 PyTorch）、devices（DeviceSpec）；竞赛评分路由 =
自研 sabre_route（廉价代理 + 最终执行一体）。
"""
from __future__ import annotations

import os
import time

import numpy as np

from qtrail.pure.candidates import CandidateResult
from qtrail.pure.circuit import Circuit
from qtrail.pure.metrics import compute_metrics as pure_metrics
from qtrail.pure.post import decompose_to_platform, post_route
from qtrail.pure.router import sabre_route
from qtrail.pure.layout import heuristic_layout

# 无 qiskit 依赖的复用组件
from qtrail.problems import build_program_graph
from qtrail.search.decoding import multistart_decode
from qtrail.search.local_search import improve_layout


def _ops_from_circuit(circ: Circuit) -> list:
    """纯 Circuit → (name, qubits) 列表（与 extract_ops 同格式）。"""
    return [(inst.name, inst.qubits) for inst in circ.ops
            if inst.name not in ("barrier", "id")]


class PureMapper:
    """状态化纯管线映射器（API 与主 Mapper 对齐）。"""

    def __init__(self, spec, policy=None, cfg=None, dev="cuda", seed=0,
                 selection_rule="swap", lam_ms=0.0, use_post=True,
                 routing_seeds=1, fidelity_predictor=None,
                 pipeline: str = "v4", full_eval: bool = True,
                 full_eval_max_n: int = 50, full_eval_topk: int = 6,
                 fold_keep_on: bool = True,
                 linear_decompose: bool = True):
        from qtrail.config import Config
        self.spec = spec
        self.policy = policy
        self.cfg = cfg or Config()
        self.dev = dev
        self.seed = seed
        if selection_rule not in ("swap", "fidelity", "depth"):
            raise ValueError(f"unknown selection rule: {selection_rule}")
        if pipeline not in ("v4", "v5", "v6"):
            raise ValueError(f"unknown pipeline: {pipeline} (v4 | v5 | v6)")
        self.pipeline = pipeline
        self.selection_rule = selection_rule
        self.lam_ms = lam_ms
        self.use_post = use_post
        self.routing_seeds = routing_seeds
        self.fidelity_predictor = fidelity_predictor
        # 管线级联合选择（2026-11-11）：≤ full_eval_max_n 比特的线路，
        # 候选布局以本管线完整产物实测评分选优（对标 inj 侧"布局×
        # 管线联合选择"的自主实现——评分器是自研管线而非外部编译器）
        self.full_eval = full_eval
        self.full_eval_max_n = full_eval_max_n
        self.full_eval_topk = full_eval_topk
        self.fold_keep_on = fold_keep_on
        # 线性分解开关（2026-11-12 补充四十预注册：qv32 1Q 差距的
        # 分解架构重排）。判据已全面达标（补充四十三裁决）：qv32
        # 1Q 15,260 ≤16,500、depth 3,281 ≤3,600、logf −30.22 优于
        # qiskit O3（−30.5）、hwb12 墙钟 755s→160s——默认开启；
        # False 保留为 fixpoint 消融参照
        self.linear_decompose = linear_decompose
        self._full_result = None
        self._rng = np.random.default_rng(seed)
        self._dist_eff_cache = {}

    def _dist_eff(self, noise_lambda):
        if noise_lambda not in self._dist_eff_cache:
            self._dist_eff_cache[noise_lambda] = \
                self.spec.distance_matrix(noise_lambda)
        return self._dist_eff_cache[noise_lambda]

    # ------------------------------------------------------------ 布局竞技
    def _score_candidates(self, circ, graph, candidates):
        """每个候选用自研路由器实测：返回 (swaps, depth, fidelity, pi) 列表。

        评分路由上限：>50 比特 20000 交换（与全分层重测完全同口径——
        注意：触及上限的候选**可以**胜出，因为后处理栈能把置换内容吸收
        到 0，截断值不惩罚、只截断——这是该管线的实测特性，勿改上限）。
        胜者最终路由用全预算（永不截断，任何规模必出映射方案）。

        2026-11-13 RL 覆盖巨型线路（用户要求不降级到贪心）：
        门数 > 2 万时评分种子 16→2（胜者交付仍全种子重路由——评分
        噪声由交付补偿）+ 候选池截断。hwb12 级 51 万门评分从
        ~23 分钟降至 ~3 分钟。
        """
        huge = len(circ.ops) > 20000
        cap = 20000 if graph.n > 50 else 100000
        if huge:
            cap = 10_000_000  # 巨型线路评分不截断（截断 raise 后
                              # 触顶即淘汰该候选——见下）
            candidates = candidates[:4]
            seeds = min(2, self.routing_seeds)
        else:
            seeds = self.routing_seeds
        scored = []
        for pi in candidates:
            layout = {i: int(pi[i]) for i in range(graph.n)}
            best_entry = None
            for rseed in (self.seed + k for k in range(seeds)):
                try:
                    routed, swaps, _ = sabre_route(circ, self.spec, layout,
                                                   seed=rseed,
                                                   lam_ms=self.lam_ms,
                                                   max_swaps=cap)
                except RuntimeError:
                    # 触顶（巨型线路路由预算不足）：该候选淘汰
                    continue
                m = pure_metrics(routed, swaps, self.spec.calib,
                                 predictor=self.fidelity_predictor)
                entry = (m["swap_count"], m["depth"],
                         m["log_est_fidelity"], pi)
                # 种子取优按选择规则（审查 P2-8：此前恒按 depth，与
                # 文档"swap 最少"矛盾——fidelity 规则按 log 保真度）
                if best_entry is None or self._seed_better(entry, best_entry):
                    best_entry = entry
            if best_entry is not None:
                scored.append(best_entry)
        return scored

    def _v5_not_worse(self, m_v5, m_v4) -> bool:
        """非劣化开关：v5 产物按决胜规则不劣于 v4 才采用（2026-11-09）。

        fidelity 规则：保真度不降；swap 规则：swap 更少或持平且保真度
        不降；depth 规则：深度更浅或持平且保真度不降。
        """
        # 2026-11-12 审查 P0-3：改用对数保真度（乘积在大型线路下溢，
        # 原 1e-12 绝对容差使守卫对 37.5% 样本失去判别力）
        l5 = m_v5.get("log_est_fidelity", float("-inf"))
        l4 = m_v4.get("log_est_fidelity", float("-inf"))
        if self.selection_rule == "swap":
            return (m_v5["swap_count"] < m_v4["swap_count"]
                    or (m_v5["swap_count"] == m_v4["swap_count"]
                        and l5 >= l4 - 1e-9))
        if self.selection_rule == "depth":
            return (m_v5["depth"] < m_v4["depth"]
                    or (m_v5["depth"] <= m_v4["depth"]
                        and l5 >= l4 - 1e-9))
        return l5 >= l4 - 1e-9

    def _seed_better(self, entry, best) -> bool:
        """多种子取优（审查 P2-8）：按选择规则比较。
        注意 entry[2] 已统一为对数保真度（越大越好）。"""
        if self.selection_rule == "depth":
            return entry[1] < best[1]
        if self.selection_rule == "fidelity":
            return entry[2] > best[2] + 1e-9
        return entry[0] < best[0] or (entry[0] == best[0]
                                      and entry[2] > best[2] + 1e-9)

    def _select(self, scored):
        """三规则决胜（与主 Mapper 同逻辑；s[2] = 对数保真度）。"""
        if self.selection_rule == "swap":
            best = min(s[0] for s in scored)
            near = [s for s in scored if s[0] <= best + 2]
            return max(near, key=lambda s: s[2])[3], best
        if self.selection_rule == "fidelity":
            best = max(s[2] for s in scored)
            near = [s for s in scored if s[2] >= best - 0.02]
            return min(near, key=lambda s: (s[0], -s[2]))[3], min(s[0] for s in near)
        best = min(s[1] for s in scored)
        tol = max(3, 0.05 * best)
        near = [s for s in scored if s[1] <= best + tol]
        return min(near, key=lambda s: (s[0], s[1]))[3], min(s[0] for s in near)

    def _build_pool(self, graph, circ, noise_lambda):
        """候选池：RL 多起点 + LS（λ 噪声/λ0 拓扑两批）+ 启发式 + 平凡。"""
        cfg = self.cfg
        dist_eff = self._dist_eff(noise_lambda)
        pool = []

        disabled = self.spec.disabled_qubits
        if self.policy is not None:
            starts = multistart_decode(self.policy, graph, self.spec,
                                       cfg.decode, noise_lambda=noise_lambda,
                                       dev=self.dev, rng=self._rng)
            for s in starts[:cfg.postprocess.starts]:
                if cfg.postprocess.enabled:
                    pi, _ = improve_layout(graph, dist_eff, s,
                                           cfg.postprocess, rng=self._rng,
                                           disabled_qubits=disabled)
                else:
                    pi = s
                pool.append(pi)
            # λ=0 拓扑候选（链式线路的噪声打散防护）
            if noise_lambda != 0.0:
                starts0 = multistart_decode(self.policy, graph, self.spec,
                                            cfg.decode, noise_lambda=0.0,
                                            dev=self.dev, rng=self._rng)
                for s0 in starts0[:cfg.postprocess.starts]:
                    if cfg.postprocess.enabled:
                        pi0, _ = improve_layout(graph, self.spec.dist, s0,
                                                cfg.postprocess, rng=self._rng,
                                                disabled_qubits=disabled)
                    else:
                        pi0 = s0
                    pool.append(pi0)
        if graph.n > 50:  # 大线路：候选池缩减（评分成本控制）
            n_noise = min(5, len(pool))
            noise_part = pool[:n_noise]
            lam0_part = [pi for pi in pool[10:13]] \
                if noise_lambda != 0.0 and len(pool) > 10 else []
            pool = noise_part + lam0_part
        if not pool:
            pi_h = heuristic_layout(graph, self.spec, self._rng)
            pi_h, _ = improve_layout(graph, self.spec.dist, pi_h,
                                     cfg.postprocess, rng=self._rng,
                                     disabled_qubits=self.spec.disabled_qubits)
            pool.append(pi_h)
        # 平凡布局候选：仅使用非禁用物理比特（2026-11-09 缺口 1 修复）
        usable = self.spec.usable_positions
        pool.append(usable[np.arange(graph.n) % len(usable)])
        return pool

    def compute_layout(self, graph, circ, noise_lambda):
        """RL 多起点 + LS + 候选池（λ0/启发式/平凡）→ 路由竞技选优。"""
        pool = self._build_pool(graph, circ, noise_lambda)
        scored = self._score_candidates(circ, graph, pool)
        best_pi, best_swaps = self._select(scored)
        return best_pi, best_swaps

    # -------------------------------------------------- 管线级联合选择
    def _v5_pipeline(self, routed, final_layout, circ, noise_lambda,
                     with_learned: bool = True, with_resynth: bool = True,
                     relabel_budget: float | None = 25.0,
                     full_resynth: bool = False,
                     pre_folded=None, layout: dict | None = None):
        """v5 后处理栈（含非劣化开关）。返回 (routed, final_layout, final_qc)。

        pre_folded=(circ, fin)：post_route 前 fold_keep 产物（审查代理
        修复版），作为候选 2b 参与评分——坐标系干净（无 push_swaps
        推挤污染）。

        layout：初始布局（输入嵌入约定）——重标记（VF2/C1/L3）置换
        线路标签时随 final_layout 同步更新，保证交付结果的
        layout/final_layout 坐标自洽（2026-11-12 审查修复：此前
        layout 陈旧，按 layout 嵌入输入即语义错位）。原地更新。

        with_learned=False 跳过 L3、with_resynth=False 跳过 A2 块重合成、
        relabel_budget=None 跳过全局重标记（候选评分用：耗时大头在
        胜者交付时全量跑；预算单位秒）。full_resynth=True（v6 管线）
        时 A2 用全量版（逐块保真度守卫 + 无收益阈值，对标 qiskit 2.x
        TwoQubitPeephole 的实际机制）。
        """
        from qtrail.pure.post_v2 import (
            collect_resynthesize_2q_blocks,
            collect_resynthesize_2q_blocks_full, decompose_to_platform_v2,
            fold_and_relink, fold_keep_coords, fold_only, learned_relabel,
            noise_aware_relabel)
        routed_v4 = routed.copy()
        fin_v4 = dict(final_layout)
        lay_v4 = dict(layout) if layout is not None else None
        final_qc_v4 = decompose_to_platform(routed_v4)

        # ---- 三候选后处理路径，代理评分按决胜规则选优（2026-11-12：
        # 重标记/折叠结果劣于裸路由时必须回退——gf2^7/dt_100 实测教训）。
        # 2026-11-13 第 4 项：候选状态类型化——帧标记进构造契约，见
        # candidates.CandidateResult（content 帧强制 input_layout ==
        # final_layout，physical 帧允许初始布局 ≠ 最终布局）。
        cands: list[CandidateResult] = []
        # 候选 1：全局重标记路径（fold_only 坐标折叠 → 固定嵌入置换
        # → 全量链化）。门数上限防护（2026-11-12）：hwb12+ 类巨型
        # 线路链化会爆炸（hwb10 已 9 万门），交给老折叠路径。
        if relabel_budget is not None and len(routed.ops) <= 100_000:
            from qtrail.pure.relabel_vf2 import global_relabel
            from qtrail.pure.post_v2 import chain_long_gates
            folded_only, arr_only = fold_only(routed)
            res = global_relabel(folded_only, self.spec,
                                 time_budget=relabel_budget)
            if res is not None:
                relabeled, mapping, _exact = res
                ra, _ = chain_long_gates(relabeled, self.spec)
                fla = {}
                for k, v in final_layout.items():
                    fla[k] = int(mapping.get(
                        int(arr_only[int(v)]), int(arr_only[int(v)])))
                # fold 类产物输入在最终坐标（swap 吸收进重标记，
                # 产物 = 最终坐标视图）——输入约定 = 候选 final_layout
                cands.append(CandidateResult(
                    ra, "content", fla, dict(fla), tag="fold_only_relabel"))
        # 候选 2：老折叠路径（逐 swap 判定 + 链化）
        folded, fold_arr = fold_and_relink(routed, self.spec)
        if fold_arr is not None:
            flb = dict(final_layout)
            for k, v in flb.items():
                flb[k] = int(fold_arr[int(v)])
            cands.append(CandidateResult(
                folded, "content", flb, dict(flb), tag="fold_relink"))
        # 候选 2b（2026-11-13）：post_route 前 fold_keep 产物
        # （审查代理 P0 修复版——最终坐标重标记，非空验 fid=1.0）。
        # 输入约定 = 最终坐标本身（fold_keep 把每门重标记到
        # fl[content[p]]，逻辑 i 的首个门落在 final_layout[i]）
        if pre_folded is not None:
            pk_c, pk_fin = pre_folded[0], pre_folded[1]
            cands.append(CandidateResult(
                pk_c, "content", pk_fin, dict(pk_fin), tag="fold_keep"))
        # 候选 3：裸路由（v4 行为，永远合法；物理帧）
        cands.append(CandidateResult(
            routed.copy(), "physical", dict(final_layout),
            dict(layout) if layout is not None else None, tag="bare"))
        # 代理评分选优（分解后指标 + 决胜规则）
        scored = []
        for cand in cands:
            q = decompose_to_platform(cand.circuit)
            m = pure_metrics(q, cand.circuit.count("swap"),
                             self.spec.calib,
                             predictor=self.fidelity_predictor)
            scored.append((m["swap_count"], m["depth"],
                           m["log_est_fidelity"], cand))
        chosen = self._select(scored)[0]
        routed = chosen.circuit
        final_layout = dict(chosen.final_layout)
        if layout is not None and chosen.input_layout is not None:
            layout.clear()
            layout.update(chosen.input_layout)
        noise_aware_relabel(routed, self.spec, final_layout, layout=layout)
        if with_learned and self.policy is not None:
            learned_relabel(self.policy, self.spec, routed, final_layout,
                            self.cfg, dev=self.dev, seed=self.seed,
                            noise_lambda=noise_lambda, layout=layout)
        if with_resynth:
            # 注：解碎片 _preslide_1q 已移除（2026-11-13 达标消融：
            # no-slide 三线路保真度全面更好——0.821/0.713/0.775 vs
            # 0.810/0.710/0.766；身份路由的门数收益不转移到达标路由
            # 的保真度，"门数 ≠ 指标"）。
            if full_resynth:
                routed = collect_resynthesize_2q_blocks_full(routed, self.spec)
            else:
                routed = collect_resynthesize_2q_blocks(routed, self.spec)
        # decompose 输入 dump 钩子（A/B 机制实验用：QTRAIL_DUMP_
        # DECOMPOSE_INPUT 环境变量指向输出 pickle；候选评分循环中
        # 每候选各写一次、胜者补跑最后写——末次即胜者输入）
        _dump = os.environ.get("QTRAIL_DUMP_DECOMPOSE_INPUT")
        if _dump:
            import pickle
            with open(_dump, "wb") as _f:
                pickle.dump(routed, _f)
        final_qc = decompose_to_platform_v2(
            routed, spec=self.spec, linear=self.linear_decompose)
        m_v4 = pure_metrics(final_qc_v4, routed_v4.count("swap"),
                            self.spec.calib,
                            predictor=self.fidelity_predictor)
        m_v5 = pure_metrics(final_qc, routed.count("swap"),
                            self.spec.calib,
                            predictor=self.fidelity_predictor)
        if not self._v5_not_worse(m_v5, m_v4):
            routed, final_layout, final_qc = routed_v4, fin_v4, final_qc_v4
            if layout is not None and lay_v4 is not None:
                layout.clear()
                layout.update(lay_v4)
        return routed, final_layout, final_qc

    def _full_score_candidates(self, circ, graph, candidates, noise_lambda):
        """每候选跑完整 v5 管线（跳过 L3）实测：返回
        [(swaps, depth, fid, pi, (routed_pre, final_layout_pre, removed))]。

        管线级联合选择的核心：评分目标 = 本管线最终产物（对标 inj 侧
        布局×管线联合选择的自主实现——inj 评分器是 qiskit O3，本实现
        评分器是自研管线，不依赖外部编译器）。胜者由调用方补跑 L3。
        """
        scored = []
        for pi in candidates:
            layout = {i: int(pi[i]) for i in range(graph.n)}
            # 巨型线路大预算（2026-11-13 RL 覆盖——评分路由不触顶）
            max_swaps = (10_000_000 if len(circ.ops) > 20000
                         else 100_000)
            routed, swaps, final_layout = sabre_route(
                circ, self.spec, layout, seed=self.seed, lam_ms=self.lam_ms,
                trials=max(1, self.routing_seeds),
                mix_rollout=(circ.n > 10), max_swaps=max_swaps)
            removed = 0
            if self.use_post:
                routed, removed = post_route(routed, final_layout,
                                             self.spec.adj)
            routed_pre = routed.copy()
            fin_pre = dict(final_layout)
            routed, final_layout, final_qc = self._v5_pipeline(
                routed, final_layout, circ, noise_lambda, with_learned=False,
                with_resynth=False, relabel_budget=3.0)
            m = pure_metrics(final_qc, routed.count("swap"),
                             self.spec.calib,
                             predictor=self.fidelity_predictor)
            scored.append((m["swap_count"], m["depth"],
                           m["log_est_fidelity"], pi,
                           (routed_pre, fin_pre, removed)))
        return scored

    def _compute_layout_full(self, graph, circ, noise_lambda):
        """布局×管线联合选择：候选池廉价路由预筛排序 → top-K 全管线
        实测 → 规则选优。胜者补跑 L3 后产物缓存供 map_circuit 交付。"""
        pool = self._build_pool(graph, circ, noise_lambda)
        scored = self._score_candidates(circ, graph, pool)
        if self.selection_rule == "fidelity":
            ranked = sorted(scored, key=lambda s: -s[2])
        elif self.selection_rule == "depth":
            ranked = sorted(scored, key=lambda s: s[1])
        else:
            ranked = sorted(scored, key=lambda s: (s[0], -s[2]))
        topk = [s[3] for s in ranked[:self.full_eval_topk]]
        full = self._full_score_candidates(circ, graph, topk, noise_lambda)
        best_pi, best_swaps = self._select(full)
        for entry in full:
            if np.array_equal(entry[3], best_pi):
                routed_pre, fin_pre, removed = entry[4]
                lay_dict = {i: int(best_pi[i]) for i in range(graph.n)}
                routed, final_layout, final_qc = self._v5_pipeline(
                    routed_pre, dict(fin_pre), circ, noise_lambda,
                    with_learned=True,
                    full_resynth=(self.pipeline == "v6"),
                    layout=lay_dict)
                self._full_result = (routed, final_layout, final_qc,
                                     removed, lay_dict)
                break
        return best_pi, best_swaps

    # -------------------------------------------------------------- 主入口
    def map_circuit(self, circ: Circuit, circuit_id: str = "circuit",
                    noise_lambda: float | None = None) -> dict:
        t0 = time.time()
        if noise_lambda is None:
            noise_lambda = (self.cfg.device.noise.lambda_n
                            if self.cfg.reward.mode in ("noise", "combined")
                            else 0.0)
        if circ.n > self.spec.n:
            raise ValueError(f"circuit uses {circ.n} qubits but device "
                             f"{self.spec.name} has {self.spec.n}")
        # 独立审计 P2-2：禁用比特不可用；只用 spec.n 检查会让超容量
        # 线路在启发式/RL 解码里回绕出重复物理位（静默语义错误）。
        usable = self.spec.n - len(set(self.spec.disabled_qubits))
        if circ.n > usable:
            raise ValueError(
                f"circuit uses {circ.n} qubits but device {self.spec.name} "
                f"has only {usable} usable qubits "
                f"({len(set(self.spec.disabled_qubits))} disabled)")
        graph = build_program_graph(circ.n, _ops_from_circuit(circ),
                                    circuit_id=circuit_id,
                                    temporal_alpha=self.cfg.graph.temporal_alpha)
        self._full_result = None
        # RL 布局覆盖巨型线路（2026-11-13，用户要求不降级）：
        # 评分成本控制后（_score_candidates 巨型线路种子 16→2+
        # 候选池截断，hwb12 级 ~3 分钟）不再需要贪心降级——
        # 贪心降级是 hwb 类 10 万 swap 与 medium 大败的根因之一。
        # 仅 >100 万门保留降级兜底（hwb12 51.4 万门走 RL）。
        if len(circ.ops) > 1_000_000:
            from qtrail.pure.layout import greedy_graph_match_layout
            pi = greedy_graph_match_layout(circ, self.spec)
        elif self.pipeline in ("v5", "v6") and self.full_eval \
                and circ.n <= self.full_eval_max_n:
            # 管线级联合选择（2026-11-11）：候选布局以本管线完整产物
            # 实测评分选优，胜者产物缓存复用（不重复路由）
            pi, _ = self._compute_layout_full(graph, circ, noise_lambda)
        else:
            pi, _ = self.compute_layout(graph, circ, noise_lambda)
        layout = {i: int(pi[i]) for i in range(graph.n)}

        if self._full_result is not None:
            if len(self._full_result) == 5:
                routed, final_layout, final_qc, removed, layout = \
                    self._full_result
            else:
                routed, final_layout, final_qc, removed = self._full_result
            swaps = routed.count("swap")
        else:
            # 2026-11-13 审查代理 P1 修复：巨型线路（hwb12 类 10 万+
            # swap）给足路由预算——默认 10 万触顶 raise（不再静默
            # 截断半线路）
            max_swaps = (10_000_000 if len(circ.ops) > 20000
                         else 100_000)
            routed, swaps, final_layout = sabre_route(
                circ, self.spec, layout, seed=self.seed, lam_ms=self.lam_ms,
                trials=max(1, self.routing_seeds),
                mix_rollout=(circ.n > 10), max_swaps=max_swaps)
            removed = 0
            # 候选 2b 前置（post_route 前坐标系干净——审查代理要求）
            # 相邻性守卫（2026-11-13）：最终布局下门不一定相邻
            # （swap 穿越后"执行时相邻"≠"最终布局相邻"，hwb12 实测
            # 15.7 万非相邻）——非相邻则退出候选（保守路径兜底）
            from qtrail.pure.post_v2 import fold_keep_coords
            pre_folded = None
            if self.fold_keep_on:
                fk_c, fk_fin = fold_keep_coords(routed, final_layout,
                                                layout)
                if all(i.nq != 2 or self.spec.adj[int(i.qubits[0]),
                                                  int(i.qubits[1])]
                       for i in fk_c.ops):
                    pre_folded = (fk_c, fk_fin)
            if self.use_post:
                # 硬件连通性感知后处理（2026-11-09）：产物 2Q 门全部落在
                # 耦合图上，保证平台可执行
                routed, removed = post_route(routed, final_layout,
                                             self.spec.adj)
            if self.pipeline in ("v5", "v6"):
                # v5：置换折叠+LNN 重链化（A1）+ 噪声感知后布局重标记（C1）+
                # 学习式重标记（L3）+ 2Q 块重合成（A2）+ CX 通勤对消（A3）+
                # 1Q 游程优化（B）+ 非劣化开关（劣于 v4 则回退）；
                # v6 = v5 + A2 全量版（full_resynth，逐块保真度守卫）
                routed, final_layout, final_qc = self._v5_pipeline(
                    routed, final_layout, circ, noise_lambda,
                    full_resynth=(self.pipeline == "v6"),
                    pre_folded=pre_folded, layout=layout)
            else:
                final_qc = decompose_to_platform(routed)
        metrics = pure_metrics(final_qc, routed.count("swap"),
                               self.spec.calib,
                               predictor=self.fidelity_predictor)
        metrics["static_cost"] = float(graph.cost(pi, self.spec.dist,
                                                  dist_mult=2.0))
        metrics["post_absorbed"] = removed
        metrics["wall_s"] = round(time.time() - t0, 3)
        return {
            "layout": layout,
            "final_layout": final_layout,
            "swap_count": routed.count("swap"),
            "routed_circuit": final_qc,
            "metrics": metrics,
            "method": "pure_rl_multistart" if self.policy is not None
                      else "pure_heuristic",
        }
