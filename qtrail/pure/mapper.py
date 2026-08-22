"""qiskit-free 映射管线：纯 Circuit 输入 → RL 布局 → 自研 SABRE 路由 →
后处理栈 → CZ 基输出（qtrail/pure 包顶层入口）。

复用 qiskit-free 组件：problems（程序图）、search（多起点/LS）、
models（RL 策略，纯 PyTorch）、devices（DeviceSpec）；竞赛评分路由 =
自研 sabre_route（廉价代理 + 最终执行一体）。
"""
from __future__ import annotations

import time

import numpy as np

from qtrail.pure.circuit import Circuit
from qtrail.pure.metrics import compute_metrics as pure_metrics
from qtrail.pure.post import decompose_to_platform, post_route
from qtrail.pure.router import sabre_route
from qtrail.pure.layout import (heuristic_layout, noise_greedy_layout,
                                 strong_noise_layout, usable_positions)

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
                 routing_seeds=1):
        from qtrail.config import Config
        self.spec = spec
        self.policy = policy
        self.cfg = cfg or Config()
        self.dev = dev
        self.seed = seed
        if selection_rule not in ("swap", "fidelity", "depth"):
            raise ValueError(f"unknown selection rule: {selection_rule}")
        self.selection_rule = selection_rule
        self.lam_ms = lam_ms
        self.use_post = use_post
        self.routing_seeds = routing_seeds
        self._rng = np.random.default_rng(seed)
        self._dist_eff_cache = {}
        self._disabled = list(getattr(spec, "disabled_qubits", []) or [])

    def _dist_eff(self, noise_lambda):
        if noise_lambda not in self._dist_eff_cache:
            self._dist_eff_cache[noise_lambda] = \
                self.spec.distance_matrix(noise_lambda)
        return self._dist_eff_cache[noise_lambda]

    # ------------------------------------------------------------ 布局竞技
    def _score_candidates(self, circ, graph, candidates):
        """每个候选用自研路由器实测 + 单遍置换折叠后评分：
        返回 (residual_swaps, depth, fidelity, pi) 列表。

        评分路由上限统一 100000（2026-08-22 起不再分档——增量路由 +
        Numba 内核后单交换 ~50μs，全预算评分成本可接受；此前 >50 比特
        的 20000 截断在折叠口径下会系统性低估需要 >20000 交换的候选
        （噪声启发式候选在 qft 上需 ~27k 交换，被截断导致错失胜者）。
        胜者最终路由全预算（永不截断，任何规模必出映射方案）。
        2026-08-22：候选评分改为**硬件可执行后处理口径**（连通性感知
        推挤 + 尾块吸收，与最终输出同口径，候选竞技看到的即交付线路
        的真实指标）。
        """
        cap = 100000
        scored = []
        for pi in candidates:
            layout = {i: int(pi[i]) for i in range(graph.n)}
            best_entry = None
            for rseed in (self.seed + k for k in range(self.routing_seeds)):
                routed, swaps, fl = sabre_route(circ, self.spec, layout,
                                                seed=rseed,
                                                lam_ms=self.lam_ms,
                                                max_swaps=cap)
                post_route(routed, dict(fl), spec=self.spec)
                m = pure_metrics(routed, routed.count("swap"),
                                 self.spec.calib)
                entry = (m["swap_count"], m["depth"],
                         m["est_fidelity"], pi)
                if best_entry is None or entry[1] < best_entry[1]:
                    best_entry = entry
            scored.append(best_entry)
        return scored

    def _select(self, scored):
        """三规则决胜（与主 Mapper 同逻辑）。"""
        if self.selection_rule == "swap":
            best = min(s[0] for s in scored)
            near = [s for s in scored if s[0] <= best + 2]
            return max(near, key=lambda s: s[2])[3], best
        if self.selection_rule == "fidelity":
            best = max(s[2] for s in scored)
            near = [s for s in scored if s[2] >= best * 0.98]
            return min(near, key=lambda s: (s[0], -s[2]))[3], min(s[0] for s in near)
        best = min(s[1] for s in scored)
        tol = max(3, 0.05 * best)
        near = [s for s in scored if s[1] <= best + tol]
        return min(near, key=lambda s: (s[0], s[1]))[3], min(s[0] for s in near)

    def compute_layout(self, graph, circ, noise_lambda):
        """RL 多起点 + LS + 候选池（λ0/启发式/平凡）→ 路由竞技选优。"""
        cfg = self.cfg
        dist_eff = self._dist_eff(noise_lambda)
        pool = []

        if self.policy is not None:
            starts = multistart_decode(self.policy, graph, self.spec,
                                       cfg.decode, noise_lambda=noise_lambda,
                                       dev=self.dev, rng=self._rng)
            for s in starts[:cfg.postprocess.starts]:
                if cfg.postprocess.enabled:
                    pi, _ = improve_layout(graph, dist_eff, s,
                                           cfg.postprocess, rng=self._rng,
                                           disabled=self._disabled)
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
                                           disabled=self._disabled)
                    else:
                        pi0 = s0
                    pool.append(pi0)
        # 噪声启发式候选（自研）：噪声贪心 + 强噪声纯噪声序——密集
        # 全连通线路（QFT/QV）的缺陷热区规避补充（RL 静态代价在设备
        # 饱和时被拓扑项主导，无法充分压低有效误差）
        pi_n = pi_sn = None
        if noise_lambda != 0.0:
            pi_n = noise_greedy_layout(graph, self.spec, dist_eff, self._rng)
            if cfg.postprocess.enabled:
                pi_n, _ = improve_layout(graph, dist_eff, pi_n,
                                         cfg.postprocess, rng=self._rng,
                                           disabled=self._disabled)
            pool.append(pi_n)
            pi_sn = strong_noise_layout(graph, self.spec, self._rng)
            if cfg.postprocess.enabled:
                pi_sn, _ = improve_layout(graph, dist_eff, pi_sn,
                                          cfg.postprocess, rng=self._rng,
                                           disabled=self._disabled)
            pool.append(pi_sn)
        if graph.n > 50:  # 大线路：候选池缩减（评分成本控制）
            n_noise = min(5, len(pool))
            noise_part = pool[:n_noise]
            lam0_part = [pi for pi in pool[10:13]] \
                if noise_lambda != 0.0 and len(pool) > 10 else []
            extra = [pi for pi in (pi_n, pi_sn) if pi is not None]
            pool = noise_part + lam0_part + extra
        if not pool:
            pi_h = heuristic_layout(graph, self.spec, self._rng)
            pi_h, _ = improve_layout(graph, self.spec.dist, pi_h,
                                     cfg.postprocess, rng=self._rng,
                                           disabled=self._disabled)
            pool.append(pi_h)
        usable = usable_positions(self.spec)
        pool.append(usable[np.arange(graph.n, dtype=np.int64) % len(usable)])

        scored = self._score_candidates(circ, graph, pool)
        best_pi, best_swaps = self._select(scored)
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
        graph = build_program_graph(circ.n, _ops_from_circuit(circ),
                                    circuit_id=circuit_id,
                                    temporal_alpha=self.cfg.graph.temporal_alpha)
        pi, _ = self.compute_layout(graph, circ, noise_lambda)
        layout = {i: int(pi[i]) for i in range(graph.n)}

        routed, swaps, final_layout = sabre_route(
            circ, self.spec, layout, seed=self.seed, lam_ms=self.lam_ms)
        removed = 0
        if self.use_post:
            routed, removed = post_route(routed, final_layout,
                                         spec=self.spec)
        final_qc = decompose_to_platform(routed)
        metrics = pure_metrics(final_qc, routed.count("swap"),
                               self.spec.calib)
        metrics["routed_swaps"] = swaps            # 折叠前路由原始交换数
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
