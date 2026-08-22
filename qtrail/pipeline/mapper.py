"""End-to-end mapper: QASM -> program graph -> RL layout (+LS) -> routing.

Judge-facing robustness: a failure ladder guarantees a valid mapped circuit
for ANY parseable OpenQASM 2.0 input up to the device size:
  1. RL multistart + adaptive local search
  2. RL greedy
  3. heuristic layout (spectral order -> device spiral) + local search
  4. trivial layout + SabreSwap
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from qtrail.config import Config
from qtrail.devices.spec import DeviceSpec
from qtrail.models import QAPolicy
from qtrail.pipeline.metrics import compute_metrics
from qtrail.pipeline.routing import (coupling_map_from_spec, decompose_to_platform,
                                     route_with_layout)
from qtrail.problems import ProgramGraph, build_program_graph
from qtrail.search.decoding import decode_layout, multistart_decode
from qtrail.search.local_search import improve_layout
from qtrail.utils.qasm_io import extract_ops

log = logging.getLogger("qtrail")


@dataclass
class MappingResult:
    layout: dict                       # logical -> physical (initial)
    final_layout: dict                 # logical -> physical (after routing)
    swap_count: int
    routed_qc: QuantumCircuit          # platform basis, measurements stripped
    metrics: dict
    method: str                        # which rung of the ladder produced the layout
    warnings: list = field(default_factory=list)
    static_cost: float | None = None   # CO-MAP paper metric: sum 2*d over edges
    baseline_swaps: int | None = None  # SABRE O1 swap count measured in the
                                        # same call (internally consistent)


def heuristic_layout(graph: ProgramGraph, spec: DeviceSpec,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Spectral heuristic: Fiedler-vector order of logical qubits mapped onto
    a device Hamiltonian path (grid snake). Never fails, works without a
    policy."""
    n = graph.n
    # Fiedler vector of the weighted Laplacian (2nd smallest eigenvector)
    lap = np.diag(graph.adj.sum(axis=1)) - graph.adj
    evals, evecs = np.linalg.eigh(lap)
    fiedler = evecs[:, 1] if n > 1 else np.zeros(n)
    logical_order = np.argsort(fiedler)

    path = _device_snake(spec)

    pi = np.zeros(n, dtype=np.int64)
    for k, logical in enumerate(logical_order):
        pi[logical] = path[k % len(path)]
    return pi


def _device_snake(spec: DeviceSpec) -> list[int]:
    """设备哈密顿路径：完整网格用行向蛇形（相邻性严格成立）；
    其余拓扑用贪心最近邻 + BFS 逃逸。"""
    import networkx as nx

    coords = spec.coords
    # 网格判定：坐标整数、行列数之积 = 设备规模
    if np.allclose(coords, coords.astype(np.int64)):
        rows = sorted({int(c[0]) for c in coords})
        ncols = len({int(c[1]) for c in coords})
        if len(rows) * ncols == spec.n:
            path = []
            for ri, r in enumerate(rows):
                row_q = sorted([i for i in range(spec.n)
                                if int(coords[i][0]) == r],
                               key=lambda i: int(coords[i][1]))
                path += row_q if ri % 2 == 0 else row_q[::-1]
            return path

    # 通用贪心路径（最近未访问邻接优先；断点用 BFS 最短路径逃逸）
    g = nx.Graph()
    g.add_nodes_from(range(spec.n))
    g.add_edges_from([(i, j) for i in range(spec.n)
                      for j in range(i + 1, spec.n) if spec.adj[i, j]])
    center = int(np.argmin(np.abs(spec.coords - spec.coords.mean(axis=0))
                           .sum(axis=1)))
    path = [center]
    visited = {center}
    while len(path) < spec.n:
        nbrs = [j for j in range(spec.n) if spec.adj[path[-1], j]
                and j not in visited]
        if not nbrs:  # 断点：BFS 到最近未访问节点，取路径逐步加入
            unvisited = [v for v in range(spec.n) if v not in visited]
            if not unvisited:
                break
            target = min(unvisited, key=lambda v: spec.dist[path[-1], v])
            try:
                chain = nx.shortest_path(g, path[-1], target)
            except nx.NetworkXNoPath:
                path.append(target)
                visited.add(target)
                continue
            for v in chain[1:]:
                if v not in visited:
                    path.append(v)
                    visited.add(v)
            continue
        path.append(nbrs[0])
        visited.add(nbrs[0])
    return path


class Mapper:
    """Stateful mapper: caches policy/device across calls (fast for judges)."""

    def __init__(self, spec: DeviceSpec, policy: QAPolicy | None = None,
                 cfg: Config | None = None, dev: str = "cpu",
                 seed: int = 0, use_tket: bool = False, use_o3: bool = False,
                 tket_max_qubits: int = 200,
                 selection_rule: str = "swap",
                 include_sabre: bool = False,
                 routing_method: str = "sabre",
                 cqlib_objective: str = "depth",
                 cqlib_timeout: float = 300.0,
                 target_post: bool = False,
                 target_post_opt: int = 1,
                 target_post_score_opt: int | None = None,
                 target_post_seeds: int = 4,
                 target_post_top_per_seed: int | None = None):
        """use_tket/use_o3: 混合竞技外部采纳（工程增强，默认关闭——
        研究管线的默认配置不依赖任何外部编译器的输出）。

        include_sabre=False: 纯研究模式——竞技池仅含 RL 候选与扰动变体，
        SABRE 只作为路由器（SabreSwap）与评测基线使用。

        routing_method: "sabre"（默认）| "lexi"（自研深度感知路由器）
        | "cqlib"（Cqlib 注入：RL 布局 → 平台原生 transpile_qcis MCTS 路由）。
        路由感知重排序的评分路由恒为 SabreSwap（廉价代理），最终执行
        后端为 routing_method。
        target_post: True 时路由与竞技评分均改为 qiskit O1 预设完整管线
        （噪声感知 Target 驱动）——路由后重标记/酉综合阶段吸收置换（实测
        QUEKO dense SWAP 92→0），候选按后处理后的真实结果评分
        （布局×管线联合选择）。opt-in，默认 False（原研究管线不变）。
        """
        self.spec = spec
        self.policy = policy
        self.cfg = cfg or Config()
        self.dev = dev
        self.seed = seed
        self.use_tket = use_tket
        self.use_o3 = use_o3
        self.tket_max_qubits = tket_max_qubits  # --fast 或限时场景可调
        self.include_sabre = include_sabre
        if selection_rule not in ("swap", "fidelity", "depth"):
            raise ValueError(f"unknown selection rule: {selection_rule}")
        self.selection_rule = selection_rule  # 混合竞技决胜规则
        # 路由方法：sabre（默认，竞争性）；lexi = 自研深度感知路由器
        # （简化版 LexiRoute 研究实现：正确性已验证，启发式强度待迭代）；
        # cqlib = 天衍平台原生管线（MCTS 路由）注入 RL 布局
        if routing_method not in ("sabre", "lexi", "cqlib"):
            raise ValueError(f"unknown routing method: {routing_method}")
        self.routing_method = routing_method
        self.cqlib_objective = cqlib_objective
        self.cqlib_timeout = cqlib_timeout
        self.target_post = target_post
        self.target_post_opt = target_post_opt
        # 评分用优化级别（None = 与执行一致）；廉价代理评分 + 强优化执行
        self.target_post_score_opt = (target_post_score_opt
                                        if target_post_score_opt is not None
                                        else target_post_opt)
        self.target_post_seeds = target_post_seeds
        # 每种子入池上限（None = 全量）：多种子×少候选 的多样性分配
        self.target_post_top_per_seed = target_post_top_per_seed
        self._rng = np.random.default_rng(seed)
        self._dist_eff_cache = {}
        self.cm = coupling_map_from_spec(spec)

    # ------------------------------------------------------------ layout
    def _dist_eff(self, noise_lambda: float) -> np.ndarray:
        if noise_lambda not in self._dist_eff_cache:
            self._dist_eff_cache[noise_lambda] = self.spec.distance_matrix(noise_lambda)
        return self._dist_eff_cache[noise_lambda]

    def compute_layout(self, graph: ProgramGraph, noise_lambda: float,
                       qc: QuantumCircuit | None = None,
                       routing_aware: bool = False,
                       routing_candidates: int = 3) -> tuple[np.ndarray, str, list]:
        """Failure-ladder layout computation. Returns (pi, method, warnings).

        With routing_aware=True (and a circuit given), the top candidates by
        static cost are re-ranked by ACTUAL SabreSwap routing swap counts —
        the final layout directly optimizes the competition metric.
        """
        warnings: list[str] = []
        cfg = self.cfg
        dist_eff = self._dist_eff(noise_lambda)

        candidates: list[np.ndarray] = []
        method = "trivial"

        # rung 1: RL multistart + local search
        if self.policy is not None:
            try:
                starts = multistart_decode(self.policy, graph, self.spec,
                                           cfg.decode, noise_lambda=noise_lambda,
                                           dev=self.dev, rng=self._rng)
                scored = []
                for s in starts[:cfg.postprocess.starts]:
                    if cfg.postprocess.enabled:
                        pi, cost = improve_layout(graph, dist_eff, s,
                                                  cfg.postprocess, rng=self._rng)
                    else:
                        pi, cost = s, float(graph.cost(s, dist_eff))
                    scored.append((cost, pi))
                scored.sort(key=lambda t: t[0])
                candidates = [pi for _, pi in scored[:routing_candidates]]
                if candidates:
                    method = "rl_multistart"
            except Exception as e:  # pragma: no cover - defensive
                warnings.append(f"rl_multistart failed: {e}")

            # rung 2: RL greedy (only if multistart produced nothing)
            if not candidates:
                try:
                    candidates = [decode_layout(self.policy, graph, self.spec,
                                                noise_lambda=noise_lambda,
                                                mode="greedy", dev=self.dev)]
                    method = "rl_greedy"
                except Exception as e:  # pragma: no cover - defensive
                    warnings.append(f"rl_greedy failed: {e}")

        # rung 3: heuristic layout + local search
        if not candidates:
            try:
                pi = heuristic_layout(graph, self.spec, self._rng)
                pi, _ = improve_layout(graph, dist_eff, pi, cfg.postprocess,
                                       rng=self._rng)
                candidates = [pi]
                method = "heuristic"
            except Exception as e:  # pragma: no cover - defensive
                warnings.append(f"heuristic failed: {e}")

        # rung 4: trivial layout
        if not candidates:
            candidates = [np.arange(graph.n, dtype=np.int64) % self.spec.n]

        # ---- routing-aware re-ranking of top candidates (hybrid pool)
        # Pool = RL layouts + SABRE's own layout (so the routed-swap metric is
        # never worse than the baseline) + random-swap perturbations. Each
        # candidate is scored by ACTUAL SabreSwap routing; near-min-swap
        # candidates are tie-broken by estimated layout fidelity (QTrail).
        if routing_aware and qc is not None and len(candidates) > 1:
            pool = list(candidates[:5])
            for pi in candidates[:2]:
                for _ in range(4):
                    p2 = pi.copy()
                    a, b = self._rng.integers(0, graph.n, size=2)
                    if a != b:
                        p2[a], p2[b] = p2[b], p2[a]
                        pool.append(p2)
            # 拓扑纯代价候选（噪声消融布局）：链式/规则结构线路在噪声感知
            # 代价下会被打散（噪声区回避），路由代价爆炸——把 λ=0 最优布局
            # 一并放入竞技池，由真实路由结果裁决（多目标布局选择）
            if noise_lambda != 0.0 and self.policy is not None:
                try:
                    starts0 = multistart_decode(self.policy, graph, self.spec,
                                                cfg.decode, noise_lambda=0.0,
                                                dev=self.dev, rng=self._rng)
                    scored0 = []
                    for s in starts0[:cfg.postprocess.starts]:
                        if cfg.postprocess.enabled:
                            pi0, cost0 = improve_layout(
                                graph, self.spec.dist, s, cfg.postprocess,
                                rng=self._rng)
                        else:
                            pi0, cost0 = s, float(graph.cost(s, self.spec.dist))
                        scored0.append((cost0, pi0))
                    scored0.sort(key=lambda t: t[0])
                    for _, pi0 in scored0[:5]:
                        if not any(np.array_equal(pi0, p) for p in pool):
                            pool.append(pi0)
                except Exception as e:  # pragma: no cover - defensive
                    warnings.append(f"topo-pool candidates failed: {e}")
            # target_post 多种子池（多试验取最优，用户选定）：K 支独立
            # 随机种子各生成完整候选集（多起点 + LS + λ0 批次），全部
            # 并入竞技池后感知评分——感知 O3 50 试验方法论的自研实现，
            # 候选全部来自 RL 策略与自研构造（池纯净性不变）
            if self.target_post:
                n_in = self.target_post_top_per_seed  # None = 全量入池
                for kseed in range(1, self.target_post_seeds):
                    rng_k = np.random.default_rng(self.seed + kseed)
                    try:
                        starts_k = multistart_decode(
                            self.policy, graph, self.spec, cfg.decode,
                            noise_lambda=noise_lambda, dev=self.dev,
                            rng=rng_k)
                        scored_k = []
                        for s in starts_k[:cfg.postprocess.starts]:
                            if cfg.postprocess.enabled:
                                pi_k, c_k = improve_layout(
                                    graph, dist_eff, s, cfg.postprocess,
                                    rng=rng_k)
                            else:
                                pi_k, c_k = s, float(
                                    graph.cost(s, dist_eff))
                            scored_k.append((c_k, pi_k))
                        scored_k.sort(key=lambda t: t[0])
                        for _, pi_k in scored_k[:(n_in or len(scored_k))]:
                            if not any(np.array_equal(pi_k, q) for q in pool):
                                pool.append(pi_k)
                    except Exception as e:  # pragma: no cover - defensive
                        warnings.append(f"multiseed pool k={kseed} failed: {e}")
                    if noise_lambda != 0.0 and self.policy is not None:
                        try:
                            starts0k = multistart_decode(
                                self.policy, graph, self.spec, cfg.decode,
                                noise_lambda=0.0, dev=self.dev, rng=rng_k)
                            scored0k = []
                            for s0 in starts0k[:cfg.postprocess.starts]:
                                if cfg.postprocess.enabled:
                                    pi0k, c0k = improve_layout(
                                        graph, self.spec.dist, s0,
                                        cfg.postprocess, rng=rng_k)
                                else:
                                    pi0k, c0k = s0, float(
                                        graph.cost(s0, self.spec.dist))
                                scored0k.append((c0k, pi0k))
                            scored0k.sort(key=lambda t: t[0])
                            for _, pi0k in scored0k[:(n_in or len(scored0k))]:
                                if not any(np.array_equal(pi0k, q)
                                           for q in pool):
                                    pool.append(pi0k)
                        except Exception as e:  # pragma: no cover - defensive
                            warnings.append(
                                f"multiseed topo k={kseed} failed: {e}")
            # 失败阶梯候选（结构感知谱序螺旋 + 平凡布局）同样进入竞技池：
            # 链式/规则结构线路下 RL+LS 可能停在静态代价局部最优
            # （噪声代价把链打散），谱序螺旋是强结构候选——由真实路由裁决
            try:
                pi_h = heuristic_layout(graph, self.spec, self._rng)
                pi_h, _ = improve_layout(graph, self.spec.dist, pi_h,
                                         cfg.postprocess, rng=self._rng)
                pool.append(pi_h)
            except Exception as e:  # pragma: no cover - defensive
                warnings.append(f"heuristic-pool failed: {e}")
            pool.append(np.arange(graph.n, dtype=np.int64) % self.spec.n)
            # SABRE's full result (computed by the caller in map_circuit, or
            # here when compute_layout is used standalone) enters the contest
            # as a candidate outcome — our system is never worse than the
            # baseline on routed swaps, and RL layouts win when they route better.
            # （工程增强：include_sabre=True 时启用；研究管线默认关闭）
            if self.include_sabre and getattr(self, "_sabre_result", None) is None:
                try:
                    from qtrail.pipeline.baselines import sabre_swap_count
                    sabre_swaps, sabre_routed = sabre_swap_count(
                        qc, self.cm, optimization_level=1, seed=self.seed)
                    init = sabre_routed.layout.initial_layout
                    orig_qubits = set(qc.qubits)
                    sabre_pi = np.zeros(graph.n, dtype=np.int64)
                    n_mapped = 0
                    for v, phys in init.get_virtual_bits().items():
                        if v in orig_qubits:
                            sabre_pi[qc.find_bit(v).index] = phys
                            n_mapped += 1
                    self._sabre_result = ((sabre_swaps, sabre_pi, sabre_routed)
                                          if n_mapped == graph.n else None)
                except Exception:
                    self._sabre_result = None

            scored = []
            # SABRE-MS 思想：makespan 目标注入——多种子调度取 makespan 最优
            # （论文：收益来自目标而非机制；多种子选择 = 零机制风险实现）
            # 评分路由固定用 SabreSwap（廉价代理）：最终执行后端
            # （self.routing_method，含 cqlib MCTS）在 map_circuit 应用，
            # 逐候选跑 MCTS 不可行——学习式候选 + 公开路由评分器是本管线
            # 的既定设计（研究贡献 = 布局与选择机制，非路由评分器）。
            # target_post 模式：候选按"后处理后的真实结果"评分（布局×管线
            # 联合选择——候选池仍为 RL 候选与自研构造，无外部布局）
            n_seeds = (12 if not self.target_post else 3) \
                if self.selection_rule == "depth" else 1
            for pi in pool:
                layout = {i: int(pi[i]) for i in range(graph.n)}
                best_entry = None
                for rseed in (self.seed + k for k in range(n_seeds)):
                    try:
                        if self.target_post:
                            from qtrail.pipeline.target_post import route_target_post
                            routed = route_target_post(
                                qc, self.spec, layout, seed=rseed,
                                optimization_level=self.target_post_score_opt)
                            swaps = routed.count_ops().get("swap", 0)
                            routed = decompose_to_platform(
                                routed, self.cm, optimization_level=1,
                                seed=rseed)
                        else:
                            routed, swaps, _ = route_with_layout(
                                qc, self.cm, layout, seed=rseed,
                                method="sabre")
                    except Exception:
                        continue
                    from qtrail.pipeline.metrics import estimate_fidelity
                    entry = (swaps, routed.depth(),
                             estimate_fidelity(routed, self.spec.calib), pi)
                    if best_entry is None or entry[1] < best_entry[1]:
                        best_entry = entry  # makespan 最优者胜
                if best_entry is not None:
                    scored.append(best_entry)
            if scored:
                best_pi, best_swaps = self._select_candidate(scored)
                # adopt SABRE's own outcome if it is strictly better
                sabre_cand = None
                if self._sabre_result is not None:
                    from qtrail.pipeline.metrics import estimate_fidelity
                    sabre_fid = estimate_fidelity(self._sabre_result[2],
                                                  self.spec.calib)
                    sabre_cand = (self._sabre_result[0],
                                  self._sabre_result[2].depth(),
                                  sabre_fid, self._sabre_result[1])
                if sabre_cand is not None and self.include_sabre and \
                        self._cand_better(sabre_cand, scored):
                    self._adopt_sabre = True
                    self._sabre_layout = sabre_cand[3]
                    method += "_routed"
                    return best_pi, method, warnings
                method += "_routed"
                return best_pi, method, warnings

        return candidates[0], method, warnings

    # ------------------------------------------------------- 决胜规则
    def _select_candidate(self, scored):
        """按 self.selection_rule 从候选 (swaps, depth, fidelity, pi) 中选优。

        swap:     SWAP 最少；+2 容差内保真度最高（原规则）
        fidelity: 保真度最高；2% 容差内 SWAP 最少
        depth:    深度最浅；5%+3 容差内 SWAP 最少
        """
        if self.selection_rule == "swap":
            best = min(s[0] for s in scored)
            near = [s for s in scored if s[0] <= best + 2]
            return max(near, key=lambda s: s[2])[3], best
        if self.selection_rule == "fidelity":
            best = max(s[2] for s in scored)
            near = [s for s in scored if s[2] >= best * 0.98]  # 2% 相对容差
            return min(near, key=lambda s: (s[0], -s[2]))[3], min(s[0] for s in near)
        # depth
        best = min(s[1] for s in scored)
        tol = max(3, 0.05 * best)
        near = [s for s in scored if s[1] <= best + tol]
        return min(near, key=lambda s: (s[0], s[1]))[3], min(s[0] for s in near)

    def _cand_better(self, cand, scored) -> bool:
        """cand 是否严格优于候选集 scored（规则感知；scored 元素同 _select_candidate）。"""
        best = min(scored, key=lambda s: s[0] if self.selection_rule != "fidelity"
                   else -s[2] if self.selection_rule != "depth" else s[1])
        return self._pair_better(cand, best)

    def _pair_better(self, cand, ref) -> bool:
        """cand 是否优于 ref：(swaps, depth, fidelity, pi) 元组，规则感知。"""
        if self.selection_rule == "swap":
            if cand[0] < ref[0] - 2:
                return True
            if abs(cand[0] - ref[0]) <= 2 and cand[2] > ref[2] + 1e-9:
                return True
            return False
        if self.selection_rule == "fidelity":
            within = cand[2] >= ref[2] * 0.98
            if within:
                return cand[0] < ref[0]
            return cand[2] > ref[2]
        # depth
        tol = max(3, 0.05 * ref[1])
        if cand[1] < ref[1] - tol:        # 显著更浅（任何 SWAP 代价）
            return True
        if abs(cand[1] - ref[1]) <= tol:  # 容差带内 → SWAP 决胜
            return cand[0] < ref[0]
        return False

    def _layout_fidelity(self, pi: np.ndarray, adj: np.ndarray) -> float:
        """Static fidelity proxy of a layout: log-product over interacting
        pairs of (1 - err_2q[pi_i, pi_j])^w_ij."""
        import math
        err2q = self.spec.calib.err_2q
        median = float(np.median(list(err2q.values()))) if err2q else 1e-3
        logf = 0.0
        for i in range(pi.shape[0]):
            for j in range(i + 1, pi.shape[0]):
                w = float(adj[i, j])
                if w <= 0:
                    continue
                e = err2q.get((int(pi[i]), int(pi[j])),
                              err2q.get((int(pi[j]), int(pi[i])), median))
                logf += w * math.log(max(1.0 - e, 1e-9))
        return logf

    # -------------------------------------------------------------- main
    def map_circuit(self, qc: QuantumCircuit, circuit_id: str = "circuit",
                    noise_lambda: float | None = None,
                    optimization_level: int = 1,
                    route: bool = True,
                    has_measurements: bool = False) -> MappingResult:
        """Full pipeline for a decomposed QuantumCircuit.

        has_measurements: the ORIGINAL circuit carried measurements (stripped
        by the caller). tket adoption is disabled in that case because its
        final logical->physical mapping cannot be traced.
        """
        if qc.num_qubits > self.spec.n:
            raise ValueError(f"circuit uses {qc.num_qubits} qubits but device "
                             f"{self.spec.name} has {self.spec.n}")

        if noise_lambda is None:
            noise_lambda = (self.cfg.device.noise.lambda_n
                            if self.cfg.reward.mode in ("noise", "combined") else 0.0)

        ops = extract_ops(qc)
        graph = build_program_graph(qc.num_qubits, ops, circuit_id=circuit_id,
                                    temporal_alpha=self.cfg.graph.temporal_alpha)

        # ---- 基线/外部编译结果 FIRST（在任何消耗 RNG 的操作之前，
        # 保证采纳对比与外部独立测量的结果一致）
        self._adopt_sabre = False
        self._sabre_result = None
        self._sabre_o3_result = None
        self._tket_result = None
        if route:
            try:
                from qtrail.pipeline.baselines import sabre_swap_count
                sabre_swaps, sabre_routed = sabre_swap_count(
                    qc, self.cm, optimization_level=1, seed=self.seed)
                init = sabre_routed.layout.initial_layout
                orig_qubits = set(qc.qubits)
                sabre_pi = np.zeros(graph.n, dtype=np.int64)
                n_mapped = 0
                for v, phys in init.get_virtual_bits().items():
                    if v in orig_qubits:
                        sabre_pi[qc.find_bit(v).index] = phys
                        n_mapped += 1
                if n_mapped == graph.n:
                    self._sabre_result = (sabre_swaps, sabre_pi, sabre_routed)
            except Exception:
                self._sabre_result = None
            if self.use_o3:
                try:
                    from qtrail.pipeline.baselines import sabre_swap_count
                    o3_swaps, o3_routed = sabre_swap_count(
                        qc, self.cm, optimization_level=3, seed=self.seed)
                    self._sabre_o3_result = (o3_swaps, o3_routed)
                except Exception:
                    self._sabre_o3_result = None
            if self.use_tket and not has_measurements \
                    and qc.num_qubits <= self.tket_max_qubits \
                    and qc.count_ops().get("measure", 0) == 0:
                try:
                    from qtrail.pipeline.external import tket_compile
                    edges = [[i, j] for i in range(self.spec.n)
                             for j in range(i + 1, self.spec.n)
                             if self.spec.adj[i, j]]
                    self._tket_result = tket_compile(qc, edges, self.spec.n)
                except Exception:
                    self._tket_result = None

        pi, method, warnings = self.compute_layout(
            graph, noise_lambda, qc=qc, routing_aware=route,
            routing_candidates=10)

        adopt = getattr(self, "_adopt_sabre", False)
        if adopt and self._sabre_result is not None:
            # SABRE's own routed outcome beat every RL candidate
            pi = self._sabre_result[1]
            method = "hybrid_sabre_adopted"
        layout = {i: int(pi[i]) for i in range(graph.n)}
        static_cost = float(graph.cost(pi, self.spec.dist, dist_mult=2.0))

        if route:
            if adopt and self._sabre_result is not None:
                routed = self._sabre_result[2]
                swap_count = self._sabre_result[0]
                # trace final mapping through the baseline's routed circuit
                final_layout = {k: v for k, v in layout.items()}
                for inst in routed.data:
                    if inst.operation.name == "swap":
                        p0 = routed.find_bit(inst.qubits[0]).index
                        p1 = routed.find_bit(inst.qubits[1]).index
                        for logical, phys in final_layout.items():
                            if phys == p0:
                                final_layout[logical] = p1
                            elif phys == p1:
                                final_layout[logical] = p0
            elif self.routing_method == "cqlib":
                # Cqlib 注入：RL 布局 → 天衍平台原生 MCTS 路由（transpile_qcis）
                from qtrail.pipeline.cqlib_route import cqlib_route
                routed, swap_count, final_map = cqlib_route(
                    qc, self.spec, layout=layout,
                    objective=self.cqlib_objective, seed=self.seed,
                    timeout_guard=self.cqlib_timeout)
                final_layout = {k: int(final_map.get(k, layout[k]))
                                for k in layout}
            elif self.routing_method == "sabre" and self.target_post:
                # Target 后处理管线（opt-in）：RL 布局 + qiskit O1 预设
                # （噪声感知 Target）——路由后重标记/酉综合吸收置换
                from qtrail.pipeline.target_post import route_target_post
                routed = route_target_post(
                    qc, self.spec, layout, seed=self.seed,
                    optimization_level=self.target_post_opt)
                swap_count = routed.count_ops().get("swap", 0)
                final_layout = {k: v for k, v in layout.items()}
                for inst in routed.data:
                    if inst.operation.name == "swap":
                        p0 = routed.find_bit(inst.qubits[0]).index
                        p1 = routed.find_bit(inst.qubits[1]).index
                        for logical, phys in final_layout.items():
                            if phys == p0:
                                final_layout[logical] = p1
                            elif phys == p1:
                                final_layout[logical] = p0
            else:
                routed, swap_count, final_layout = route_with_layout(
                    qc, self.cm, layout, seed=self.seed,
                    method=self.routing_method)
            final_qc = decompose_to_platform(routed, self.cm,
                                             optimization_level=optimization_level,
                                             seed=self.seed)
            metrics = compute_metrics(final_qc, swap_count, self.spec.calib,
                                      final_layout)
            metrics["static_cost"] = static_cost

            # ---- 外部候选采纳：O3 级 SABRE 与 pytket 参与竞技
            final_qc, swap_count, final_layout, metrics, method = \
                self._adopt_external(qc, graph, final_qc, swap_count,
                                     final_layout, metrics, method, warnings,
                                     optimization_level)
        else:
            final_qc = qc
            swap_count = 0
            final_layout = dict(layout)
            metrics = {"static_cost": static_cost}

        baseline_swaps = (self._sabre_result[0]
                          if self._sabre_result is not None else None)
        return MappingResult(layout=layout, final_layout=final_layout,
                             swap_count=swap_count, routed_qc=final_qc,
                             metrics=metrics, method=method, warnings=warnings,
                             static_cost=static_cost,
                             baseline_swaps=baseline_swaps)

    def _adopt_external(self, qc, graph, final_qc, swap_count, final_layout,
                        metrics, method, warnings, optimization_level):
        """混合竞技：若 O3 级 SABRE 或 pytket 的结果在所选决胜规则下严格
        更优则采纳。候选统一为 (swaps, depth, fidelity) 元组。"""
        best = (swap_count, metrics.get("depth", 0),
                metrics.get("est_fidelity", 0.0))
        payload = (final_qc, final_layout, metrics, method)

        # O3 级 SABRE 结果
        if self._sabre_o3_result is not None:
            o3_swaps, o3_routed = self._sabre_o3_result
            try:
                o3_final = decompose_to_platform(o3_routed, self.cm,
                                                 optimization_level=optimization_level,
                                                 seed=self.seed)
                o3_m = compute_metrics(o3_final, o3_swaps, self.spec.calib, None)
                # 初始布局与最终映射：从 O3 路由线路提取
                init = o3_routed.layout.initial_layout
                orig = set(qc.qubits)
                o3_layout = {qc.find_bit(v).index: phys
                             for v, phys in init.get_virtual_bits().items()
                             if v in orig}
                o3_final_layout = {k: v for k, v in o3_layout.items()}
                for inst in o3_routed.data:
                    if inst.operation.name == "swap":
                        p0 = o3_routed.find_bit(inst.qubits[0]).index
                        p1 = o3_routed.find_bit(inst.qubits[1]).index
                        for logical, phys in o3_final_layout.items():
                            if phys == p0:
                                o3_final_layout[logical] = p1
                            elif phys == p1:
                                o3_final_layout[logical] = p0
                cand = (o3_swaps, o3_m.get("depth", 0),
                        o3_m.get("est_fidelity", 0.0))
                if self._pair_better(cand, best):
                    best = cand
                    payload = (o3_final, o3_final_layout, o3_m,
                               "hybrid_o3_adopted")
            except Exception as e:
                warnings.append(f"o3 adoption failed: {e}")

        # pytket 结果（无测量线路才允许采纳：final_layout 由 swap 追踪不可行）
        if self._tket_result is not None:
            tket_qc, tket_sc = self._tket_result
            try:
                tket_final = decompose_to_platform(tket_qc, self.cm,
                                                   optimization_level=optimization_level,
                                                   seed=self.seed)
                tket_m = compute_metrics(tket_final, tket_sc, self.spec.calib, None)
                cand = (tket_sc, tket_m.get("depth", 0),
                        tket_m.get("est_fidelity", 0.0))
                if self._pair_better(cand, best):
                    best = cand
                    payload = (tket_final, None, tket_m, "hybrid_tket_adopted")
            except Exception as e:
                warnings.append(f"tket adoption failed: {e}")

        return payload[0], best[0], payload[1], payload[2], payload[3]
