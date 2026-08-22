"""HSRL-Map 分层执行模拟器（路由环境，正确性由态矢量等价验证保证）。

确定性、快速：DAG 依赖跟踪 + 前沿判定 + 内容位置追踪。
输出线路与原始线路功能等价（全局相位不敏感）。
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


class RoutingEnv:
    """路由决策环境：维护线路 DAG、内容位置、前沿；执行并行门与 SWAP。

    用法：
      env = RoutingEnv(qc, cm, layout)
      while not env.done:
          if env.has_executable():
              env.execute_parallel()          # 同层并行执行（深度 +1）
          else:
              env.apply_swap(a, b)            # SWAP（深度 +1）
      指标：env.depth, env.swap_count
    """

    def __init__(self, qc: QuantumCircuit, cm, layout: dict, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_phys = cm.size()
        self.adj = {(min(a, b), max(a, b)) for a, b in cm.get_edges()}
        self.cm = cm

        import networkx as nx
        gx = nx.Graph()
        gx.add_nodes_from(range(self.n_phys))
        gx.add_edges_from(list(self.adj))
        self.dist = nx.floyd_warshall_numpy(gx).astype(np.int32)

        # ---- 指令流（1Q/2Q 门；barrier/measure 忽略）
        self.gates = []  # (kind, w1, w2, op, clbits)
        for inst in qc.data:
            name = inst.operation.name
            if name in ("barrier", "measure", "reset"):
                continue
            qs = [qc.find_bit(q).index for q in inst.qubits]
            if len(qs) == 2:
                self.gates.append(("2q", qs[0], qs[1], inst.operation,
                                   list(inst.clbits)))
            elif len(qs) == 1:
                self.gates.append(("1q", qs[0], None, inst.operation,
                                   list(inst.clbits)))

        # ---- 依赖（同线序）
        self.dep = [set() for _ in self.gates]
        self.succ = [set() for _ in self.gates]
        last = {}
        for i, g in enumerate(self.gates):
            wires = [g[1]] if g[0] == "1q" else [g[1], g[2]]
            for w in wires:
                if w in last:
                    self.dep[i].add(last[w])
                    self.succ[last[w]].add(i)
                last[w] = i

        # ---- 内容位置
        self.pos = {k: v for k, v in layout.items()}  # 逻辑 -> 物理
        self.frontier = {i for i in range(len(self.gates)) if not self.dep[i]}
        self.executed = set()
        self.done = False
        self.depth = 0
        self.swap_count = 0
        self._swap_history = {}
        self._feat_cache = None
        # 物化线路（决策记录为实际量子线路，供正确性验证与推理输出）
        self.out = QuantumCircuit(self.n_phys)
        for creg in qc.cregs:
            self.out.add_register(creg)

    # ---------------------------------------------------------- queries
    def has_executable(self) -> bool:
        """前沿是否有可执行门（1Q 恒可执行；2Q 需端点相邻）。"""
        for i in self.frontier:
            g = self.gates[i]
            if g[0] == "1q":
                return True
            p1, p2 = self.pos[g[1]], self.pos[g[2]]
            if (min(p1, p2), max(p1, p2)) in self.adj:
                return True
        return False

    def frontier_gates(self) -> list:
        return sorted(i for i in self.frontier if self.gates[i][0] == "2q")

    def current_distance(self, i: int) -> int:
        g = self.gates[i]
        p1, p2 = self.pos[g[1]], self.pos[g[2]]
        return int(self.dist[p1, p2])

    def candidates(self) -> list:
        """候选 SWAP：前沿端点所在的最短路径首边（保证距离单调下降）。"""
        cands = set()
        for i in self.frontier_gates():
            g = self.gates[i]
            p1, p2 = self.pos[g[1]], self.pos[g[2]]
            d = self.current_distance(i)
            if d <= 1:
                continue
            for nxt in self._neighbors(p1):
                if int(self.dist[nxt, p2]) == d - 1:
                    cands.add((min(p1, nxt), max(p1, nxt)))
        if not cands:
            # 防御：前沿端点相邻的任意边
            eps = set()
            for i in self.frontier_gates():
                g = self.gates[i]
                eps.add(self.pos[g[1]])
                eps.add(self.pos[g[2]])
            for (a, b) in self.adj:
                if a in eps or b in eps:
                    cands.add((a, b))
                    break
        # 迭代 4（学习化 LexiRoute）：仅路径首边候选（移除迭代 2 的随机
        # 注入——随机动作扰动学习，机制优势应来自扩展集特征而非随机探索）
        return sorted(cands)

    def _neighbors(self, p: int):
        for (a, b) in self.adj:
            if a == p:
                yield b
            elif b == p:
                yield a

    # ---------------------------------------------------------- actions
    def execute_parallel(self) -> int:
        """并行执行同层所有可执行门（深度 +1）；返回执行的门数。"""
        exec_now = [i for i in self.frontier
                    if self.gates[i][0] == "1q" or self.current_distance(i) <= 1]
        for i in exec_now:
            g = self.gates[i]
            self.frontier.discard(i)
            self.executed.add(i)
            # 物化：门执行在内容当前位置
            if g[0] == "1q":
                p = self.pos[g[1]]
                self.out.append(g[3], [self.out.qubits[p]], g[4])
            else:
                p1, p2 = self.pos[g[1]], self.pos[g[2]]
                self.out.append(g[3], [self.out.qubits[p1],
                                       self.out.qubits[p2]], g[4])
            for s in self.succ[i]:
                self.dep[s].discard(i)
                if not self.dep[s]:
                    self.frontier.add(s)
        self.depth += 1
        self._feat_cache = None  # 门执行改变剩余交互/前沿
        if not self.frontier:
            self.done = True
        return len(exec_now)

    def swap_benefit(self, a: int, b: int) -> int:
        """交换后立即可执行的前沿门数。"""
        benefit = 0
        for i in self.frontier_gates():
            g = self.gates[i]
            p1 = self.pos[g[1]]
            p2 = self.pos[g[2]]
            p1 = b if p1 == a else (a if p1 == b else p1)
            p2 = b if p2 == a else (a if p2 == b else p2)
            if (min(p1, p2), max(p1, p2)) in self.adj:
                benefit += 1
        return benefit

    def swap_pot_delta(self, a: int, b: int) -> float:
        """交换后前沿门距离和的变化（负 = 距离和下降）。"""
        delta = 0.0
        for i in self.frontier_gates():
            g = self.gates[i]
            old = int(self.dist[self.pos[g[1]], self.pos[g[2]]])
            p1 = b if self.pos[g[1]] == a else (a if self.pos[g[1]] == b
                                                else self.pos[g[1]])
            p2 = b if self.pos[g[2]] == a else (a if self.pos[g[2]] == b
                                                else self.pos[g[2]])
            delta += int(self.dist[p1, p2]) - old
        return delta

    def state_features(self):
        """策略输入（迭代 2 丰富版）：剩余交互权重 [n,n] 与 5 维节点特征 [n,5]：
          [前沿度, 阻塞门数, 前沿距离和, 剩余距离和, 最大剩余距离(关键路径代理)]
        给策略超越贪心所需的全眼前瞻信息。
        """
        if self._feat_cache is not None:
            return self._feat_cache
        n = max(self.pos.keys()) + 1
        rem = np.zeros((n, n), dtype=np.float32)
        feats = np.zeros((n, 5), dtype=np.float32)
        for i, g in enumerate(self.gates):
            if g[0] != "2q" or i in self.executed:
                continue
            rem[g[1], g[2]] += 1
            rem[g[2], g[1]] += 1
            d = int(self.dist[self.pos[g[1]], self.pos[g[2]]])
            feats[g[1], 3] += d
            feats[g[2], 3] += d
            feats[g[1], 4] = max(feats[g[1], 4], d)
            feats[g[2], 4] = max(feats[g[2], 4], d)
            if i in self.frontier:
                feats[g[1], 0] += 1
                feats[g[2], 0] += 1
                feats[g[1], 2] += d
                feats[g[2], 2] += d
            else:
                # 阻塞门数：依赖链上仍未就绪的门
                feats[g[1], 1] += 1
                feats[g[2], 1] += 1
        self._feat_cache = (rem, feats)
        return rem, feats

    def _invalidate_features(self):
        self._feat_cache = None

    def apply_swap(self, a: int, b: int) -> None:
        """执行 SWAP(a,b)（深度 +1），更新内容位置与历史。"""
        for logical, p in self.pos.items():
            if p == a:
                self.pos[logical] = b
            elif p == b:
                self.pos[logical] = a
        self.swap_count += 1
        self.depth += 1
        key = (min(a, b), max(a, b))
        self._swap_history[key] = self._swap_history.get(key, 0) + 1
        self.out.swap(a, b)

    def swap_decay(self, a: int, b: int) -> int:
        return self._swap_history.get((min(a, b), max(a, b)), 0)

    # ------------------------------------------- 扩展集（SABRE 机制复刻）
    def extended_gates(self) -> list:
        """扩展集：当前可执行门执行完毕后，依赖立即满足的下一批门。"""
        exec_now = {i for i in self.frontier
                    if self.gates[i][0] == "1q" or self.current_distance(i) <= 1}
        ext = set()
        for i in exec_now:
            for s in self.succ[i]:
                if s not in exec_now and self.dep[s] <= exec_now:
                    ext.add(s)
        return [i for i in ext if self.gates[i][0] == "2q"]

    def swap_ext_features(self, a: int, b: int) -> tuple:
        """交换 (a,b) 对扩展集的影响：(使能扩展门数, 扩展门距离和变化)。"""
        benefit = 0
        pot_delta = 0.0
        for i in self.extended_gates():
            g = self.gates[i]
            old = int(self.dist[self.pos[g[1]], self.pos[g[2]]])
            p1 = b if self.pos[g[1]] == a else (a if self.pos[g[1]] == b
                                                else self.pos[g[1]])
            p2 = b if self.pos[g[2]] == a else (a if self.pos[g[2]] == b
                                                else self.pos[g[2]])
            if (min(p1, p2), max(p1, p2)) in self.adj:
                benefit += 1
            pot_delta += int(self.dist[p1, p2]) - old
        return benefit, pot_delta

    def run_policy(self, policy_fn) -> None:
        """用外部策略函数 policy_fn(env) -> swap 边 运行至完成。"""
        while not self.done:
            if self.has_executable():
                self.execute_parallel()
            else:
                a, b = policy_fn(self)
                self.apply_swap(a, b)

    # ---------------------------------------------------------- output
    def final_circuit(self) -> QuantumCircuit:
        raise NotImplementedError("使用 RoutingEnv 仅作训练环境；"
                                  "推理输出用完整管线")

    def final_pos(self) -> dict:
        return dict(self.pos)
