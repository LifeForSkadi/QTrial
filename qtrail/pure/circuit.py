"""qiskit-free 电路 IR（qtrail/pure/ 包基础）。

轻量指令列表 + 惰性依赖 DAG：纯 Python，无任何外部 SDK 依赖。
门集：cx、cz、u3、u1、swap、id、barrier（忽略）、measure。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Inst:
    """不可变指令（2026-11-13 执行顺序第 4 项：Inst 冻结）。

    历史上 Inst 是可变对象，C1/L3 重标记原地改 qubits；共享 Inst 的
    浅拷贝副本会被静默污染（qaoa_8 审计 P1）。冻结后所有"修改"必须
    dataclasses.replace 产生新对象——共享安全由类型系统保证。
    """
    name: str                 # cx/cz/u3/u1/swap/id/barrier/measure
    qubits: tuple             # (q0,) 或 (q0, q1)
    params: tuple = ()        # u3(θ,φ,λ) / u1(λ)
    cbits: tuple = ()         # measure 目标经典比特
    tu_locked: bool = False   # Tuccil 冻结段标记（块收集跳过）

    @property
    def nq(self):
        return len(self.qubits)


class Circuit:
    """指令列表电路。ops 索引即指令 id。"""

    def __init__(self, n: int, name: str = ""):
        self.n = n
        self.name = name
        self.ops: list[Inst] = []
        self.measures: list[Inst] = []
        self._deps = None  # list[set[int]]：每指令的依赖 id 集

    # ------------------------------------------------------------------ 构建
    def append(self, inst: Inst):
        if inst.name == "measure":
            self.measures.append(inst)
        else:
            self.ops.append(inst)
        self._deps = None
        return self

    def cx(self, a, b):
        return self.append(Inst("cx", (a, b)))

    def cz(self, a, b):
        return self.append(Inst("cz", (a, b)))

    def u3(self, th, ph, la, q):
        return self.append(Inst("u3", (q,), (float(th), float(ph), float(la))))

    def u1(self, la, q):
        return self.append(Inst("u1", (q,), (float(la),)))

    def swap(self, a, b):
        return self.append(Inst("swap", (a, b)))

    def id(self, q):
        return self.append(Inst("id", (q,)))

    def barrier(self, *qs):
        return self.append(Inst("barrier", tuple(qs)))

    def measure(self, q, c):
        return self.append(Inst("measure", (q,), cbits=(c,)))

    # ------------------------------------------------------------------ DAG
    def deps(self) -> list[set[int]]:
        """每指令依赖的前驱指令 id 集（按量子比特最后触碰关系）。"""
        if self._deps is None:
            all_ops = self.ops
            last = {}
            self._deps = []
            for i, inst in enumerate(all_ops):
                d = set()
                for q in inst.qubits:
                    j = last.get(q)
                    if j is not None:
                        d.add(j)
                self._deps.append(d)
                for q in inst.qubits:
                    last[q] = i
        return self._deps

    def longest_path_2q(self) -> int:
        """2Q 关键路径深度：依赖 DAG 最长路径上的 2Q 指令数。"""
        deps = self.deps()
        n = len(self.ops)
        dp = [0] * n
        for i in range(n):
            w = 1 if self.ops[i].nq == 2 else 0
            pred = max((dp[j] for j in deps[i]), default=0)
            dp[i] = pred + w
        return max(dp, default=0)

    def layer_depth(self) -> int:
        """全门并行深度（贪心分层：指令层 = 前驱最大层 + 1）。"""
        deps = self.deps()
        dp = [0] * len(self.ops)
        for i in range(len(self.ops)):
            pred = max((dp[j] for j in deps[i]), default=0)
            dp[i] = pred + 1
        return max(dp, default=0)

    def count_2q(self) -> int:
        return sum(1 for i in self.ops if i.nq == 2)

    def count(self, name: str) -> int:
        return sum(1 for i in self.ops if i.name == name)

    # ------------------------------------------------------------------ 变换

    def strip_ids(self):
        self.ops = [i for i in self.ops if i.name != "id"]
        self._deps = None
        return self

    def copy(self):
        c = Circuit(self.n, self.name)
        # Inst 冻结后（2026-11-13）共享对象安全：列表浅拷贝即可。
        # 历史：2026-11-12 审计捕获浅拷贝 + 原地改 qubits 的候选污染
        # （qaoa_8 态矢量重叠 0.0024）——该 bug 类型已被 Inst 冻结
        # 从根上消除，无需逐指令复制。
        c.ops = list(self.ops)
        c.measures = list(self.measures)
        return c
