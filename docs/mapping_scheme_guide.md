# QTrial 映射方案总览：组件、数据流与学术背景

> **面向读者**：有计算机本科基础、第一次系统接触量子编译的人。
> **目标**：读完后能向别人解释 QTrial 从一条 OpenQASM 线路到天衍-287 可执行指令的完整数据流，并知道每个组件背后的学术概念；同时为报告书的“映射方案总图”提供结构依据。
> **代码状态**：v8 生产状态（2026-11-13，git HEAD `6f4fdc7`）。交付管线是 `qtrail/pure/`；评测口径为 587 条基准、583 条完成（4 条 900 s 超时）。

---

## 0. 总览

QTrial 把“量子线路映射”拆成三个可以分别优化的子问题：

1. **布局（layout）**：给每个逻辑比特选一个物理比特，让交互频繁的逻辑比特尽量靠得近；
2. **路由（routing）**：在不改变量子语义的前提下插入 SWAP，让所有双比特门都落在芯片的相邻比特上；
3. **后处理（post-processing）**：利用 SWAP 的代数性质，把路由产生的置换“折叠”掉，再做门级重合成、消解和平台基分解，降低门数与深度。

布局用强化学习（GAT + 指针网络 + REINFORCE）学习；路由用自研 SABRE；后处理用置换共轭重写 + Weyl/Vatan-Williams 2Q 块合成 + VF2++ 重标记；最后分解到天衍平台的 `[rz, sx, x, cz]` 基，输出 QASM/QCIS。

先记住这张“数据流骨架”：

```text
OpenQASM 文本
  → Circuit IR（指令列表 + 依赖 DAG）
  → ProgramGraph（逻辑交互图） + DeviceSpec（物理耦合图 + 噪声）
  → 布局 π（逻辑比特 → 物理比特）
  → 路由后的 Circuit（物理坐标 + SWAP） + final_layout
  → 后处理后的 Circuit（平台基 [rz, sx, x, cz]）
  → QASM / QCIS / metrics.json
```

### 0.1 先修知识（不会也可以先跳过）

| 领域 | 需要知道什么 | 在本文哪里用到 |
|---|---|---|
| 线性代数 | 矩阵乘法、酉矩阵、特征值/特征向量 | 量子门、U3 分解、KAK |
| 图论 | 邻接矩阵、最短路径、子图同构 | 耦合图、QAP、VF2++ |
| 机器学习 | 全连接网络、注意力、梯度下降、策略梯度 | GAT、指针网络、REINFORCE |
| 量子基础 | qubit、门、测量、保真度 | 第 1 节会给最小背景 |

如果只学过计算机本科的算法/数据结构，没有量子力学背景，建议按“第 1 节 → 第 2 节 → 第 5 节 → 第 3 节”的顺序读；公式先看直觉，不必逐行推导。

---

## 1. 问题背景：为什么一条量子线路不能直接跑在芯片上

### 1.1 量子比特、门、线路（最小背景）

- 量子比特是二维量子系统，状态写成 \(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\)，满足 \(|\alpha|^2 + |\beta|^2 = 1\)。
- 量子门是酉矩阵（\(U^\dagger U = I\)）。单比特门作用于一个比特；双比特门（CNOT/CZ）作用于两个比特。
- 量子线路是按时间顺序排列的门序列；测量把量子态变成经典比特。
- 超导芯片上，单比特门通常是 **RZ**（虚拟 Z 旋转，几乎零误差、零时间）和 **SX/X**（微波脉冲）；双比特门是 **CZ**（通过耦合器）。

> **学术背景｜通用门集**
> 任意单比特酉操作都可以写成 \(U3(\theta,\phi,\lambda)\)；CNOT 与单比特门构成通用门集。编译器前端的任务是把高层门（H、T、Toffoli、受控相位等）展开成硬件能执行的基本门。Nielsen & Chuang 的《Quantum Computation and Quantum Information》第 4 章是标准入门材料。

### 1.2 连通性约束

芯片的物理比特不是全连接，只能做“相邻”的双比特门。这个相邻关系叫**耦合图（coupling graph）**。

- 逻辑线路里任意两个比特都可能有双比特门；
- 如果它们被放到不相邻的物理比特上，就必须插入 SWAP，让其中一个比特“走”到另一个旁边；
- 走的步数越多，SWAP 越多，错误和深度越大。

> **学术背景｜NISQ 与连通性**
> NISQ（Noisy Intermediate-Scale Quantum）时代的芯片没有容错，连通性、门错误率、退相干时间都是硬约束。量子编译（quantum compilation）的核心任务之一，就是把逻辑线路“适配”到这些约束上。关键词：connectivity constraint、coupling map、qubit mapping。

### 1.3 SWAP 是什么

SWAP 交换两个物理比特的量子态。在超导硬件上，SWAP 通常用 3 个 CNOT 实现：

```text
CX(a,b); CX(b,a); CX(a,b)
```

一次 SWAP 让一个比特“走一步”；把逻辑比特沿路径移动 \(d\) 步需要 \(d\) 次 SWAP。SWAP 本身不是免费的：3 个 CNOT 带来额外的双比特门错误和深度。

> **学术背景｜SWAP 插入**
> 路由（routing）的本质是：在不改变线路语义的前提下，插入 SWAP 把逻辑上的“远距离交互”变成物理上的“相邻交互”。SWAP 可以移动到线路的不同位置，这给后处理留下了优化空间。

### 1.4 三个子问题

| 子问题 | 做什么 | 输出 | 典型方法 |
|---|---|---|---|
| 布局 layout | 初始把逻辑比特放到物理比特上 | 双射 \(\pi\) | 启发式、QAP 求解器、强化学习 |
| 路由 routing | 插入 SWAP，使每个双比特门执行时端点相邻 | 物理线路 + `final_layout` | SABRE、A*、SAT/ILP |
| 后处理 post-processing | 推挤/消解/吸收 SWAP，门级重合成，平台基分解 | 更短、更合法的线路 | 置换共轭重写、KAK 合成、VF2 |

> **学术背景｜为什么后处理能“赚回来”**
> SWAP 与不触碰它两端比特的门对易；与单边触碰的 CNOT 对易时只改变标签；与双边 CNOT 对易时翻转控制/靶。利用这些恒等式，可以把 SWAP 一路推到线路末端，最后整体“吸收”为一次最终布局重标记（final relabeling）。这是量子编译里非常漂亮的一类代数优化。

### 1.5 数学形式：QAP

把逻辑线路的交互关系写成加权图 \(G_C=(V_C,E_C)\)：

- 节点 = 逻辑比特；
- 边权 \(w_{ij}\) = 逻辑比特 \(i,j\) 之间的双比特门数。

把设备写成耦合图 \(G_D=(V_D,E_D)\)，\(d(p,q)\) 是物理比特 \(p,q\) 之间的最短路径距离。布局是一个双射 \(\pi: V_C \to V_D\)。静态代价定义为：

\[
C(\pi) = 2 \sum_{i<j} w_{ij}\, d\bigl(\pi(i),\pi(j)\bigr)
\]

目标是最小化 \(C(\pi)\)。这就是**二次分配问题（Quadratic Assignment Problem, QAP）**的加权形式。

- QAP 是 NP-难的组合优化问题；105 个比特无法精确求解，所以用学习/启发式。
- 系数 2 是 CO-MAP 的归一化约定；常数因子不改变最优解，只改变数值尺度。核心直觉是：**交互越多的比特对，物理距离越要小**。
- 这个静态代价是路由 SWAP 数的代理（proxy），不是精确的路由成本；所以 QTrial 后面还要用路由器实测候选布局。

> **学术背景｜QAP**
> QAP 最早由 Koopmans 与 Beckmann 在 1957 年提出，用于设施选址：把 \(n\) 个设施放到 \(n\) 个位置，使“流量 × 距离”的总和最小。它是 NP-难的，精确算法只能处理很小规模；实际中常用局部搜索、模拟退火、遗传算法、强化学习。量子比特布局是 QAP 的一个自然实例。

### 1.6 评价指标

| 指标 | 含义 | 方向 |
|---|---|---|
| SWAP 数 | 路由插入的交换门数 | 越少越好 |
| 深度 depth | 全门关键路径长度 | 越浅越好 |
| 2Q 深度 | 只统计双比特门的关键路径 | 越浅越好 |
| 估计保真度 | 乘积模型 \(\prod(1-\varepsilon_{1q})\prod(1-\varepsilon_{2q})\prod(1-\varepsilon_{ro})\) | 越高越好 |
| 运行时间 | 端到端墙钟 | 越短越好 |

为什么不能只看 SWAP：SWAP 少但深度大、或 SWAP 落在噪声大的边上，实际保真度可能更差。所以 QTrial 用“SWAP / 保真度 / 深度”三种决胜规则，默认保真度。

### 1.7 两条管线（重要）

| 管线 | 位置 | 用途 | 是否零 qiskit |
|---|---|---|---|
| 交付管线 | `qtrail/pure/` | 评委跑的映射内核；`python -m qtrail.pure_cli` | 是（A2 默认自研 Weyl/VW；qiskit 仅显式对照后端） |
| qiskit 支撑版 | `qtrail/pipeline/` | 对比实验、Web 演示、机制取证 | 否 |

“零 qiskit”指的是**映射内核与交付 CLI**，不是整个仓库。训练数据生成、基线对比、真机验证、Web 演示会用到 qiskit；报告里要写清楚这个边界，避免被误解为“所有实验都不用 qiskit”。

---

## 2. 数据流：从 QASM 到 QCIS

### 2.1 输入与输出

输入是一个 OpenQASM 2.0 文件，例如 `examples/qft5.qasm` 或 `examples/ghz10.qasm`。运行：

```bash
python -m qtrail.pure_cli examples/ghz10.qasm
```

输出在 `out/` 下：

- `<name>_mapped.qasm`：平台基的映射后线路；
- `<name>.qcis`：天衍平台 QCIS 指令；
- `<name>_metrics.json`：SWAP 数、深度、2Q 深度、保真度、布局、耗时。

一次真实的 GHZ-10 运行（`out/ghz10_metrics.json`）：

```json
{
  "swap_count": 1,
  "depth": 48,
  "twoq_depth": 12,
  "est_fidelity": 0.8256,
  "layout":      {"0": 76, "1": 75, "2": 82, "3": 83, "4": 89, ...},
  "final_layout":{"0": 76, "1": 75, "2": 89, "3": 83, "4": 82, ...},
  "wall_s": 7.434
}
```

这组数字说明一个关键区别：

- **`layout` 是初始布局**：路由开始前，逻辑比特放在哪里；
- **`final_layout` 是路由后的最终位置**：SWAP 把逻辑 2 和逻辑 4 的位置交换了（82 ↔ 89）；
- 报告里说“某个逻辑比特映射到哪个物理比特”，通常指 `final_layout`；验证时则要用 `layout` 把原线路嵌入物理空间，再和映射线路比较。

### 2.2 八个阶段与数据结构

| 阶段 | 输入 | 输出 | 主要代码 |
|---|---|---|---|
| 1 解析 | QASM 文本 | `Circuit`（`Inst` 列表） | [`qtrail/pure/qasm.py`](../qtrail/pure/qasm.py) |
| 2 IR | `Inst` 列表 | 依赖 DAG、深度 | [`qtrail/pure/circuit.py`](../qtrail/pure/circuit.py) |
| 3 程序图 | `Circuit` | `ProgramGraph`（邻接矩阵 + 6 维特征） | [`qtrail/problems/program_graph.py`](../qtrail/problems/program_graph.py)、[`features.py`](../qtrail/problems/features.py) |
| 4 设备规格 | 拓扑/校准 | `DeviceSpec`（邻接、距离、噪声距离、7 维特征） | [`qtrail/devices/spec.py`](../qtrail/devices/spec.py)、[`calibration.py`](../qtrail/devices/calibration.py) |
| 5 布局 | `ProgramGraph` + `DeviceSpec` | `π`（逻辑→物理） | [`qtrail/models/`](../qtrail/models/)、[`qtrail/search/`](../qtrail/search/)、[`pure/mapper.py`](../qtrail/pure/mapper.py) |
| 6 路由 | `Circuit` + `π` | 物理 `Circuit` + `final_layout` | [`qtrail/pure/router.py`](../qtrail/pure/router.py) |
| 7 后处理 | 物理 `Circuit` | 平台基 `Circuit` | [`qtrail/pure/post.py`](../qtrail/pure/post.py)、[`post_v2.py`](../qtrail/pure/post_v2.py)、[`synthesis.py`](../qtrail/pure/synthesis.py)、[`vf2pp.py`](../qtrail/pure/vf2pp.py) |
| 8 输出/验证 | 平台基 `Circuit` | QASM/QCIS/metrics | [`qtrail/pure/export.py`](../qtrail/pure/export.py)、[`qtrail/verify/`](../qtrail/verify/) |

### 2.3 一张总图（Mermaid 草案）

```mermaid
flowchart TD
  A[OpenQASM 2.0] --> B[自研解析器<br/>qasm.py]
  B --> C[Circuit IR<br/>Inst + 依赖 DAG]
  C --> D[ProgramGraph<br/>交互权重 + 6D 特征]
  C --> E[DeviceSpec<br/>耦合图 + 噪声 + 7D 特征]
  D --> F[QAP 建模<br/>C(π)=2Σw·d]
  E --> F
  F --> G[RL 布局<br/>GAT + 指针 + 多起点 + LS]
  G --> H[候选池 + 路由竞技<br/>swap/fidelity/depth]
  H --> I[SABRE 路由<br/>front layer + 前瞻 + 衰减]
  I --> J[后处理栈<br/>折叠/重合成/重标记/消解]
  J --> K[平台基分解<br/>rz, sx, x, cz]
  K --> L[QASM / QCIS / metrics]
  L --> M[等价性验证<br/>态矢量/真机]
  N[训练数据 + REINFORCE] -.-> G
  E -. 噪声/校准 .-> G
  M -. 反馈 .-> J
```

实线 = 在线映射的数据流；虚线 = 训练、校准、验证的支撑关系。总图不要把所有后处理 pass 平铺；用一个“后处理栈”框展开。

### 2.4 一次映射的坐标变化（GHZ-10）

1. **逻辑线路**：10 个比特的 GHZ 态；逻辑比特编号 0..9。
2. **初始布局 `layout`**：`0→76, 1→75, 2→82, 3→83, 4→89, 5→81, 6→80, 7→79, 8→78, 9→85`。
3. **路由**：插入 1 个 SWAP（物理 82 ↔ 89）；`final_layout` 变为 `2→89, 4→82`，其他不变。
4. **后处理**：本次 `post_absorbed=0`；最终 12 个 2Q 门、深度 48、2Q 深度 12、估计保真度 0.8256。
5. **验证**：把原线路按 `layout` 嵌入物理空间，与映射线路做态矢量比较；理想情况下保真度为 1。

> **学习提示**：`layout` 和 `final_layout` 是整份文档里最容易混的两个词。记住一句话：**`layout` 是“起点”，`final_layout` 是“终点”**。输出线路的比特标签是物理坐标，原线路的比特标签是逻辑坐标；比较两者时要用这两个映射做“翻译”。

### 2.5 关键数据结构（读代码时对照）

| 对象 | 关键字段 | 含义 |
|---|---|---|
| `Inst` | `name`, `qubits`, `params`, `cbits` | 一条指令；`qubits` 在管线不同阶段可能是逻辑索引或物理索引 |
| `Circuit` | `n`, `ops`, `measures`, `_deps` | `n` 个比特；`ops` 指令列表；`_deps` 惰性依赖 DAG |
| `ProgramGraph` | `n`, `adj`, `node_features`, `logical_order` | `adj[i,j]` = 双比特门数；`node_features` 形状 `[n,6]` |
| `DeviceSpec` | `n`, `adj`, `dist`, `noise_dist`, `node_features`, `calib`, `disabled_qubits` | `adj` 耦合图；`dist`/`noise_dist` 形状 `[N,N]`；`node_features` 形状 `[N,7]` |
| `layout`（π） | `dict[int,int]` | 逻辑比特 → 物理比特（初始布局） |
| `final_layout` | `dict[int,int]` | 逻辑比特 → 物理比特（路由后） |
| `CandidateResult` | `circuit`, `frame`, `final_layout`, `input_layout`, `tag` | 后处理候选；`frame` = `content` / `physical` |
| `metrics` | `swap_count`, `depth`, `twoq_depth`, `est_fidelity`, `log_est_fidelity`, `wall_s` | 指标字典 |

> 读代码时先看这个表，再看每个模块的 docstring；大部分“看不懂”的情况其实是没分清 `layout` / `final_layout` / `content` / `physical` 这四个坐标约定。

---

## 3. 组件详细设计

### 3.1 解析器（`qtrail/pure/qasm.py`）

**功能**：读 OpenQASM 2.0，展开 qelib1 标准门到 `u3/u1/cx`，支持自定义 `gate`、`ccx`（Toffoli）分解、`cp(λ)` 展开、`measure`/`barrier`。

**设计要点**：

- 自研，零 qiskit；参数表达式用 AST 白名单求值，避免 `eval` 安全漏洞。
- 不支持 `opaque`/`if`/`reset`，遇到明确报错，不静默跳过。
- 自定义 gate 递归展开，深度上限 32，防止递归炸弹。
- 输出 `Circuit`：每条指令是 `Inst(name, qubits, params, cbits)`。

**学术背景**：

- OpenQASM 2.0 是量子线路的文本 IR；qelib1 是标准门库。
- 任意单比特门可写成 \(U3(\theta,\phi,\lambda)\)；CNOT/CZ 是双比特门；Toffoli 可用 15 个基本门分解（Nielsen & Chuang 图 4.9）。
- 门分解是编译器前端的工作：把高层门降到硬件能执行的基本门。

### 3.2 电路 IR（`qtrail/pure/circuit.py`）

**功能**：轻量指令列表 + 惰性依赖 DAG。

**设计要点**：

- `Inst` 是不可变 dataclass（2026-11-13 冻结），避免共享对象被原地修改；所有“修改”都用 `dataclasses.replace` 生成新对象。
- `deps()` 按“每个比特最后一次触碰”建立依赖边；`layer_depth()` 和 `longest_path_2q()` 用 DAG 最长路径计算深度。
- 惰性：只有需要时才建 DAG；修改指令后置空缓存。

**学术背景**：

- 线路的依赖关系是一个 DAG；深度 = 关键路径长度；2Q 深度 = 只计双比特门的关键路径。
- 调度/并行性是量子编译的核心：同一层内互不依赖的门可以并行执行。

### 3.3 程序图（`qtrail/problems/program_graph.py`、`features.py`）

**功能**：把逻辑线路变成加权交互图。

**设计要点**：

- 节点 = 逻辑比特；边权 \(w_{ij}\) = 两比特之间的双比特门数。
- 时序感知：`temporal_alpha=0.5`，边权额外加上 \(\alpha \times\)（该交互对出现的去重层数）；强调跨多层的持续交互。
- 6 维节点特征（CO-MAP Appendix A）：
  1. 单比特门密度 \(\mu_s\)；
  2. 控制端纠缠密度 \(\mu_c\)；
  3. 目标端纠缠密度 \(\mu_t\)；
  4. 4 跳可达影响（无向）；
  5. 加权 PageRank（\(\alpha=0.85\)）；
  6. 4 跳因果锥（控制→目标方向）。
- 静态代价 `ProgramGraph.cost` 实现上面的 QAP 公式。

**学术背景**：

- 图神经网络需要节点特征；这 6 维来自 CO-MAP 论文 Appendix A。
- PageRank 衡量节点中心性；因果锥近似衡量一个比特对后续操作的影响范围；影响分数衡量局部连通性。
- 边权用双比特门数，是因为双比特门是路由成本的主要来源；单比特门不产生 SWAP。

### 3.4 设备规格（`qtrail/devices/spec.py`、`calibration.py`）

**功能**：描述物理芯片的耦合图、距离、噪声和节点特征。

**设计要点**：

- 耦合图 `adj[N,N]`；`dist` 是无权 BFS 全对最短路径；`noise_dist` 是噪声加权 Dijkstra。
- 噪声边权：
  \[
  w_e = 1 + \alpha(\phi_e - 1) + \beta \cdot \mathrm{T1_{pen}}(e)
  \]
  其中 \(\phi_e = \mathrm{clip}(\varepsilon_{2q}(e)/\mathrm{median}, 0.2, 5)\)，\(\mathrm{T1_{pen}}\) 衡量两端 T1 差异；\(\alpha=0.5\)，\(\beta=0.1\)。
- 有效距离：
  \[
  D_{\mathrm{eff}} = D_{\mathrm{topo}} + \lambda\,(D_{\mathrm{noise}} - D_{\mathrm{topo}})
  \]
  \(\lambda=0\) 还原纯拓扑，\(\lambda=0.5\) 是生产默认。
- 7 维节点特征：T1、T2、1Q 误差、读出误差、度、行、列；按设备自身 min-max 归一化。
- 校准：
  - 合成校准（评测口径）：T1≈72 μs、T2≈50 μs、1Q≈1e-3、2Q≈3.8e-3、读出≈8.2e-3；2% 关联缺陷中心放大 5–10×。
  - 真机快照：T1 50.4 μs、T2 38.8 μs、1Q 0.07%、读出 2.4%、CZ 0.53%；4 个禁用比特、168 条可用耦合器。

**学术背景**：

- T1 = 能量弛豫时间；T2 = 退相干时间；1Q/2Q/读出误差是校准给出的错误率。
- 噪声加权距离：把“走这条边要付出的错误代价”作为边权，Dijkstra 找“最不吵”的路径；再用 \(\lambda\) 与拓扑距离插值，保持奖励景观平滑。
- 设备特征按设备自身归一化，使同一模型可以跨设备迁移（换校准/换拓扑不需要重训）。
- **口径提醒**：评测口径是 15×7 完整矩形网格（188 条边）+ 合成校准；真机是 182 个耦合器、168 条可用边、4 个禁用比特的弯曲拓扑。两者不同，报告里要区分，不能把评测数字说成“真机实测”。

### 3.5 布局层：强化学习（`qtrail/models/`、`qtrail/search/`、`pure/mapper.py`）

这是整个方案最“AI”的部分，分成四块：编码器、解码器、训练、推理搜索。

#### 3.5.1 问题建模

- 输入：`ProgramGraph`（逻辑图，6 维特征）+ `DeviceSpec`（物理图，7 维特征）。
- 输出：一个排列 \(\pi\)，第 \(i\) 个逻辑比特放到物理比特 \(\pi(i)\)。
- 目标：最小化 QAP 代价 \(C(\pi)\)。
- 序列决策：按固定顺序逐个放置逻辑比特，每步选一个未占用的物理比特。

#### 3.5.2 GAT 编码器

- 4 层 Graph Attention，8 头，\(d=128\)；每层：
  - 线性变换 \(W h_i\)；
  - 注意力分数 \(e_{ij} = \mathrm{LeakyReLU}(a^\top [W h_i \,\|\, W h_j])\)，只对邻居和自身计算；
  - softmax 得到 \(\alpha_{ij}\)；聚合 \(h_i' = \sigma(\sum_j \alpha_{ij} W h_j)\)；
  - 多头拼接；残差；GraphNorm；ReLU。
- 程序图和设备图各有一个编码器；设备编码器输出 \(N\) 个物理比特的嵌入。

**学术背景**：

- GAT（Veličković et al., ICLR 2018）用注意力聚合邻居，适合稀疏图；4 层 = 4 跳感受野。
- GraphNorm（Cai et al., ICML 2021）对每个图单独归一化，避免不同规模图的统计量互相污染。
- 为什么不用 GCN：GAT 的注意力可以学习“哪些邻居更重要”，对噪声/结构差异更灵活。
- 可选的 Graph Transformer 变体：全局注意力 + 距离偏置，用于捕捉长程依赖；生产 c0.05 是 GAT。

#### 3.5.3 指针解码器

- 16 头指针注意力；每步用当前上下文 \(c\)（当前逻辑比特嵌入 + 上一个动作的嵌入）计算每个物理比特的分数：
  \[
  s_{m,j} = C \cdot \tanh\!\left(\frac{q_m(c)\cdot k_j}{\sqrt{d}}\right),\quad C=10
  \]
- 对已占用的物理比特加 `-1e9` 掩码；softmax 后采样或 argmax；选中的（头，物理比特）作为动作。

**学术背景**：

- Pointer Network（Vinyals et al., NeurIPS 2015）解决“输出是输入的一个排列”的问题；输出维度随输入变化，普通全连接层做不到。
- 指针注意力让模型直接“指向”某个物理比特；掩码保证双射。
- \(\tanh\) 截断防止 softmax 在长距离/大分数时饱和，保留探索。

#### 3.5.4 REINFORCE 训练

- 策略 \(\pi_\theta(a|s)\)；整条轨迹得到一个终局奖励 \(R\)；损失
  \[
  L = -\bigl(R - b\bigr)\log p_\theta(\tau)
  \]
  \(b\) 是自批评贪心 rollout 的奖励。
- 奖励：
  \[
  R = -\frac{C(\pi)}{\sum w} + \lambda_{\text{depth}}\frac{\sum_q h(q)^2}{\sum_q h(q)} + \lambda_{\text{compact}}\cdot \mathrm{diameter}
  \]
  其中 \(h(q)\) 是比特 \(q\) 的交互负载；depth 项惩罚热点，compactness 项惩罚布局的物理直径。
- 训练配置（生产 c0.05）：
  - GAT 4×8、\(d=128\)、指针 16 头、`rich_context=true`；
  - batch 512、Adam lr \(3\times10^{-4}\) 余弦退火、梯度裁剪 1.0、最多 400 轮、早停 patience 50；
  - 数据：50% ER 随机图（\(p=0.3\)，规模 5–40/41–80/81–105 按 50/30/20）+ 50% 线路图池（MQTBench/QUEKO 等）；时序加权图池；
  - 奖励：`combined`（`dist_mult=2`、`depth_lambda=0.1`、`compactness_lambda=0.05`）；`temporal_alpha=0.5`。

**学术背景**：

- REINFORCE（Williams 1992）是最朴素的策略梯度；基线 \(b\) 降低方差；自批评（Kool et al. 2019）用同一策略的贪心 rollout 作基线，不需要额外网络。
- 稀疏终局奖励：只在轨迹结束时给奖励，避免中间塑形引入偏差；代价是方差大，需要基线和批量。
- 为什么用 RL 而不是监督学习：QAP 没有现成的最优标签；RL 可以直接优化代价。也可以用 A* 生成小规模最优标签（项目里有 `optimal_route.py`），但大规模无法精确求解。

#### 3.5.5 多起点 + 局部搜索（推理时）

- 多起点：5 个采样解码 + 4 个乱序贪心 + 1 个普通贪心，按代价去重，取 top-10。
- 自适应局部搜索：每个起点跑两阶段：
  - 前 30% 预算：大扰动（热比特 3 连交换、随机重分配），跳出局部最优；
  - 后 70%：首次改进单交换 + 嵌入定向交换；
  - 每步 \(O(n)\) 增量代价；patience 50；总预算 5000 步/起点。
- 候选池：RL 多起点 + \(\lambda=0\) 拓扑候选 + 启发式（Fiedler 序 → 蛇形路径）+ 平凡布局；然后每个候选用路由器实测评分，按规则决胜。
- 管线级联合选择（≤50 比特）：候选池先做廉价路由预筛，取 top-6 跑完整 v5 管线实测，再按决胜规则选优；胜者产物缓存复用，不重复路由。

**学术背景**：

- 局部搜索（2-opt/交换邻域）是 QAP 的经典启发式；多起点 + 扰动是 iterated local search 的思想。
- 嵌入定向交换：优先让交互对落在物理相邻位置，借鉴 GraphPlacement 的结构匹配。
- 候选池 + 实测评分：学习模型给出“好起点”，路由实测给出“真代价”；这是 learning-to-initialize，而不是 end-to-end。

### 3.6 路由层：SABRE（`qtrail/pure/router.py`）

**输入**：`Circuit`（逻辑坐标）+ 初始布局 \(\pi\)。
**输出**：物理坐标的 `Circuit`（含 SWAP）+ `final_layout`。

**机制**：

- **front layer**：当前所有依赖已满足且端点空闲的双比特门；可以并行执行。
- **extended set**：front layer 之后、依赖即将满足的门；用于前瞻。
- 每步考虑候选 SWAP（front layer 活跃比特的邻边）；评分
  \[
  H = \sum_{g\in F} w_g d(\mathrm{after}_g) + w_{ext}\sum_{g\in E} w_g d(\mathrm{after}_g) + \delta\cdot \mathrm{history} + \lambda_{ms}\cdot \mathrm{makespan}
  \]
- 执行所有因 SWAP 变成相邻的 front 门；重复。
- 工程细节：
  - `w_ext = 0.5/N`（qiskit decay 口径）；
  - 门权重恒为 1（SABRE 论文的逐门衰减 \(0.5^{t_g}\) 在当前实现中被恒权 + 可选禁忌替代；早期实验显示衰减 + 确定性 argmin 会陷入换位循环）；
  - 有界禁忌可选（默认关）；
  - 可选 lookahead：`lookahead_w`（部分混合种子启用 0.5；历史实验收益 15–20%）；
  - `attempt_limit=10n` 防卡死逃逸；
  - 多种子：最终评测 16 个种子取 SWAP 最少者；最终路由不截断；
  - 候选评分上限 20k/100k SWAP（仅用于候选筛选，胜者最终路由全预算）。

**学术背景**：

- SABRE（Li, Ding, Xie, ASPLOS 2019）是 NISQ 路由的代表性贪心算法；front layer + extended set + 逐门衰减 + 历史惩罚是论文的核心机制。QTrial 当前实现用恒权 + 可选禁忌 + 部分种子的 rollout 前瞻替代逐门衰减，属于工程变体（早期实验显示衰减 + 确定性 argmin 会陷入换位循环）。
- 路由是“带次序的图嵌入”：不仅要把交互图嵌入耦合图，还要考虑门的依赖顺序；所以比纯 QAP 更难。
- 贪心的优点：快、可解释、对 105 比特可用；缺点：不保证最优，所以用多起点/多种子/后处理弥补。

### 3.7 后处理层：置换代数与门级重合成

这是“把 SWAP 赚回来”的部分。分四组。

#### 3.7.1 基础后处理（v4，`pure/post.py`）

- **推挤**：SWAP(a,b) 可以穿过不触碰 a/b 的门；穿过单边触碰的 CX 时重标记；穿过双边 CX 时翻转控制/靶；把 SWAP 推到线路末端。
- **消解**：同边相邻 SWAP 对相互抵消；同向同对 CX 对消；u1 角度合并。
- **尾部吸收**：末端 SWAP 直接删除，把置换吸收进 `final_layout`。
- **连通性约束**：推挤后的 2Q 门必须仍在耦合图上，否则不允许。

**学术背景｜SWAP 共轭恒等式**：

```text
SWAP(a,b); CX(a,c) → CX(b,c); SWAP(a,b)      （单边触碰：重标记）
SWAP(a,b); CX(a,b) → CX(b,a); SWAP(a,b)      （双边触碰：控制/靶翻转）
SWAP(a,b); R(a)    → R(b);    SWAP(a,b)      （单比特门：重标记）
```

这些恒等式说明 SWAP 可以“穿过”门，只改变标签；推到末端后，整个线路等价于“原线路 + 一个最终置换”。

#### 3.7.2 v5 后处理栈（最终评测使用）

| 组件 | 作用 | 一句话机制 |
|---|---|---|
| A1 置换折叠 + LNN 重链化 | 删除 SWAP，长程门用相邻 CX 链实现 | 把 SWAP 并入标签置换 `arr`；折叠后不相邻的 2Q 门用 \(4(d-1)\) 个 CX 的 LNN 链；逐 SWAP 判断净收益，收益不够就放弃折叠 |
| A2 2Q 块重合成 | 用 3 个 CX 替换一整块 2Q 门 | 收集同一对物理比特上的连续 2Q 门块，计算块酉 \(U\)；Weyl/KAK 分解 + Vatan-Williams 3-CX 合成；恒等块删除，合成重叠不够则保留原块；Tuccil 3-CZ 模板是可选变体 |
| A3 CX 通勤对消 | 删除可对易的重复 CX | 窗口内 `cx(a,b)···cx(a,b)`，若中间门都与它俩对易，则删除这一对 |
| B 1Q 游程优化 | 合并单比特门，减少 1Q 门数 | 同一比特的 rz/sx/x 游程矩阵乘合并为 U3，再合成；H 块奇偶消解；分解后不动点循环 |
| C1 噪声感知重标记 | 把线路挪到更安静的耦合器上 | 物理位置 \(p\) 的标签整体移到空位 \(p'\)，要求相关 2Q 门仍相邻且边误差和下降；只改“落在哪些耦合器”，不改门数/深度 |
| L3 学习式重标记 | 用已训练策略再选一次标签置换 | 把路由后线路视为新逻辑线路，QAPolicy 多起点解码生成候选；过滤合法候选，按估计保真度仲裁 |
| VF2++ 全局重标记 | 找 0-SWAP 的精确嵌入 | 交互图子图同构嵌入设备图；噪声加权分数选最优；找不到时回退到局部搜索/部分嵌入 |
| 非劣化开关 | 保证后处理不劣于 v4 | v5 产物按决胜规则不劣于 v4 才采用；否则回退到 v4 |

**学术背景｜KAK/Weyl 与 3-CX**：

- 任意两比特酉 \(U \in SU(4)\) 可以写成
  \[
  U = e^{i\phi}(A\otimes B)\,\exp\!\bigl(i(aXX+bYY+cZZ)\bigr)\,(C\otimes D)
  \]
  其中 \(A,B,C,D\in SU(2)\)，\((a,b,c)\) 在 Weyl 腔中。
- Vatan & Williams（PRA 69, 032315, 2004）给出用 3 个 CX + 单比特旋转实现 canonical 块的闭式构造；对通用 SU(4)，3 个 CX 是最优的（特殊块可能 2 个或 1 个）。
- Drury & Love（arXiv:0806.4015）给出数值稳定的 Weyl 分解算法。
- QTrial 的 `weyl.py` 是独立 numpy 实现，默认 `A2_BACKEND="vw"`；qiskit 的 `TwoQubitBasisDecomposer` 仅在显式 `QTRAIL_A2_BACKEND=qiskit` 时作为对照后端。

**学术背景｜VF2++**：

- 子图同构：给定交互图 \(H\) 和设备图 \(G\)，找单射 \(f:V_H\to V_G\)，使 \(H\) 的每条边都映射到 \(G\) 的边（非诱导；\(H\) 的非边允许落到 \(G\) 的边上）。
- VF2（Cordella et al., IEEE TPAMI 23(12), 2001）是经典回溯算法，用五条可行性规则剪枝；VF2++（arXiv:2209.15512）改进节点排序（最大度起点 + BFS 度数/稀有度）。
- QTrial 还用了 Warnsdorff 结构优先（自由邻居越少越先），在长环/哈密顿路径类结构上避免搜索爆炸。
- 找到精确嵌入意味着线路的所有 2Q 门都能落在耦合边上，理论上可以 0 SWAP；噪声加权分数则在多个嵌入中选“最安静”的那个。

**学术背景｜为什么后处理是保语义的**：

- 推挤/消解/吸收都是 SWAP 共轭恒等式或自逆门的对消，逐个变换都可证明等价；
- 2Q 块重合成用块酉的精确分解，重叠校验 \(|\langle U_2|U\rangle|^2 > 1-10^{-7}\)；
- 1Q 游程合并是矩阵乘法的结合律；
- 平台基分解是精确的酉分解（模全局相位）。
- 正确性最终由态矢量等价测试和 A/B 门禁兜底，而不是“看起来对”。

#### 3.7.3 平台基分解（`decompose_to_platform_v2`）

目标基：`[rz, sx, x, cz]`。步骤：

- `cx → H-CZ-H`；
- `swap → 3×cx`；
- `u3 → RZ-SX-RZ` 模板；
- `u1 → rz`；
- 合并相邻 `rz`；消除 H 块；1Q 游程合并。

模板（模全局相位）：

```text
U3(θ,φ,λ) = RZ(λ)·SX·RZ(θ+π)·SX·RZ(φ+3π)
H         = RZ(π/2)·SX·RZ(π/2)
X         = x
θ=0       = 纯 RZ(φ+λ)
```

QCIS 输出：`RZ`、`X2P`、`CZ`、`M`；平台没有 `X` 指令，`X = X2P·X2P`；`SX = X2P`。

**学术背景**：超导平台的原生门集通常是 RZ（虚拟）、SX/X（脉冲）、CZ（耦合器）；把任意 U3 降到这个基是编译器的后端工作。

#### 3.7.4 坐标帧（content vs physical）

后处理有四类候选：`fold_only` + 全局重标记、`fold_and_relink`、`fold_keep`、裸路由。它们处在不同的“坐标帧”：

- **content 帧**：线路标签是内容/初始坐标（折叠类产物）。消费者必须用候选自己的 `final_layout` 作为输入嵌入；构造时强制 `input_layout == final_layout`。
- **physical 帧**：线路标签是物理坐标（裸路由）。输入嵌入是路由初始布局，可以和 `final_layout` 不同。

为什么重要：折叠后如果又发射物理 SWAP，会把同一个置换应用两次，语义错位。`CandidateResult` 把帧变成构造契约，在构造时检查，而不是靠注释提醒。

> **工程启示**：这是一个很好的“把隐式约定变成类型约束”的例子。量子编译里最容易出错的不是公式，而是坐标/标签/置换的约定；用类型系统把约定固化，比事后调试便宜得多。

### 3.8 输出与验证（`pure/export.py`、`qtrail/verify/`）

**输出**：

- QASM：平台基线路，measure 保留；
- QCIS：天衍平台指令；比特标签 `Q1..Q105`；`RZ`、`X2P`、`CZ`、`M`；
- metrics：SWAP 数、深度、2Q 深度、2Q 数、乘积保真度、log 保真度、时间感知保真度、墙钟。

**验证**：

- ≤20 比特：全空间态矢量，\(|\langle\psi_{\text{orig}}|\psi_{\text{mapped}}\rangle|^2 \ge 0.999\)；
- 21–28 比特：随机直积输入态采样 32 个，比较输出态保真度；
- 真机：平台经典保真度 ≥0.98 且 TVD ≤0.05；
- 发布前审计：31 条样本，真机合法 31/31、measure 语义 31/31、端到端保真度 1.000000；
- 测试：232 个 pytest 通过（另有 2 项已知环境性断言失败）。

**学术背景**：

- 态矢量模拟的复杂度是 \(2^n\)；20 比特约 1M 振幅，可行；再大就要采样/平台验证。
- 保真度 \(|\langle\psi|\phi\rangle|^2\) 对全局相位不敏感；映射只是重命名比特，所以应该等于 1（数值误差内）。
- 验证是编译器的“正确性门禁”：代数重写很容易写错，必须有独立的等价性检查。

---

## 4. 训练与模型侧（可选但重要）

生产 checkpoint：`checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt`（c0.05，v3 配置）。

| 变体 | 位置 | 用途 |
|---|---|---|
| c0.05 | `checkpoints/...c0.05_best.pt` | 生产映射；583 条评测 |
| multi7topo | `checkpoints/multi7topo_...best.pt` | 7 拓扑混合训练；跨拓扑迁移实验 |
| LAUREL | `checkpoints/demo_laurel/...best.pt` | 可学习残差；训练效率实验 |
| 路由评分器 | `checkpoints/rl_route_scorer_best.pt` | 学习式路由评分（研究变体） |

**LAUREL 残差**：把普通残差 \(x_{i+1}=f(x_i)+x_i\) 换成可学习形式：

- RW：\(out = \alpha\cdot branch + \beta\cdot resid\)，\(\alpha,\beta=\mathrm{softmax}(logits)\)；
- LR：\(out = branch + (AB+I)resid\)，\(A\in\mathbb{R}^{d\times r}\)、\(B\in\mathbb{R}^{r\times d}\)，初始化近零；
- RW+LR：两者组合。

**QuEst 保真度预测器**：输入门图（15 维节点特征：门类型 one-hot、比特索引、T1/T2、门误差、读出误差），GraphTransformer 编码 + 均值池化 + FC → sigmoid；用含 T1/T2 的 PST 标签训练；纯 PyTorch，管线仍零 qiskit。

**路由 oracle**：用 qiskit SabreSwap 生成真实 SWAP 奖励（训练侧，非交付）；小规模 A* 最优路由可作标签。

> **边界**：这些是模型侧改进/研究变体，不是主链路必需。报告里要区分“生产配置（c0.05 + v5）”和“研究变体（LAUREL、QuEst、Tuccil、学习式路由评分器）”，不要把实验变体的收益写成生产收益。

---

## 5. 总图设计建议（给报告用）

### 5.1 推荐布局：三栏 + 底部训练支撑带（验证链挂右栏末端）

```text
┌─────────────────────────────────────────────────────────────────────┐
│  左栏：逻辑侧            中栏：决策              右栏：物理侧        │
│                                                                     │
│  OpenQASM 2.0   ──►   QAP 代价 C(π)   ──►   DeviceSpec              │
│       │                    │                 （耦合图+噪声+7D）     │
│       ▼                    ▼                       │                │
│  解析器 / IR        RL 布局（GAT+指针）             │                │
│       │            多起点 + 局部搜索                │                │
│       ▼                    │                       │                │
│  ProgramGraph      候选池 + 路由竞技 ◄─────────────┘                │
│  （6D 特征）               │                                        │
│                      SABRE 路由                                     │
│                            │                                        │
│                            ▼                                        │
│                     后处理栈（折叠/重合成/重标记/消解）             │
│                            │                                        │
│                            ▼                                        │
│                     平台基分解 + QASM/QCIS/metrics                  │
│                            │                                        │
│                            ▼                                        │
│                     验证链（态矢量 / 真机 / 审计）                  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  底部支撑带：训练回路（数据 + REINFORCE + 奖励 + checkpoint）       │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 视觉编码

| 元素 | 建议 |
|---|---|
| 颜色 | 蓝色 = 主映射链路；橙色 = 噪声/校准；绿色 = 验证/输出；灰色 = 可选/研究变体 |
| 箭头 | 实线 = 数据流；虚线 = 训练/反馈；点线 = 验证 |
| 箭头标签 | 标数据类型：`Circuit`、`ProgramGraph`、`π`、`routed Circuit`、`final_layout`、`QCIS` |
| 后处理 | 不要平铺 6 个 pass；用一个“后处理栈”框，里面 4 个小 chip：折叠/重标记、A2 块重合成、A3+B 门消解、平台基 |
| 字号 | 按最终插入宽度设计：A4 正文 14–15 cm 宽，图内正文 ≥8 pt；不要先画 13 in 再缩到 14 cm |
| 口径 | 图注写清“评测口径 15×7 网格 + 合成校准；真机拓扑另见 1.2 节” |

### 5.3 每个块对应的代码/数据/学术概念

| 图中块 | 代码 | 输入 → 输出 | 学术概念 |
|---|---|---|---|
| 解析器 | `pure/qasm.py` | QASM 文本 → `Circuit` | OpenQASM、门分解 |
| 电路 IR | `pure/circuit.py` | `Circuit` → 依赖 DAG | DAG、关键路径 |
| 程序图 | `problems/program_graph.py` | `Circuit` → `ProgramGraph` | 加权图、PageRank、因果锥 |
| 设备规格 | `devices/spec.py` | 拓扑/校准 → `DeviceSpec` | 耦合图、T1/T2、Dijkstra |
| RL 布局 | `models/policy.py` | 图 + 设备 → `π` | GAT、指针网络、REINFORCE |
| 多起点/LS | `search/decoding.py`、`local_search.py` | `π` 候选 → 更优 `π` | 局部搜索、iterated local search |
| 路由竞技 | `pure/mapper.py` | 候选 `π` → 实测分数 | 代理评分、决胜规则 |
| SABRE | `pure/router.py` | `Circuit` + `π` → routed `Circuit` | front layer、lookahead、history penalty |
| 后处理 | `pure/post_v2.py`、`synthesis.py`、`vf2pp.py` | routed `Circuit` → 平台基 `Circuit` | SWAP 共轭、KAK、VF2++ |
| 输出 | `pure/export.py` | 平台基 `Circuit` → QASM/QCIS | 平台基、QCIS |
| 验证 | `qtrail/verify/` | 原线路 + 映射线路 → 保真度 | 态矢量、fidelity |
| 训练 | `training/reinforce.py` | 图池 + 设备池 → checkpoint | 策略梯度、自批评基线 |

### 5.4 图 3.1 与总图的关系

- 现有图 3.1 是“五段主链路 + 两条支撑链路”的简版，适合放在第 3 章开头；
- 新的“大图”建议放在第 2 章或第 3 章末尾，作为“方案总览”，把数据流、组件、学术概念一次讲清；
- 如果版面有限，保留图 3.1 作为简版，总图只画 6 个主阶段 + 3 条支撑链路，细节放正文/表格。

**图注草案**：

> 图 X.X QTrial 映射方案总览：左栏为逻辑侧（解析、IR、程序图），中栏为决策层（QAP 代价、RL 布局、候选池、SABRE 路由），右栏为物理侧（设备规格、后处理栈、平台基、输出与验证）。实线为数据流，虚线为训练/验证支撑。评测口径为 15×7 网格 + 合成校准；真机拓扑与禁用比特处理见 1.2 节。

---

## 6. 如何学习这个方案

### 6.1 建议路径

| 时间 | 读什么 | 目标 |
|---|---|---|
| 1 小时 | 第 0、1、2、5 节 | 记住数据流和三个子问题 |
| 半天 | 第 3.1–3.8 节 | 理解每个组件的输入/输出和“为什么” |
| 一天 | 第 3.5–3.7 节 + 技术报告对应章节 | 能解释 GAT/指针/REINFORCE/SABRE/后处理 |
| 进阶 | CO-MAP、SABRE、Vatan-Williams、VF2++、LAUREL、QuEst 论文 | 能对照代码看实现 |

### 6.2 常见误解

| 误解 | 事实 |
|---|---|
| “RL 直接输出路由” | RL 只做布局；路由是 SABRE；后处理是代数重写 |
| “零 qiskit = 所有实验零 qiskit” | 映射内核/CLI 零 qiskit；训练/对比/Web 用 qiskit |
| “保真度是实测” | 主表是乘积代理；真机验证是另一条链 |
| “SWAP 越少越好” | 不一定；深度/保真度/运行时间要一起看 |
| “layout 就是最终位置” | `final_layout` 才是最终位置 |
| “3 个 CX 一定最优” | 对通用 SU(4) 最优；特殊块可能 2 个或 1 个 |
| “噪声感知就是路由时避开坏边” | 主要在布局层；路由的噪声加权默认关（会破坏跳数优化） |

### 6.3 关键公式速查

| 概念 | 公式 |
|---|---|
| QAP 代价 | \(C(\pi)=2\sum_{i<j}w_{ij}d(\pi(i),\pi(j))\) |
| 噪声距离 | \(D_{\mathrm{eff}}=D_{\mathrm{topo}}+\lambda(D_{\mathrm{noise}}-D_{\mathrm{topo}})\) |
| 奖励 | \(R=-C/\sum w+\lambda_d\sum h^2/\sum h+\lambda_c\cdot\mathrm{diam}\) |
| 保真度 | \(F=\prod(1-\varepsilon_{1q})\prod(1-\varepsilon_{2q})\prod(1-\varepsilon_{ro})\) |
| SABRE 评分 | \(H=\sum_F w_g d+\ w_{ext}\sum_E w_g d+\delta\cdot\mathrm{hist}+\lambda_{ms}\cdot\mathrm{ms}\) |
| U3 模板 | \(U3(\theta,\phi,\lambda)=RZ(\lambda)SX\,RZ(\theta+\pi)SX\,RZ(\phi+3\pi)\) |

---

## 7. 术语表

| 术语 | 含义 |
|---|---|
| 逻辑比特 / 物理比特 | 线路里的比特 / 芯片上的比特 |
| 耦合图 | 物理比特之间允许做双比特门的图 |
| 布局 layout | 初始的逻辑→物理双射 |
| 路由 routing | 插入 SWAP，使双比特门端点相邻 |
| final_layout | 路由后的逻辑→物理双射 |
| SWAP | 交换两个物理比特状态的门 |
| QAP | 二次分配问题；布局的数学模型 |
| GAT | 图注意力网络 |
| 指针网络 | 输出为输入排列的序列模型 |
| REINFORCE | 策略梯度算法 |
| SABRE | 一种贪心量子路由算法 |
| 共轭重写 | 利用 SWAP 恒等式把交换推到末端 |
| KAK / Weyl | 两比特酉的 Cartan 分解 |
| VF2++ | 子图同构的改进回溯算法 |
| 平台基 | 硬件原生门集 `[rz, sx, x, cz]` |
| QCIS | 天衍平台的指令格式 |
| 乘积保真度 | \(\prod(1-\varepsilon)\) 的代理模型 |

---

## 8. 参考文献与进一步阅读

1. **CO-MAP**: *A Reinforcement Learning Approach to the Qubit Allocation Problem*（本项目 QAP 建模与 GAT+指针网络范式的复现对象）。
2. **SABRE**: G. Li, Y. Ding, Y. Xie, “Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices,” ASPLOS 2019.
3. **Vatan & Williams**, “Optimal quantum circuits for general two-qubit gates,” Phys. Rev. A 69, 032315 (2004).
4. **Drury & Love**, “A new algorithm for two-qubit decomposition,” arXiv:0806.4015.
5. **Cordella et al.**, “An improved algorithm for matching large graphs,” IEEE TPAMI 23(12), 2001（VF2）.
6. **VF2++**: arXiv:2209.15512.
7. **Kool et al.**, “Attention, Learn to Solve Routing Problems!”, arXiv:1803.08475（自批评基线）.
8. **Veličković et al.**, “Graph Attention Networks,” ICLR 2018.
9. **Vinyals et al.**, “Pointer Networks,” NeurIPS 2015.
10. **Williams**, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” 1992（REINFORCE）.
11. **LAUREL**: arXiv:2411.07501.
12. **QuEst**: arXiv:2210.16724.
13. **Qiskit**: SabreSwap、VF2PostLayout、UnitarySynthesis 文档（对照实现）.
14. **Nielsen & Chuang**, *Quantum Computation and Quantum Information*（量子门/线路基础）.

---

## 9. 版本与口径说明

- 本文对应 v8 生产状态（2026-11-13）；最终评测使用 v5 管线（`pipeline="v5"`），CLI 默认 v4 是稳定回退。
- 评测口径：587 条基准，583 条完成；15×7 完整网格（188 边）+ 合成校准。
- 真机口径：182 个耦合器、168 条可用边、4 个禁用比特的弯曲拓扑；与评测口径不同。
- 数字口径：报告/PPT 使用 583 条完成的 0.227 / 0.246 / 0.187 等；本文不重复具体实验数字，避免与最新图表版本混淆。
- 进一步阅读：`docs/technical_report.md`（学术推导）、`docs/competition_report.md`（竞赛版）、`README.md`（运行指南）。
