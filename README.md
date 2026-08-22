# QTrial

用强化学习做量子线路映射的项目。名字是 Qubit mapping + Trial 拼的，取"量子比特布局的试炼"那个意思。

一句话介绍：给你一条量子线路（OpenQASM 2.0 格式）和一台超导芯片的拓扑 + 校准数据，QTrial 会告诉你每个逻辑比特该放在哪个物理比特上（布局）、中间该插多少个 SWAP（路由），输出一条能直接上真机的线路。布局部分用了一个小型的 GAT + 指针网络，靠 REINFORCE 稀疏奖励训练；路由和后处理是我们自己写的 SABRE 实现 + 置换重写，**整个交付管线不依赖 qiskit**。

目标硬件是[天衍量子计算云平台](https://tianyan.ustc.edu.cn)的「天衍-287」——祖冲之三号同款芯片，105 个比特，15×7 网格。项目参加了 2026 年"揭榜挂帅"擂台赛的"量子+AI 双向赋能"榜题（AI 赋能量子计算 → 量子编译与线路优化方向）。

想先看原理的，读 [docs/technical_report.md](docs/technical_report.md)（学术风格，含全部公式推导）；想看比赛版材料，读 [docs/competition_report.md](docs/competition_report.md)。本 README 负责把你从零带到能跑起来。

### 结果速览（587 条线路，完整 Benchpress + MQTBench + QUEKO）

均值口径的估计保真度：QTrial **0.196**，噪声感知 O3 0.244（我们是它的 0.80 倍），pytket 0.187（**超过**），盲目 O1 0.123 / 盲目 O3 更低。逐线路胜场（534 条全完成）：vs 噪声感知 O3 29、vs pytket **134**、vs 盲目 O1 **226**。详细分层表（≤10 / 11-50 / 51-105 比特，均值/截尾/中位三口径）见 [tables/final_report/report.md](tables/final_report/report.md)。说句实话：大线路（51-105 比特）是我们目前最弱的一层——自研后处理栈还做不到 qiskit 那种把置换完全吸收进酉重合成的 0-SWAP，差距的来源和技术报告 §9 里写得明明白白，不藏着掖着。

---

## 1. 环境安装

### 1.1 你需要的

- Python 3.10 到 3.13（我在 3.13.7 上开发，3.10+ 应该都能跑）
- 显卡不是必需的。模型只有 60 万参数左右，CPU 推理也就慢个零点几秒。有 NVIDIA 卡 + CUDA 版 torch 会更快。
- 操作系统：Windows / Linux / macOS 都行，我主要在 Windows 11 上开发。

### 1.2 一键安装（推荐）

Windows 直接双击或运行：

```bat
scripts\setup.bat
```

Linux / macOS：

```bash
bash scripts/setup.sh
```

脚本会做四件事：建虚拟环境 `.venv` → 升级 pip → 装依赖（核心依赖 + numba 加速）→ 跑一遍冒烟测试确认环境没问题。想连天衍真机或者跑 qiskit 对比实验的话，再加装完整依赖（见 1.4）。

### 1.3 手动安装（不想用脚本的话）

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

国内网络 pip 慢的话，换清华源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.4 依赖分层

[requirements.txt](requirements.txt) 里有注释，这里说人话：

| 什么时候需要 | 依赖 |
|---|---|
| 跑 `pure_cli`（交付管线） | torch, numpy, networkx, pyyaml（numba 可选，没有就自动回退纯 Python，结果完全一样） |
| 跑对比实验 / 数据集生成 | 上面的 + qiskit, pytket, pytket-qiskit, mqt.bench |
| 连天衍真机 / 用平台原生映射做对比 | 上面的 + cqlib（需要天衍平台账号） |
| Web 演示 | fastapi, uvicorn, python-multipart |

一句话原则：**评委跑你交付的东西，只需要前四个包**。qiskit 在这个项目里纯粹是"对照组选手"和数据处理工具，交付管线一行都不 import 它。这个设计是刻意的（评委环境不装 qiskit 也能跑），原理见技术报告第 2 节。

### 1.5 验证安装

```bash
python -m qtrail.pure_cli examples/ghz10.qasm
```

输出正常（出现 swap_count、fidelity 之类的指标和 out/ 下的产物）就说明装好了。顺带一提，仓库里没有权重文件也能跑——会自动降级到启发式布局，不会报错。

---

## 2. 快速上手

### 2.1 映射一条线路

```bash
python -m qtrail.pure_cli examples/qft5.qasm
```

产物在 `out/` 下：

- `<名字>_mapped.qasm` — 映射 + 路由 + 后处理之后的线路（平台基 [rz, sx, x, cz]）
- `<名字>.qcis` — 天衍平台的 QCIS 指令，平台网页上可以直接提交
- `<名字>_metrics.json` — 指标：SWAP 数、深度、2Q 深度、估计保真度

常用参数：

```
--device tianyan-287              # 纯管线目前只支持天衍-287
--rule swap|fidelity|depth        # 候选决胜规则，默认 fidelity
--seed 42                         # 随机种子
--checkpoint 路径                 # 换权重（默认用仓库自带 c0.05 权重）
```

### 2.2 验一下正确性

光看 SWAP 数没法确认映射后线路对不对，所以有个验证模块：

```bash
python -m qtrail.verify examples/qft5.qasm
```

20 比特以下做精确态矢量对比，21-28 比特做随机化采样，都会输出保真度。这个功能建议每次给评委演示前都跑一遍。

### 2.3 天衍真机验证（需要平台 API key）

**步骤一：配置 key**。两种方式任选：

```bash
# 方式 A：环境变量（本次会话生效）
set TIANYAN_LOGIN_KEY=你的key      # Windows cmd；Linux/macOS 用 export

# 方式 B：项目根目录放一个 .tianyan_key 文件（一行 key）——最省事
```

**注意：`.tianyan_key` 是你的私有凭据，打包/提交作品前务必删除！**

**步骤二：用真实校准数据映射**（输出 QCIS 用真实比特标签，平台直接可跑）：

```bash
python -m qtrail.pure_cli examples/qft5.qasm --calibration live
# 287 在维护/校准时换别的机器：
python -m qtrail.pure_cli examples/qft5.qasm --calibration live --device tianyan176
```

live 模式会拉取真实拓扑/校准/禁用比特（287 目前 105 比特 168 耦合器、4 个禁用比特），映射出的 `.qcis` 已经过平台格式校验口径（M 指令、真实标签、真实耦合器）。

**步骤三：真机验证**（提交映射线路，把实测计数回映射到逻辑空间，与经典模拟的测量分布对照）：

```bash
python -m qtrail.verify examples/qft5.qasm --on-platform \
    --qcis out/qft5.qcis --layout-json out/qft5_metrics.json \
    --machine tianyan176 --shots 12000
```

判定口径：经典保真度 ≥ 0.98 且 TVD ≤ 0.05。**这是对映射优化算法正确性的真机级证明**——本地态矢量验证保证语义，真机对照保证在真实噪声/拓扑下行为一致。

### 2.3 Web 演示

```bash
python -m qtrail.web
```

浏览器开 http://127.0.0.1:8000 ，粘贴 QASM 一键映射，结果页有指标卡和对比图。给不太懂命令行的队友/评委演示用。

---

## 3. 项目结构

```
QTrial/
├── qtrail/                      # 主包
│   ├── pure/                    # ★ 交付管线（零 qiskit），这是核心
│   │   ├── qasm.py              #   自研 OpenQASM 2.0 解析器（支持自定义 gate、ccx 分解）
│   │   ├── circuit.py           #   电路 IR：Inst + Circuit，惰性依赖 DAG
│   │   ├── router.py            #   SABRE 路由器（front layer + 衰减 + 历史惩罚）
│   │   ├── swap_kernel.py       #   路由评分 Numba 融合内核
│   │   ├── post.py              #   后处理栈：SWAP 共轭推挤/消解/尾部吸收/U3 分解
│   │   ├── post_numba.py        #   后处理 Numba 内核
│   │   ├── mapper.py            #   PureMapper：布局→评分→路由→后处理总装
│   │   ├── metrics.py           #   深度/2Q 深度/乘积保真度
│   │   ├── layout.py            #   启发式布局（蛇形/谱方法，RL 失败时的兜底）
│   │   └── export.py            #   QASM / QCIS 输出器
│   ├── pure_cli.py              # ★ python -m qtrail.pure_cli 的入口
│   ├── problems/                # 程序图构建 + 节点特征（6 维）
│   ├── devices/                 # 设备规格：天衍-287 拓扑、距离矩阵、合成校准
│   ├── models/                  # GAT / GraphTransformer 编码器 + 指针解码器
│   ├── envs/                    # 终局稀疏奖励
│   ├── training/                # REINFORCE 训练循环
│   ├── search/                  # 多起点解码 + 自适应局部搜索
│   ├── pipeline/                # qiskit 支撑版管线（内部参照/对比实验用）
│   ├── cli/  web/  verify/      # 命令行 / Web / 功能等价验证
│   └── submit.py                # 天衍真机提交（需 login key）
├── configs/                     # 设备几何 + 训练配置（YAML）
├── checkpoints/                 # 预训练权重（c0.05 为交付模型）
├── examples/                    # 示例线路
├── scripts/                     # 实验与评测脚本（见下）
├── tables/                      # 全部实验数据（见 tables/index.md）
├── tests/                       # pytest（96 测试，含态矢量等价验证）
└── docs/                        # 技术报告、竞赛报告
```

**两个容易搞混的东西**：

- `qtrail/pure/` 和 `qtrail/pipeline/` 是两套管线。pure 是交付版（零 qiskit，自研路由），pipeline 是 qiskit 支撑版（用 qiskit 的 SabreSwap 做路由，当初做机制取证和对比实验用的）。功能上后者结果略好（qiskit 生产级路由的工程积累），但前者才是评委要看的东西。
- `checkpoints/` 里一堆权重。交付用 `tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt`（名字里的 c0.05 是紧凑性正则系数）。剩下的是训练过程中的 A/B 变体，不用管。

### 3.1 公式在哪实现的

技术报告附录 B 有完整的"公式 → 代码位置"对照表。快速版：式 (1) 静态代价在 `qtrail/envs/qap_env.py`，式 (7) 交换评分在 `qtrail/pure/router.py` 的 `sabre_route` 里，式 (8) 共轭重写在 `qtrail/pure/post.py` 的 `_swap_past`，式 (9) U3 分解在 `_u3_to_rzsx`。找公式先翻那张表。

---

## 4. 数据集与实验

`data/` 下的数据集（大文件已移到项目外备份，需要时拷回来）：

- `data/benchpress/qasm/` — IBM Qiskit 官方评测基准（656 条线路）
- `data/mqtbench/stratified/` — MQTBench 分层生成缓存；`data/mqtbench/graph_pool_*.pkl` 为训练图池
- `data/Queko/` — QUEKO 基准（BIGD/BNTF/BSS）

主要实验脚本（`scripts/`）：

| 脚本 | 作用 |
|---|---|
| `final_report_bench.py` | 最终分层报告：完整 Benchpress+MQTBench+QUEKO ≈590 条 × 6 方法，原始数据逐线路落盘（可断点续跑） |
| `speed_large_vs_o3.py` | 大比特线路耗时对比（QTrial pure vs 感知 O3） |
| `ab_router_v2.py` | 路由器回归门禁（v1/v2 逐位等价） |
| `ab_post_numba.py` | 后处理回归门禁 |
| `defect_sweep.py` / `cross_topo_bench.py` / `noise_aware_o1_bench.py` | 噪声规避机制证据（缺陷扫描/跨拓扑/感知基线） |
| `cqlib_inject_bench.py` | RL 布局注入天衍平台原生 MCTS 管线 |
| `benchmark_goat.py` | 5 编译器同台对比（含 GOAT 调研） |
| `paper_parity.py` | CO-MAP 论文同设置对照 |
| `run_training.py` | 一键复现训练（三配置接力） |

结果都在 `tables/`，每个文件的含义见 [tables/index.md](tables/index.md)。

### 4.1 复现训练

```bash
python scripts/run_training.py --dev cuda
```

三配置接力：GAT 拓扑（CO-MAP 复现基线）→ GAT 噪声感知 → GT 噪声感知。RTX 5090 上总共一小时左右，CPU 也能跑就是慢。权重输出到 `checkpoints/`。

---

## 5. 测试

```bash
python -m pytest tests/ -q
```

全绿是 96 个测试。最值钱的两类：态矢量等价（路由/后处理保不保语义）和 A/B 门禁（优化前后结果是否逐位一致）。改 `qtrail/pure/` 下任何东西之后，跑一遍 `tests/test_pure.py` 是最低要求。

---

## 6. 常见问题

**Q：报 `Unknown gate 'xxx'` 解析失败？**
解析器支持 OpenQASM 2.0 的 qelib1 标准门 + u1/u2/u3 + ccx（Toffoli 分解）+ cp（受控相位分解）。用了 `cswap`、`cu1` 之类冷门门的线路会解析失败——Benchpress 里就有几条例外，测试时已如实跳过。

**Q：映射很慢？**
大线路（50 比特以上）一条几十秒是正常的：布局候选池要逐个用路由器实测评分。装了 numba 会快一档；没装也不会错，就是慢。想了解耗时构成的看技术报告 §7.4。

**Q：我想用另一台机器（比如天衍-294）？**
纯管线目前只支持天衍-287（内置几何）。换机器需要改 `configs/device_tianyan287.yaml` 的拓扑参数和 `qtrail/devices/` 里的设备构建函数。模型权重不用重训——设备特征按设备自身统计归一化，这是设计目标之一（qiskit 支撑版已有 `--calibration live` 支持从平台拉实时配置）。

**Q：web 打不开？**
确认 uvicorn 装没装（`pip install fastapi uvicorn python-multipart`），然后看终端的报错。端口被占用用 `--port` 换一个。

**Q：想提交天衍真机？**
`.qcis` 文件直接拿去平台网页提交最简单。要程序化提交的话装 cqlib、设环境变量 `TIANYAN_LOGIN_KEY`，走 `qtrail/submit.py`。

---

## 7. 致谢与出处

- CO-MAP: *A Reinforcement Learning Approach to the Qubit Allocation Problem*（本项目复现的对象，QAP 建模与 GAT+指针网络范式）
- SABRE: *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*（ASPLOS'19，路由器机制）
- REINFORCE with self-critic baseline: Kool et al. (2019)
- 祖冲之三号 / 天衍-287 芯片参数：中电信量子集团公开资料
