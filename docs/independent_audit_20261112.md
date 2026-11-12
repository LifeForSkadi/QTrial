# QTrial 独立审计报告（v8 更新后复核）

- 审计日期：2026-11-12
- 审计对象：`main` HEAD `1cf1109`（v8 管线）；`delivery/` 快照；`tables/final_report_v8/`、`tables/release_audit/`
- 审计方式：通读代码与文档；运行全套 pytest、A/B 门禁、自研合成器随机 SU(4) 复验；用 qiskit 做门级矩阵与端到端态矢量对照；与 2026-11-12 上一轮审计逐项对账
- 性质：独立审计，不是项目自带审查文档的复述；所有"已确认"项均以本次实测为准

---

## 0. 结论摘要

**v8 是一次实质性升级。** 上一轮最严重的"生产 A2 依赖 qiskit"已被真正修复：默认后端改为自研 `qtrail/pure/weyl.py`（Drury–Love Weyl + Vatan–Williams 3-CX），当前 `pure_cli --pipeline v5` 的零 qiskit 断言通过。新增的自研 VF2++、线性分解架构、log 保真度、T5–T7 不变式与候选帧类型化，使核心数学与语义层比上一版可信得多。

**但仍有 1 个 P0 与 4 个 P1：**

| 编号 | 严重度 | 问题 | 状态 |
|---|---|---|---|
| P0-1 | 致命 | 解析器 `y` 门展开错误（主仓与 delivery 相同） | **未修复** |
| P1-1 | 高 | delivery 的 `qtrail.verify` 仍依赖 qiskit，与 README 矛盾 | 未修复 |
| P1-2 | 高 | v8 summary 仍用乘积保真度、qiskit 基线 SWAP=0 口径假象 | 未修复 |
| P1-3 | 高 | v8 全量合法性与语义未重扫（release audit 仅 31 条） | 未修复 |
| P1-4 | 高 | `tables/final_report_v8/summary.md` 内容实际是 v7 表 | 未修复 |

另有 P2：紧凑性奖励 padding 偏差、真机容量边界、mid-circuit measure 重排、`_eval` 代码执行面、torch 合成回退精度不达标、1Q-only 位置不在 VF2/C1 评分集合。

---

## 1. 审计范围与方法

### 1.1 范围

- 交付管线：`qtrail/pure/`（qasm、circuit、router、post、post_v2、synthesis、weyl、vf2pp、relabel_vf2、metrics、export、mapper）、`qtrail/pure_cli.py`
- 学习层：`models/`、`problems/`、`search/`、`training/`、`devices/`
- 交付快照：`delivery/`（README、verify、pure_cli、pure/）
- 实验数据：`tables/final_report_v8/`、`tables/release_audit/`、`tables/final_report/`（基线）、`tables/compare_models/rl_o3_587_rows.jsonl`
- 文档：`docs/correctness_theorems.md`、`docs/academic_review_gaps.md`、`docs/negative_results_ledger.md`、`docs/pending_decisions.md`、根 README、`docs/technical_report.md`

### 1.2 方法

1. **静态核对**：逐模块读实现，与论文公式/项目文档对照。
2. **动态门禁**：`pytest tests/ -q`；`scripts/ab_router_kernel.py`；`scripts/ab_post_numba.py`。
3. **独立数值对照**：
   - qiskit `Operator` 对 qelib1 全部门的矩阵对照；
   - 自研 `weyl.synth_insts` 对 200 个随机 SU(4) 与特殊门的重建重叠；
   - 端到端：v8 产物按 `final_layout` 嵌入后与 qiskit 真实原线路的 Statevector 对照（含/不含 `y` 两条线路）；
   - C1 重标记最小碰撞例；
   - v5 CLI 零 qiskit 断言。
4. **数据复核**：v8 行文件去重、错误行统计、y 影响行定位、与旧基线同集对比。

---

## 2. 项目状态与学术原理

### 2.1 版本口径

- `delivery/`：v4 快照（纯映射管线零 qiskit；`pure_cli` 默认 `--pipeline v4`）。它已包含新复制进来的 `weyl.py`/`vf2pp.py`/`post_v2.py`，但默认路径仍是 v4。
- `main`：v8 研究管线。v8 数字来自 `pipeline="v5"` + `linear_decompose=True`（`PureMapper` 默认）。
- 根 README 与技术报告仍停留在 v4 时代数字（0.196/0.244、2026-08-22 版本），不能与 v8 summary 混用。

### 2.2 学术来源与实现对应

| 层 | 当前实现 | 学术来源/对应关系 |
|---|---|---|
| 布局 | GAT（4×8×128）+ 指针解码 + REINFORCE 自批评 + 多起点 LS + 噪声距离 | CO-MAP（arXiv:2605.13638）的 QAP 建模与 GAT/指针/REINFORCE 范式；时序图、深度/紧凑奖励、多拓扑为自主扩展 |
| 路由 | 自研 SABRE：front layer/extended set/decay/历史/随机平局/10n 逃逸/双步前瞻/lookahead 层 | SABRE（ASPLOS'19）+ qiskit LightSABRE 语义对齐；**不是 LightSABRE 论文算法**（项目文档已如实说明） |
| 2Q 合成 | 自研 Weyl/KAK + Vatan–Williams 3-CX（`weyl.py`，默认 `vw`） | Drury–Love（arXiv:0806.4015）+ Vatan–Williams（PRA 69, 032315）；qiskit `TwoQubitBasisDecomposer` 仅 `QTRAIL_A2_BACKEND=qiskit` 时作对照 |
| 重标记 | 自研 VF2++（非诱导单态射 + VF2++ 排序 + log 打分 + 分支限界 + 递归内预算） | Cordella VF2（TPAMI 2001）+ VF2++（arXiv:2209.15512）；对标 qiskit VF2PostLayout 架构 |
| 分解 | 线性 pass 架构 `decompose_linear`：cx 对消 → 平台翻译 → 游程合并/对角累加 → 固定轮 A2 → 最小收尾 | qiskit O3 pass 顺序的逆向对齐；项目明确承认 pass 顺序是经验反推，非先验推导 |
| 保真度 | 乘积模型（默认）+ 时间感知模型（opt-in）+ log 口径用于比较 | 乘积模型是代理指标；时间模型含 T1/Tφ 空闲退相干、rz 零误差、swap=3×2Q |
| 正确性框架 | T1 语义保持、T2 终止性、T3 条件化非劣化、T4 合法性、T5 链调度、T6 坐标帧、T7 重标记双射 + property 测试 | 自建定理层；T5/T6/T7 对应历史 P0 与 C1 双射 bug |

### 2.3 对学术新颖性的判断

- **可信的自主增量**：噪声感知 RL 布局（CO-MAP 复现 + 时序/深度/紧凑扩展）、多起点自适应 LS、管线级联合选择、线性分解架构、自研 Weyl/VW 合成器、自研 VF2++、链调度不变式、log 保真度。
- **复现/对齐而非原创**：SABRE/LightSABRE 路由语义、qiskit O3 的 pass 顺序、VF2PostLayout 架构。
- 项目文档已把"路由器是 qiskit SabreSwap 语义复现 + 自研前瞻，不是 LightSABRE"写清楚，这是正确的学术叙事。

---

## 3. 独立核验：确认成立的部分

### 3.1 自研 Weyl/Vatan–Williams 合成器（上一轮 P0 已修复）

- 代码：`qtrail/pure/weyl.py`；默认后端 `qtrail/pure/synthesis.py` 的 `_synth_block_vw`。
- 独立测试：200 个随机 SU(4)（QR 归一化）经 `synth_insts` 重建，`_block_unitary` 重叠最低 **0.9999999999999998**；h/x/y/z/s/t/rx(0.3)/ry(0.7)/rz(0.9)/u3(0.2,0.4,0.6)/cx/cz/swap 特殊门全部 overlap=1.000000000000。
- 在 `sys.modules['qiskit']=None` 的进程内，默认后端仍可合成；`pair` 取 (0,1)/(3,1)/(5,2)/(4,3) 均通过。
- 结论：上一轮"生产 A2 直接调用 qiskit `TwoQubitBasisDecomposer`"的 P0 已真正修复；qiskit 仅显式 opt-in 对照。

### 3.2 零 qiskit 断言（v5 生产路径）

```
python -m qtrail.pure_cli examples/qft5.qasm --pipeline v5 --output out/_qa_v5b
```

输出包含：

```
A2 backend  : 自研 Weyl/Vatan-Williams 闭式合成（零 qiskit 依赖）
```

`pure_cli.py:120-125` 的断言允许 `qiskit` 只在 `QTRAIL_A2_BACKEND=qiskit` 时出现；默认路径通过。

### 3.3 语义与合法性测试

| 测试/门禁 | 结果 |
|---|---|
| `tests/test_production_semantics.py`（6 语料 × linear/fixpoint × 全候选路径 × 特殊角） | **24 passed** |
| `tests/test_invariants_property.py`（T5 链调度 + T6 fold 帧 + T7 双射） | **5 passed** |
| `tests/test_candidate_frame.py` | **4 passed** |
| `tests/test_vf2pp.py` | **5 passed** |
| `tests/test_weyl_synthesis.py` | **4 passed** |
| `tests/test_fidelity_time.py` | **5 passed** |
| `tests/test_api_surface.py` | **3 passed** |
| `scripts/ab_router_kernel.py` | 0 失败 |
| `scripts/ab_post_numba.py` | ALL EQUAL |
| `pytest tests/ -q` | **210 passed, 2 failed** |

两个失败：`test_fit_block_torch_exact`（ov=0.99999939，未达 1-1e-9）与已知的 `.tianyan_key` 环境假设。

### 3.4 端到端语义（独立复验）

对一条含 h/x/rz/sx/cx/cz/u1/u2/u3 的 6 比特随机线路，v8 产物按 `final_layout` 嵌入后与 qiskit 真实原线路的 Statevector 重叠：**1.000000000**。

对 C1 双射最小复现例（`cx(0,1)` + 仅 1Q 的 `rz(2)`，移动 0→2）：修复后 `moved=0`、`final_layout` 保持双射。

### 3.5 v8 数据集与错误

- small 129/129；medium 255 完成 + 4 错误；large 199/199。
- 4 条错误均为 `timeout_900s`：`bp_feynman_hwb11`、`bp_feynman_hwb12`、`bp_vqe_uccsd_n28_vqe_uccsd_n28`、同 transpiled 变体。
- 相对 v7 的 61/259 超时，v8 的选择偏差已大幅缩小；但这 4 条恰是最难线路，报告必须写明"完成子集"。

---

## 4. 发现的问题

### P0-1：`y` 门解析错误（未修复，主仓与 delivery 同）

**位置**：`qtrail/pure/qasm.py:24`（`delivery/qtrail/pure/qasm.py:24` 相同）

```python
"y": ("u3", (math.pi / 2, math.pi / 2, math.pi / 2)),
```

qelib1 权威定义（本机 qiskit 2.5.2 的 `qelib1.inc`）：

```
gate y a { u3(pi,pi/2,pi/2) a; }
```

**证据**：

- qiskit `Operator` 对照：`y` 的 overlap = **0.707106781187**（不是相位差）。其余 qelib1 门（h/x/z/s/sdg/t/tdg/rx/ry/rz/u1/u2/u3/sx/sxdg/u/cp/ccx）全部 overlap=1。
- 端到端：含 `y` 的 3 比特线路，v8 产物与 qiskit 真实原线路的端到端 fid = **0.25**；而与 QTrial 自己解析的"原线路"比较 fid=1.0（自验空验）。
- Git 历史：该行自 `1828cee`（2026-08-22）起从未修改。

**影响**：

- v8 结果中实际含 `y` 的行：3 条 small（`vqe_uccsd_n4/n6/n8` **原始版**；transpiled 变体不含 y）与 1 条 large（`clifford_100`）。
- 排除这些行后，small 均值从 0.7349 变为 0.7471（+1.7%），中位 0.8482→0.8611。
- 更严重：任何用户输入含 `y` 的 QASM 都会被静默编译成另一条线路；现有验证（含 release audit）用同一解析器做基准，无法发现。
- `delivery/` 快照同样受影响。

**修复**：

```python
"y": ("u3", (math.pi, math.pi / 2, math.pi / 2)),
```

并补一个"qelib1 全部门 vs qiskit 矩阵"的回归测试（放在 `tests/test_pure.py` 或新文件）。

---

### P1-1：delivery 验证路径依赖 qiskit

**现象**：`delivery/README.md:35` 声称"语义等价验证使用自研精确态矢量（≤20 比特）与随机化采样（21-28 比特），无需安装模拟器"；但：

- `delivery/qtrail/verify/equivalence.py:13-14` 模块级 `from qiskit import QuantumCircuit` / `from qiskit.quantum_info import Statevector`
- `delivery/qtrail/utils/qasm_io.py:9-10` 模块级 `from qiskit import ...`

**证据**：在 `sys.modules['qiskit']=None` 的进程中，`from qtrail.verify.equivalence import verify_equivalence` 与 `from qtrail.utils.qasm_io import load_qasm2` 均抛 `ModuleNotFoundError`。README 的"评审最小验证路径"第 5 步（`python -m qtrail.verify`）在只装核心依赖的环境跑不通。

**修复**：补纯 numpy 态矢量实现；或在 README 明确标注 `qtrail.verify` 需要 qiskit 实验依赖。不能同时声称"零 qiskit"和"无需模拟器"。

---

### P1-2：v8 指标口径与表格表述

**位置**：`tables/final_report_v8/summary_v8.md`；`scripts/final_report_bench_v8.py:197`；`scripts/aggregate_v8.py`。

1. **乘积下溢**：summary 的"估计保真度"仍写 `qt_fidelity`（`exp(log)`）。large 层相对感知 O3 的中位比值 **0.056** 没有物理意义；内部守卫已改用 `log_est_fidelity`，但 summary 未同步。
2. **SWAP 口径假象**：qiskit 基线的 `count_ops()["swap"]` 恒为 0（swap 已分解成 cz），与 QTrial 的"分解前残留 swap"不可比。应改用 2Q 门数（cz 数）或 3×swap+cx。
3. **表文件混淆**：`tables/final_report_v8/summary.md` 内容实际是 **v7 表**（标题/数字均为 v7），与 `summary_v8.md` 并存，容易误引。
4. **基线区分**：v8 相对 qiskit 自带感知 O3 的中位比值是 small 0.970、medium 0.730、large 0.056；相对"RL 布局 + 感知 O3"才是 small 1.006/55%、medium 1.404/76%。两类基线必须分开表述，不能合并成"超过 O3"。

**修复**：summary 改用 log 口径；SWAP 列改 2Q 门数；删除/重命名 `summary.md`；在表头标注基线与完成子集。

---

### P1-3：v8 全量合法性与语义未重扫

**现象**：`tables/release_audit/summary.md` = 31 条样本（small 13 / medium 12 / large 6），合法 31/31、语义 15 条 fid=1。`docs/correctness_theorems.md` T4 仍写"对 587 条全样本逐一执行该扫描"，但：

- 595 行全量合法性是 **v7-pre-linear** 旧口径（`tables/final_report_v7_pre_linear/legality_rows_*`）；
- v8 的新分解/VW/VF2 改动后没有新的 587 全量合法性扫描；
- release audit 的语义对照使用与 mapper 相同的 `parse_qasm` 作为"原线路"基准，因此**解析器错误（如 y 门）不会被发现**。

**影响**：当前只能说"组件级证明 + 31 条抽样 + 24 条生产语义"，不能说"v8 全量已验证"。若对外声称"587 条全部真机可运行 / 语义正确"，证据不足。

**修复**：用当前代码重跑 `scripts/verify_all_legality.py`（v8 全量）+ 至少分层抽样的独立解析器（qiskit）语义对照；更新 T4 的表述。

---

### P2-1：紧凑性奖励 padding 偏差

**位置**：`qtrail/envs/qap_env.py:73`

```python
diam = d_pi.amax(dim=(1, 2))   # 未 mask padding
```

`collate_instances` 将 padding 的 `pi` 置 0；当布局不使用物理位 0 时，直径会被 `d(0, far)` 抬高。c0.05 模型使用 `compactness_lambda=0.05`，训练信号受影响（量级取决于 `w_sum`，通常 <1%）。修复：用 `p_mask`/`p_adj` 掩码后再取 max。

---

### P2-2：真机容量边界

**位置**：`qtrail/pure/mapper.py:439`、`qtrail/pure/layout.py:25`

`map_circuit` 只检查 `circ.n > spec.n`，未检查 `> len(usable_positions)`。真机 105 比特只有 101 可用（4 禁用）；超出时 `heuristic_layout` 的 `k % len(path)` 与平凡候选会回绕产生重复物理位，RL 解码在 `d_mask` 耗尽后也会静默发出重复动作。应在入口显式报错。

---

### P2-3：mid-circuit measure 语义

**位置**：`qtrail/pure/circuit.py:41-47`、`qtrail/pure/router.py:754`

`Circuit.measures` 独立存储，路由器在所有门之后统一发射。无经典控制的末端测量没问题；但中途测量会被静默移到末尾。若之后还有作用在同一比特上的量子门，语义不等价。`if/reset` 已显式拒绝；应在文档限定，或解析器显式拒绝 mid-circuit measure。

---

### P2-4：`_eval` 代码执行面

**位置**：`qtrail/pure/qasm.py:41`

```python
return float(eval(expr, env))  # env = {"__builtins__": {}, **binds}
```

清空 `__builtins__` 不能阻止 `().__class__.__bases__[0].__subclasses__()` 这类逃逸。映射不受信任的 QASM 时存在代码执行风险（Web 端走 qiskit 解析器，不受影响；纯 CLI 受影响）。建议换成 `ast` 白名单解析（数字/括号/四则运算/`pi`）。

---

### P2-5：torch 合成回退精度

**位置**：`tests/test_synthesis.py:141`；`qtrail/pure/synthesis.py` 的 `_synth_block_torch`

`test_fit_block_torch_exact` 实测 `ov=0.999999389`，未达测试的 `1-1e-9`，也低于生产守卫 `1-1e-7`。生产默认 `vw`，但 `vw` 守卫失败时会回退到该路径；回退块可能被守卫拒绝而跳过（语义安全，质量下降）。要么修 KAK 初值/拟合，要么明确把该路径标为"尽力而为"并降低测试断言。

---

### P2-6：1Q-only 位置不在 VF2++/C1 评分集合

**位置**：`qtrail/pure/relabel_vf2.py:130-149`、`qtrail/pure/vf2pp.py`

`active` 只含 2Q 门触碰的位置；`w1q` 只统计 active 上的 1Q 门。仅有 1Q 门的逻辑比特的 1Q 误差不参与嵌入选优；C1 的 `used` 也只由 2Q 门填充。考虑到 1Q 误差在 small 层占保真度的大头，这是布局目标与保真度指标之间的一个理论错配（joint selection 与 L3 部分补偿）。

---

### P3：文档滞后

- 根 README 仍引用 v4 的 0.196/0.244 与 587 旧表；`docs/technical_report.md` 版本仍写 2026-08-22。
- `docs/academic_review_gaps.md` §1.2 仍说 A2 调用 qiskit（§3.5 已说修复），文档内部不一致。
- README 写"109 测试"，当前 210。
- `delivery/README.md` 的验证声明与实现不符（见 P1-1）。

---

## 5. 实验数字评估

### 5.1 v8 分层结果（product 口径，来自 `summary_v8.md`）

| 层 | 完成/错误 | v8 均值/中位 fid | vs qiskit 自带感知 O3（中位/胜率） | vs RL+感知 O3（中位/胜率） |
|---|---|---|---|---|
| ≤10 | 129/0 | 0.7349/0.8482 | 0.970 / 13% | **1.006 / 55%** |
| 11-50 | 255/4 | 0.1458/0.0230 | 0.730 / 30% | **1.404 / 76%** |
| 51-105 | 199/0 | 0.0028/0.0000 | 0.056 / 19%* | 无基线行 |

\* large 的 0.056 是乘积下溢后的无意义比值，应以 log 保真度或 2Q 门数/深度比较（v8 depth 8207 vs 6814，2Q depth 3030 vs 2263，仍落后）。

### 5.2 判断

- v8 已把 v7 的 61/259 超时选择偏差缩小到 4/259，这是重要改进；但这 4 条恰是最难的 hwb/vqe_n28，报告必须写明"完成子集"。
- small/medium 的"超过 RL+感知 O3"成立；**对 qiskit 自带感知 O3 仍落后**，尤其 large。两个结论并不矛盾，但对外必须分开表述。
- 乘积保真度是相对排序的代理指标，不是物理保真度；时间模型（opt-in）与 qiskit 基线的双模型重评尚未完成（`tables/qiskit_baseline_models/` 仅有 small shard0）。
- y 门修复后，small 的绝对数字约 +1.7%，不会翻转与 qiskit 自带感知 O3 的比较结论，但这些行的现有数字不可信。

---

## 6. 修复优先级

1. **P0**：修 `y`（一行），补 qelib1 全门矩阵回归测试；重跑 3 条 small + 1 条 large 受影响线路。
2. **P1**：delivery 验证路径去 qiskit 或如实标注；修 v8 summary 的 log/2Q 口径与 `summary.md` 表文件；重跑 v8 全量合法性 + 独立解析器语义抽样。
3. **P2**：compactness mask、真机容量检查、mid-circuit measure 语义、`_eval` 白名单、torch 回退精度、1Q-only 位置纳入 VF2/C1 评分。
4. **P3**：根 README/technical_report/academic_review_gaps 同步到 v8；交付前统一测试数与数字口径。

---

## 7. 附录：关键复现命令与输出

### 7.1 测试

```bash
python -m pytest tests/ -q
# 210 passed, 2 failed in 104.40s
# FAILED tests/test_synthesis.py::test_fit_block_torch_exact
# FAILED tests/test_verify.py::test_platform_verifier_needs_key

python -m pytest tests/test_weyl_synthesis.py tests/test_vf2pp.py \
  tests/test_candidate_frame.py tests/test_invariants_property.py \
  tests/test_production_semantics.py tests/test_fidelity_time.py \
  tests/test_api_surface.py -q
# 4/5/4/5/24/5/3 passed
```

### 7.2 自研 Weyl/VW 随机复验

```python
from qtrail.pure.weyl import synth_insts
from qtrail.pure.post_v2 import _block_unitary
# 200 个随机 SU(4)：worst overlap = 0.9999999999999998
```

### 7.3 `y` 门矩阵对照

```python
# qiskit Operator 对照
# y: overlap = 0.707106781187  MISMATCH
# h/x/z/s/t/sx/sxdg/u2/cp/ccx: 1.000000000000 OK
```

### 7.4 端到端含 y 线路

```python
# v8 产物 vs QTrial 自己解析的原线路：fid = 1.000000000
# v8 产物 vs qiskit 真实原线路：      fid = 0.250000000
```

### 7.5 v5 CLI 零 qiskit

```bash
python -m qtrail.pure_cli examples/qft5.qasm --pipeline v5
# A2 backend : 自研 Weyl/Vatan-Williams 闭式合成（零 qiskit 依赖）
```

### 7.6 v8 错误行

```
bp_feynman_hwb11        timeout_900s
bp_feynman_hwb12        timeout_900s
bp_vqe_uccsd_n28_vqe_uccsd_n28              timeout_900s
bp_vqe_uccsd_n28_vqe_uccsd_n28_transpiled   timeout_900s
```

---

## 8. 与上一轮审计的对账

| 上一轮发现 | 本轮状态 |
|---|---|
| P0：A2 直接 import qiskit | **已修**：默认自研 `weyl.py`；v5 CLI 断言通过；200 随机 SU(4) 复验通过 |
| P0：`y` 门解析错误 | **仍未修**（本报告 P0-1） |
| P1：delivery 验证依赖 qiskit | **仍未修**（本报告 P1-1） |
| P1：C1 重标记非双射 | **已修**：`occupied` 全集；T7 + property 测试；最小例 moved=0 |
| P1：v7 medium 61/259 超时选择偏差 | **已改善**：v8 255/259 + 4 超时；仍需披露完成子集 |
| P1：SWAP=0 / 乘积下溢 | **部分修复**：内部用 log；summary 仍未同步 |
| P2：候选种子选择与规则不一致 | **已修**：`_seed_better` 按规则取优 |
| P2：真机容量边界 | 未修 |
| P2：mid-circuit measure | 未修 |
| P2：compactness padding | 未修 |
| P2：`_eval` 安全 | 未修 |
| P3：KAK/`fit_block_torch` 精度 | 未修（生产默认 vw，影响回退） |
| P3：`_merge_u3_chain` 死代码 | **已删除**（第 7 项清理） |
| P3：文档版本滞后 | 仍在（本报告 P3） |

---

## 9. 总体评价

**核心数学与实现层**：自研 Weyl/VW、VF2++、线性分解、T5–T7 不变式、Inst 冻结/候选帧类型化，都是真实的工程与理论进步；独立复验（200 随机 SU(4)、端到端 fid=1、210 项测试、A/B）支持其正确性。项目对"路由器是 SABRE 语义复现而非 LightSABRE""2Q 合成已自研""587 fid=1 不成立"等叙事也已如实更正，学术诚实度明显提高。

**剩余风险**：不在核心代数，而在输入边界与声称——`y` 门是唯一仍会静默改变线路的 P0；delivery 验证的 qiskit 依赖、v8 指标口径与"全量已验证"的表述，是评审最可能追问的三处。修掉 `y` 并把 v8 合法性/语义全量重扫、指标改成 log/2Q 口径后，当前 v8 管线可以作为一份自洽的交付主张。
