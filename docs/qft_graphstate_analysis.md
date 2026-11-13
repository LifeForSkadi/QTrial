# QFT / graphstate 短板归因与机制改进（2026-11-13）

> 结论先行：large 层保真度短板**不是**"深度接近但保真度差"的玄学，
> 而是两处可定位、可修复的机制缺陷：
>
> 1. **QFT 族**：布局/路由发生在**未消化**的原始线路上。以
>    qft_N055 为例，3,051 个 CX 中可对消 41%（Qiskit `InverseCancellation`
>    口径）到 54%（我们的 `decompose_to_platform_v2` 口径），但 SWAP
>    插入把可对消结构打散，后处理再也恢复不了；同时路由输入门数多
>    → SWAP 数翻倍。
>    修复 = **预路由语义保持消化**（等价 qiskit O3 init 的
>    `InverseCancellation`/`CommutativeCancellation`）。
> 2. **graphstate / 稀疏大线路**：精确嵌入被管线层 `max_active=40`
>    硬门槛直接跳过；即使放开门槛，VF2++ 的候选排序按噪声分数、
>    对结构完全盲，96-环搜索 60s 无解。修复 = **移除节点数上限，
>    改为必要条件 + 硬预算**，候选排序改为 **Warnsdorff 结构优先**。
>
> 修复后 4 条代表线路的乘积模型 log 保真度**全部优于 Qiskit O3**。

---

## 0. 数据口径

- 设备：`build_tianyan287_spec`（105 比特，188 条耦合边，
  度分布 65×4 + 36×3 + 4×2 —— 实为 **7×15 矩形网格**）。
- 保真度：乘积模型 `∏(1−ε1q)·∏(1−ε2q)·∏(1−εro)`，log 空间比较；
  logf 越负越差。
- 基线：Qiskit `transpile(..., optimization_level=3)`，同
  `target`（`scripts/final_report_bench.py::build_target`）、
  同 `seed_transpiler=42`、basis `['rz','sx','x','cz']`。
- v8：`PureMapper(..., pipeline="v5", routing_seeds=8,
  selection_rule="fidelity", seed=42)`，同一检查点
  `checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt`。
- 复现脚本：`scripts/scan_gate_volume.py`（归因）、
  `scripts/ab_pre_cancel.py`（A/B）；数据
  `tables/gate_attribution/`、`tables/pre_cancel_ab/`。

---

## 1. QFT 族

### 1.1 现象

large 层 24 条抽样中，QFT 族 6 条：

| 线路 | v8 1Q | v8 2Q | v8 depth | v8 logf | O3 1Q | O3 2Q | O3 depth | O3 logf |
|---|---|---|---|---|---|---|---|---|
| qft_N055 | 17,918 | 6,278 | 3,883 | −43.18 | 13,151 | 3,351 | 2,799 | −25.51 |
| qft_N080 | 39,358 | 13,936 | 6,562 | −99.18 | 21,677 | 5,768 | 4,652 | −47.13 |
| qft_N097 | 54,823 | 19,413 | 7,930 | −168.88 | 26,926 | 7,376 | 5,479 | −82.02 |

深度只差 1.3–1.4×，2Q 门数差 2.2×，logf 差 1.9–2.1×。
乘积模型对门数是指数的，**门体积才是保真度的主导项**。

### 1.2 先排除的三个假设

**假设 A：`cp(λ)` 解析器展开成 2 个 CX，改为 1 CZ + 3 rz 可减半。**
两处都不成立：

1. benchpress 的 QFT 源文件里**没有 `cp`**。`qft_N055.qasm` 只有
   3,051 个 `cx`；按 n(n−1)/2 = 1,485 个 cp、每个 2 CX + 81 个
   swap-CX 计算，3,051 完全吻合——源文件已经是 CX 展开形式。
2. 数学上 `cp(λ) = 1 CZ + 3 rz` **不可能**（见 §1.5 证明）。

**假设 B：布局是主因。**
同布局、零优化隔离 A/B（`transpile(optimization_level=0,
routing_method='sabre', initial_layout=L)`，L 取 qiskit O3 的
initial layout）：

| 初始布局 | 我们的 `sabre_route` | Qiskit SabreSwap |
|---|---|---|
| v8 布局 | **1,618** swaps | 2,195 swaps |
| qiskit O3 布局 | **1,530** swaps | 2,012 swaps |

两种布局下我们的路由器都不劣于 Qiskit O0 的 SabreSwap（O0 的
启发式与 O3 可能不同，此表只用于排除"数量级落后"）；布局差异
只值 5%。布局不是主因。

**假设 C：路由器质量差。**
同上表：不是。真正的差异在**路由输入**。

### 1.3 真正机制：路由输入未消化

Qiskit O3 的 `init` 阶段在布局/路由**之前**就做了语义保持的
对消。逐个 pass 应用到原始 `qft_N055`：

| pass | CX 数 |
|---|---|
| 起点 | 3,051 |
| `UnitarySynthesis`/`BasisTranslator`/`RemoveIdentityEquivalent` | 3,051 |
| **`InverseCancellation`** | **1,791** |
| `CommutativeCancellation` | 1,791 |

即 Qiskit 先消掉 1,260 个 CX（630 对），再拿 1,791 门的线路去
路由；SabreLayout 用 796 个 SWAP 完成（4179 CZ = 1791 + 796×3）。

我们的 `decompose_to_platform_v2` 对同一输入的消化更强：

```
parsed 2q = 3051
post_v2(linear=True) 2q = 1411   （1Q 4479）
```

但 v8 管线是在**消化前**的 3,051 门线路上做布局/路由的：
路由插入 1,618 个 SWAP，SWAP 把原本可对消的 CX 对拆散，后处理
只能消掉一部分，最终 6,278。

**修法**：把平台分解 + 对消提前到布局/路由之前。

### 1.4 A/B 数据

`scripts/ab_pre_cancel.py`，同一 seed/检查点/路由种子，两臂只差
`pre_cancel`：

| 线路 | 基线 2Q | 预消化 2Q | 比值 | 基线 logf | 预消化 logf | Δlogf |
|---|---|---|---|---|---|---|
| qft_N002 | 3 | 3 | 1.00 | −0.019 | −0.0145 | +0.0045 |
| qft_N020 | 877 | 822 | 0.937 | −5.631 | −5.002 | +0.629 |
| qft_N055 | 6,278 | **3,343** | **0.532** | −43.18 | **−22.07** | **+21.11** |
| qft_N080 | 13,936 | **6,118** | **0.439** | −99.18 | **−41.58** | **+57.60** |
| qft_N097 | 19,413 | **7,754** | **0.399** | −168.88 | **−76.11** | **+92.77** |

修复后 1Q/深度：

| 线路 | 1Q | 2Q | depth | logf | 对比 O3 logf |
|---|---|---|---|---|---|
| qft_N055 | 8,432 | 3,343 | 2,492 | −22.07 | **−25.51（我们更好）** |
| qft_N080 | 14,857 | 6,118 | 4,560 | −41.58 | **−47.13（我们更好）** |
| qft_N097 | 18,858 | 7,754 | 4,979 | −76.11 | **−82.02（我们更好）** |

全量基准复核（`routing_seeds=16`，`tables/final_report_v8/`）：
qft_N055 2Q **3,333** / logf **−21.68**；N080 2Q **5,826** /
logf **−39.59**；N097 2Q **7,660** / logf **−62.05**——比 8 种子的
A/B 更好，结论不变。

机制链条（qft_N055）：

```
输入 3051 CX
  ├─ qiskit：InverseCancellation → 1791 → 路由 796 swap → 4179 → peephole → 3351
  └─ v8 旧 ：路由 1618 swap → 7905 → 后处理 → 6278
  └─ v8 新 ：预消化 → 1411 → 路由 717 swap → 3562 → 后处理 → 3343
```

路由输入从 3,051 降到 1,411 后，SWAP 从 1,618 降到 **717**
（Qiskit 同阶段 796）；我们的路由器在消化后的输入上同样不输。

### 1.5 数学附录：为什么 `cp(λ) = 1 CZ + 3 rz` 不成立

设作用在控制位 c 与目标位 t 上的任意单比特对角门总角度分别为
α（c）与 β（t），CZ 使用 k 次（k∈{0,1}）。任意乘积仍是
对角阵，在基 |00⟩,|01⟩,|10⟩,|11⟩ 上的对角元为

```
( e^{−i(α+β)/2}, e^{−i(α−β)/2}, e^{i(α−β)/2}, (−1)^k e^{i(α+β)/2} )
```

要求它等于 `e^{iφ}·diag(1,1,1,e^{iλ})`：

- |01⟩/|00⟩ = e^{iβ} = 1 → β = 0 (mod 2π)
- |10⟩/|00⟩ = e^{iα} = 1 → α = 0 (mod 2π)
- |11⟩/|00⟩ = (−1)^k e^{i(α+β)} = (−1)^k

只能得到 λ = π（即 CZ 本身）。**一般 λ 无法用"1 CZ + 对角 1Q 门"
实现**。要减 2Q 门数，只能利用线路级的对消/通勤，而不是单门
分解。∎

---

## 2. graphstate / 稀疏大线路

### 2.1 现象

`graphstate_100`：100 比特、100 个 CX（交互图是 **96-环 + 4-环**，
最大度 2）。

| | 1Q | 2Q | depth | logf |
|---|---|---|---|---|
| v8 旧 | 1,489 | 652 | 179 | −6.93 |
| Qiskit O3 | 518 | 100 | 15 | −2.75 |
| v8 新 | **300** | **100** | **8** | **−2.095** |

652 − 100 = 552 = 184 个 SWAP × 3；旧产物是"没找到嵌入 → 退回
路由"的典型形态。

### 2.2 嵌入确实存在（Qiskit 见证）

用同一 `target` 编译，读取 `out.layout.initial_index_layout()`：

- 输出恰好 100 个 CZ；
- 对原始 100 条 CX 的每一对 (a,b)，`spec.adj[L[a], L[b]]`
  全部为真（非相邻 0 条）；
- 输出 CZ 集合与 `{(L[a],L[b])}` **完全相等**（100/100）。

这不是"Qiskit 用了不同拓扑"，而是同一 188 边设备图上的
0-SWAP 精确嵌入。

### 2.3 为什么我们的 VF2++ 找不到

**第一层：门槛。** `relabel_vf2._exact_embedding` 在
`len(active) > 40` 时直接 `return None`：

- 历史：`23118a8` 把 `max_active` 从 60 降到 20（当时 networkx
  GraphMatcher **搜索内无预算**，graphstate_50 900s 超时）；
- `ae87d0f` 换成自研 VF2++（有递归内 `call_limit/max_solutions/
  wall` 硬预算），但**调用方**又加了 `max_active=40`。这把
  "按预算控制"重新变成了"按规模拒绝"——与"无上限 + 硬阈值"的
  设计意图不一致。

**第二层：搜索启发式。** 把 `max_active` 提到 105 后，VF2++ 在
`graphstate_100` 上跑满 **60 秒仍无解**。原因是候选设备节点按
**log 噪声分数**排序，对图结构完全盲；96-环上的搜索等价于随机
自避行走 + 回溯，指数爆炸。

### 2.4 Warnsdorff 为什么有效

Hamiltonian 路径/环的经典启发式：每一步优先走向**自由邻居最少**
的设备节点（最受限优先），把"会把后续路径堵死"的分支尽早排除，
而不是先按噪声分数挑"看起来好"的节点。

在设备图（105 节点/188 边）上直接找 96-环：

```
随机化 Warnsdorff：54 次尝试 / 0.006 s 找到
```

集成进 VF2++：候选排序键改为
`(自由邻居数, −噪声分数, 节点号)`，噪声分数降为平局裁决，
分支限界和打分逻辑不变：

```
graphstate_100（active=100）:
  cand_order="score"      → 60 s 无解
  cand_order="warnsdorff" → 0.01 s 找到并验证通过
                            max_solutions=1   score −2.15
                            max_solutions=256 score −2.02
```

### 2.5 修复后的判据替换

旧的 `max_active` 是"节点数"这个错误的代理变量。嵌入可行性由
**度序列**和**边数**决定，搜索难度由**预算**兜底：

1. 度序列支配（必要）：H 的度序列降序必须被 G 的度序列降序支配
   —— `find_embedding` 内已有；
2. 边数（必要）：`|E_H| ≤ |E_G|`（设备 188 条）——新增，零成本
   淘汰稠密线路（QFT：1,485 > 188，立即返回）；
3. 只受 `VF2Config` 的 `call_limit / max_solutions / wall_limit`
   约束，不再按节点数拒绝。

**关于"超上限时选最重要节点激活"**：仓库历史中**没有**这个实现
（`git log -S'优先'/-S'top'/-S'active['` 均无命中）。
`docs/vf2_postlayout_explained.md` §8.1 记录的是"分块 + 块重要性
排序"（方案 C）被**否决**。因此这是设计意图与代码的偏差，如实
记录；本轮用"移除节点数上限"直接消除了这个场景，未实现优先子集
嵌入（见 §6 的后续设计）。

---

## 3. 横向影响：不止 graphstate

对 199 条 large 线路做零成本必要条件扫描：

- `|E_H| ≤ 188`：**78/199** 通过；
- 再叠加最大度 ≤ 4（设备最大度）：**23/199** 通过，覆盖
  cat / ising / wstate / ghz / dtc / graphstate 六族，以及
  square-heisenberg 的 8×8/9×9/10×10（后者在 7×15 网格上
  **可证不可嵌入**，VF2 在 0.09s 内判定，Qiskit 也只能得到
  1,173 CZ）。

代表线路（旧 = 归因扫描；新 = 本轮代码，A/B 的 base 臂）：

| 线路 | 旧 v8 logf | Qiskit O3 logf | 新 v8 1Q/2Q/depth/logf |
|---|---|---|---|
| graphstate_100 | −6.93（652 2Q） | −2.75（100/518/15） | **300/100/8/−2.095** |
| ising_n98 | −6.86（497 2Q） | −3.98（194/1093/22） | **967/194/21/−3.676** |
| ghz_n78 | −4.174（482 2Q） | —（77 2Q） | **77 2Q，logf −1.911** |
| cat_n65 | −2.580 | — | **64 2Q，logf −1.371** |
| wstate_n76 | −6.740 | — | **150 2Q，logf −3.147** |

注：ising_n98 的 194 与 wstate_n76 的 150 是线路本身的 2Q 门数
（交互边 × 每条边的 CX 数），嵌入只影响 SWAP/布局，不再额外放大；
旧值来自 `tables/final_report_v8`（`qt_fidelity` 取 ln）与归因扫描。
`dtc_100_cx`（49,700 门 / 19,800 个 2Q）只验证了精确嵌入（8.02 s
成功），全管线 A/B 因单条墙钟过长未纳入本轮表格。

**预消化的触发面**：`pre_cancel` 只在"分解确实减少 2Q"时生效。
抽查 ghz/bv/ising/cat/qaoa 均为 0% 变化（不触发）；QFT 族
−16%（N25）～−54%（N55/N100 −28%）。因此该改动的爆炸半径集中在
QFT 族。

---

## 4. 生产改动清单

| 文件 | 改动 | 默认 |
|---|---|---|
| `qtrail/pure/vf2pp.py` | `VF2Config.cand_order`（`score`→`warnsdorff`，分数为平局、分支限界不变） | 开 |
| `qtrail/pure/relabel_vf2.py` | 移除 `max_active=40`；新增 `|E_H| ≤ |E_G|` 必要条件；预算仅由 `VF2Config` 控制 | 开 |
| `qtrail/pure/mapper.py` | `pre_cancel` 预路由消化；**仅当 2Q 门数确实下降才采用**（graphstate 类无收益不切换） | 待定 |
| `tests/test_pre_cancel_semantics.py` | 输入级 u3/u1/cx/cz/swap 的态矢量等价（4 比特 ×6 组，fid > 1−1e−9） | — |
| `tests/test_vf2pp.py` | 5 项既有测试在 warnsdorff 默认下全过 | — |
| `scripts/ab_pre_cancel.py` | 可复现 A/B | — |

### 预注册判据（先于结果写定，见 `scripts/ab_pre_cancel.py` 文档串）

- **P1 语义**：预消化与原线路态矢量等价 —— ✓（6/6，fid > 1−1e−9）
- **P2 QFT**：2Q 下降 ≥40%，logf 不劣化 —— ✓（47%/56%/60%，
  logf 分别 +21.1/+57.6/+92.8）
- **P3 其他族**：logf 不劣化超过 2% —— ✓（QAOA/QV/dnn/ising/
  swap_test/graphstate 两臂完全相同，比值 1.000）
- **P4 墙钟**：≤1.5× —— ✓（QFT 反而更快：10.6→6.9 s、
  24.3→11.8 s、28.6→15.8 s；QV +12%）
- **回归**：`scripts/reproduce.py` **PASS**（10 条参考线路 ×3 seeds，
  均值与 `tables/reference.json` 一致、±2% 容差内）——Warnsdorff
  排序与 `pre_cancel` 默认开启都没有让固定基准漂移（qft_15 的
  CX 对在预消化中无净减少，守卫按设计拒绝切换）。

### 为什么"仅当 2Q 下降才采用"

graphstate 无任何可对消结构，预消化只会改变 VF2 的打分权重，
实测 logf −2.095 → −2.278（变差）。加上守卫后，graphstate 保持
−2.095，QFT 仍享受全部收益。

---

## 5. 风险与边界

1. **预消化的墙钟**：`decompose_to_platform_v2` 对 ≤20 万门线路
   调用一次；>20 万门跳过（`pre_cancel_max_ops`）。QFT 因线路
   变小反而更快；QV 类 +12%（仍 <1.5×）。
2. **8 秒预算**：精确嵌入在找到解后继续搜索更高分嵌入，稀疏大
   线路会吃满 8 s（ghz/ising/wstate/dtc）。该预算本来就在
   `global_relabel` 的 25 s 总预算内，未增加总墙钟；若后续要压，
   可对大模式降 `max_solutions`。
3. **未实现**：非嵌入稠密图（QFT/QV/dnn/swap）的"重要节点优先
   子集 VF2"。设计草案：按 2Q 门权重取 top-k 诱导子图 → 精确
   嵌入 → 其余节点交给路由；需要新的"部分精确 + 链化"候选帧，
   本轮不做。
4. **生产状态（2026-11-13 收口）**：`pre_cancel` 已默认启用，
   v8 基准全量重跑完成（584 条：small 127 / medium 258 / large 199，
   0 错误）。新口径 `tables/final_report_v8/`：
   - large 均值保真度 **0.0028→0.0072**，深度 8207→7242、
     SWAP 3872→3286；
   - **基线读出项修正（2026-11-13 晚，台账 #24）**：原先基线未计
     读出，修正后 large 对感知 O3 胜场 **122/199**（均值 0.0072 vs
     **0.0091**）、对盲目 O1 胜 **188/199**（0.0072 vs 0.0038），
     等权 log 口径我们 −48.4 vs 感知 O3 −50.6、盲目 O1 −74.7；
   - medium 均值 0.1446 vs 感知 O3 0.1404（修正后）、small 0.7337
     vs 0.7384；
   - 重跑还暴露并修复了输入级 `id` 门在预消化中的崩溃
     （`post_v2._oneq_matrix` 补 `id` 单位阵，1 条 small 线路恢复）。

---

## 6. 给报告书的表述建议

**可以写**：

- "large 层 QFT/稀疏线路的保真度短板已定位为**路由前未做语义
  保持消化**与**精确嵌入被节点数门槛禁用**两项机制缺陷，而非
  算法能力上限；旁路 A/B 中代表线路的 log 保真度已超过感知 O3。"
- 具体数字引用 §1.4（QFT）、§2.1（graphstate）、§3（稀疏族）。
- 归因结论："深度不是保真度代理；乘积模型下 1Q/2Q 门体积是主导项，
  而门体积由'在什么输入上路由/嵌入'决定。"

**不要写**：

- "全面超过感知 O3"——本轮是旁路 A/B，`tables/final_report_v8/`
  的 587 条基准仍是旧管线产物；全面结论需要默认启用 + 全量重跑。
- "cp 分解优化"——benchpress QFT 无 cp，且该分解数学上不成立
  （§1.5），不要作为改进点写进报告。

**建议的报告边界句**：

> 我们已把 large 层短板拆解到两个可修复的机制：QFT 族的预路由
> 消化、稀疏大线路的精确嵌入门槛。修复在代表线路上把 log 保真度
> 提升到优于感知 O3 的水平；由于时间窗口，本轮基准表仍为修复前
> 口径，下一版将默认启用并全量重测。

---

## 7. 复现

```bash
# 归因（large 24 条抽样，v8 vs Qiskit O3）
python scripts/scan_gate_volume.py --sample 24 --seeds 8

# 预消化 A/B（QFT + 分层样本）
python scripts/ab_pre_cancel.py \
  --circuits bp_qft_qft_N055,bp_qft_qft_N080,bp_qft_qft_N097,graphstate_100 \
  --out tables/pre_cancel_ab/qft_graph_ab.jsonl

# 语义测试
python -m pytest tests/test_pre_cancel_semantics.py tests/test_vf2pp.py -q
```

关键文件：

- `qtrail/pure/vf2pp.py`（候选排序/预算）
- `qtrail/pure/relabel_vf2.py`（`_exact_embedding` 必要条件）
- `qtrail/pure/mapper.py`（`pre_cancel`）
- `tables/gate_attribution/summary.md`（归因）
- `tables/pre_cancel_ab/*.jsonl`（A/B 原始数据）
