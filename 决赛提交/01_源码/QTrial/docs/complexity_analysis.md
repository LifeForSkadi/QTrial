# 管线时间复杂度分析（2026-11-13，用户要求补的欠账）

记 n = 线路门数、q = 比特数、m = swap 数、C = 候选数、S = 种子数、
R = 循环轮数（12）。

## 1. 布局阶段

| 组件 | 复杂度 | 说明 |
|---|---|---|
| build_program_graph | O(n) | 单遍门统计（51 万门 ~30s——Python 循环常数大，可 numpy 化） |
| GAT 解码（单起点） | O(q²·d) 前向 + q 步自回归 | d=特征维；与 n 无关（特征已聚合） |
| multistart_decode | O(samples·解码) | 串行；可 batch（decode_layouts_batch 已存在） |
| improve_layout（LS） | O(q²·LS 步) | 与 n 无关 |
| _score_candidates | **O(C·S·T_route)** | T_route = 单种子路由时间；hwb12 级 ≈ 8s → 11×16×8 ≈ 23 分钟（已修：巨型 4×2×8 ≈ 1 分钟） |

## 2. 路由阶段

| 组件 | 复杂度 | 说明 |
|---|---|---|
| sabre_route 单种子 | O(n·q) 每门评分扩展集 | 每门 O(扩展集·q)；51 万门 × 20 比特 ≈ 8s 实测 |
| sabre_route trials=16 | 16×（线程并行 8） | 实测 120s（hwb12） |
| 触顶行为 | 修复前：截断 O(1) 但语义错；修复后：raise | 已修 |

## 3. 后处理阶段

| 组件 | 复杂度 | 备注 |
|---|---|---|
| post_route（push_swaps） | O(m·n) 最坏（max_rounds=200 × 单遍） | numba 内核 O(200·n)；_rebuild O(n)（已修） |
| fold_and_relink | **O(m·前瞻集)** | 逐 swap 判定 × 影响集（含 a/b 的后续 2Q 门）——**未证上界**；hwb12 实测 15s |
| fold_keep_coords | O(n) | content 重放单遍 |
| chain_long_gates | O(n·链长) | 长程门链化：d 距离 → 4(d-1) 门 |
| merge_adjacent_rz | O(n)（已修） | 原 O(n²)（del 搬尾）——hwb12 爆炸根因 |
| merge_1q_runs | O(n) 扫描 + 游程矩阵乘 | 矩阵乘 O(游程²) 小常数 |
| _cancel_cz_global | O(n) | 单遍段划分 |
| commute_rz_through_cz | O(轮数·n) | max_rounds=1 |
| A2 collect（块收集） | **O(n·块候选)** | 未证上界；max_blocks 限制 |
| A2 fit_block_torch | O(块数·步数) | 步数上限 300+条件（已修） |
| decompose 循环（12 轮） | O(R·Σ组件) | 每轮 O(n) 级组件 ×12——hwb12 后处理 ~10 分钟实测 |
| relabel（VF2） | 指数最坏（图同构）+ 时间预算 | relabel_budget 25s 兜底 |

## 4. 已知超线性风险点（审计结果）

1. ~~merge_adjacent_rz O(n²)~~ 已修（O(n) 单遍）
2. ~~numba _rebuild 腐蚀~~ 已修（按源门 nq）
3. ~~评分路由 C×S 爆炸~~ 已修（巨型线路种子/候选缩减）
4. **fold_and_relink 的逐 swap 前瞻**：O(m·前瞻)——hwb12 15s 实测尚可，但 m 更大时未验证
5. **chain_long_gates 的链化膨胀**：fold_only 路径 hwb12 实测 111 万 cx——**超线性风险点**（链长 × 门数）
6. **A2 块收集**：未证上界
7. **循环 ×12 轮的全组件重复**：每轮 O(n) 组件 ×12 = 常数倍，可接受；但若某组件超线性则 ×12 放大

## 5. 待验证清单（实证）

- qft_25/50、qaoa_25/50、graphstate_25/50 的 timeout 栈转储定位
  （进行中：diag_qft25）
- hwb12 RL 覆盖后的端到端时间
- 51 万门级 build_program_graph 的 numpy 化收益
