# 相位多项式机制设计文档（CZ 域 1Q 优化）

状态：**数学侦察阶段**（2026-11-13）。先数学后实现，预注册判据。

## 0. 问题定位

1Q 差距（heisenberg N16 同口径）：我们 736 vs qiskit O3 489。
三层拆解（diag_gap_decomp / diag_a2_ablate / diag_block_density）：

| 层 | 数据 | 结论 |
|---|---|---|
| 后处理 rz 重排 | 理想跨段合并 389→363（3.5%） | rz 被 sx 锁死，否决 |
| A2 块合成 | A2 on 790→736（−54） | A2 是减 1Q 的，非膨胀源 |
| 不动点循环 | A2 off 产物 H 块 = 0 | **循环把 H 混入 u3 合成，H 身份丢失** |

qiskit 的 pass 顺序：CommutativeCancellation（H 对消）**先于**
Optimize1qGates（1Q 合成）。我们的不动点循环两者混在一起跑。

**攻坚目标：H 优先消解（H-first cancellation）**。

## 1. 恒等式基础（全部数值验证，scripts/verify_phase_poly_identities.py）

### A 组：平台基 {rz, sx, x, cz}
- A1: rz(α)·rz(β) = rz(α+β)（精确）
- A2: rz_q(θ) 与 cz(q,r) 对易（cz 对角）——rz 在 cz 域自由移动
- A3: sx·sx = x（平台定义精确）
- A4: x_q·cz(q,r) = cz·x_q·rz_r(π)·i（全局相位 i）
- A5: x·rz(θ) = rz(−θ)·x
- A7: sx·rz(θ)·sx = X·R_y(θ)（θ=π/2→H、θ=π→Z、θ=0→X）
- A8: rz(π/2)·sx·rz(π/2) = H（模全局相位）——平台 H 块
- A9: sx·rz(π)·sx = Z
- A11[反例]: rz 与 sx 不满足角度翻转——**rz 不能穿 sx**

### B 组：H 与 cz/cx 的交换
- B1: H_b·cz(a,b) = cx(a,b)·H_b（H 在第二比特侧穿 → CNOT）
- B1b: H_a·cz(a,b) = cx(b,a)·H_a（H 在第一比特侧穿 → 方向翻转）
- B3/B3b: H 共轭 cz → CNOT（方向随侧翻转）
- B4: H 与不相交 cz 对易（平凡）
- B6/B6b: cz·H = H·cx 镜像

**结论：H 穿 cz 破坏 cz 骨架（变 CNOT）——H 行走机制不成立（台账 12）**。

### P 组：CNOT 域（相位多项式文献域）
- P1: rz 与 CNOT 控制位对易
- P2: cx(a,b)·rz_b(θ) = RZZ_ab(θ)·cx(a,b)——rz 穿目标位 → 两比特
- P4: x_a·cx(a,b) = cx·x_a·x_b（X 穿控制位复制）
- P5: H_b·cx(a,b) = cz·H_b；P5b: H_b·cx·H_b = cz

### C 组：H 吸收
- C3: H·rz(θ)·H = rx(θ)（模全局相位）
- C4: H·rz(θ) = rx(θ)·H

## 2. 否决记录（数学先行成果）

| 机制 | 否决依据 | 台账 |
|---|---|---|
| 全局 rz 重排（cz 域相位多项式） | rz 不能穿 sx（A11）；理想合并上限 3.5% < 5% 判据 | #11 |
| H 行走（H 沿 cz 骨架复制） | B1：穿后骨架变 CNOT，无封闭行走 | #12 |
| 4 门 u3 模板 | 4 门族 2 参数 < 3 非通用 | #13 |
| A2 = 膨胀源假设 | A2 实测 −54 门 | #14（否证） |

## 3. 攻坚中：H 优先消解（预注册）

**机制假设**：qiskit 的 1Q 优势来自 pass 顺序——H 对消（H·H → I、
H·rz·H → rx 吸收）在 1Q 游程合成之前完成，H 身份不在合成中丢失；
我们的不动点循环 merge_1q_runs 把 H 块与邻 rz 合并成 u3 → 分解
5 门 ZSXXZ → H 结构永久丢失 → sx/rz 双膨胀。

**代码证据（2026-11-13 发现）**：decompose_to_platform_v2 的循环
顺序 = _sort_segment_1q → _merge_sx_pairs → slide_x → **merge_1q_runs**
（H 在此被合成 u3）→ … → _cancel_adjacent_h / _absorb_h（H 已不存在）。
而 v4 分解输出（cx → H-cz-H）天然富含：
- 相邻 H 对（共享目标比特的相邻 cx → H_b·cz·H_b·H_b·cz·H_b）
- H·rz·H 结构（cx 分解的 H 夹原始 rz）
→ 第一轮就被合成销毁。**修复 = 循环内重排序：H 消解先行、
1Q 合成后行**（qiskit O3 的 pass 顺序语义）。

**数学表述**（收益来源，恒等式全部已验证）：
1. 每个 H·H 对消 = 6 门（B5）。
2. 每个 H·rz·H → rx = 7 门 → 1 符号门（最后展开 4 门，C3）。
3. rx 池合并：rx(α)·rx(β) = rx(α+β)（X 轴同轴加法）。
4. rz 池合并：A1。
5. 收敛保证：H 消解/H 吸收每步净减 ≥2 门，门数单调不增
   （有势函数 = 总门数——与不动点无势函数理论的 9 条负结果
   不同，本机制每步严格减）。

**H 滑行规则（辅助，待验证恒等式）**：
- H·rz(θ) → rx(θ)·H（C4 ✓，门数不变）
- H·sx → rz(π/2)·H（模全局相位，门数 +1、可合并抵消）
- H 与不相交门对易（B4，免费）
- H 不穿 cz（B1：骨架破坏——段内消解，不跨段）

**预注册判据**：3 条线路（heisenberg N16、N25、qft N20 或同类）
bar 对比 +≥5% 门数减少，且语义 fid = 1.0（恒等式精确）。

**停止条件**：diag_h_first.py 模拟显示 H 优先消解收益 < 5%
→ 记账否决，换方向（上游模板改造）。

## 4. 参考文献域

- Nam et al. 2018（Automated optimization of large quantum circuits）：
  相位多项式 = CNOT 域的 {Rz, CNOT} 线性结构——我们的 cz 域对偶
  = rz 与 cz 对易（A2）+ X 穿 cz 留 rz(π)（A4）。
- qiskit O3 pass 顺序（transpiler/preset_passmanagers/level3.py）：
  CommutationAnalysis/CommutativeCancellation 先于 Optimize1qGates。
- Vatan-Williams 2012（3-CNOT 酉分解）：A2 块模板的理论基础
  （台账 9：中间模板无两侧因子不通用）。
