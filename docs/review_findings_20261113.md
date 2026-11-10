# QTrial pure 后处理管线独立审查报告（2026-11-13）

审查者：独立审查代理（实验研究员 + 竞赛评委双角色）
范围：`docs/post_pipeline_correctness.md` 的 A/B/C/D 框架、`qtrail/pure/{post,post_v2,synthesis,router,mapper}.py`、
`scripts/verify_e2e_production.py` 及 `scripts/` 下全部实验结论文件。
方法：全部结论由数值实验验证（`scripts/review_identities.py`、`scripts/review_router_semantics.py`），
矩阵口径 numpy 直算、模全局相位比较、atol=1e-9。

---

## 一、任务 1：B1 恒等式数值验证（最高优先级）

验证脚本：`scripts/review_identities.py`（31 项检查，全部通过）。逐项结果：

| 恒等式 | 位置 | 结果 | 偏差 |
|---|---|---|---|
| cx(a,b) = H_b·cz·H_b | post.py:270 | PASS（精确，无相位） | 9.7e-17 |
| H = rz(π/2)·sx·rz(π/2) 模全局相位 | post.py:20 | PASS | 1.7e-16；相位 **e^{-iπ/4}** 与文档一致 |
| 平台 cx 展开 = g·cx | post.py:270-278 | PASS | 2.5e-32；g = **-i**（全局相位，fid 口径不敏感） |
| swap = 3×cx | post.py:281 | PASS（精确） | 0.0 |
| sx·sx = X；sx⁴ = I | post_v2.py:581 | PASS（精确） | 0.0 |
| X_a·CZ = CZ·X_a·rz_b(π)·g（slide_x） | post_v2.py:629 | PASS | 6.1e-17；g = **+i**，与 docstring 一致 |
| H·rz(θ)·H = rx(θ) | post_v2.py:355 | PASS（20 随机 θ） | 2.5e-16 |
| 平台H·rz(θ)·平台H = g·rx(θ) | post_v2.py:355 | PASS；g = **-i**（平台 H 相位平方） | 2.4e-16 |
| rz(θ)·H = H·rx(θ)；H·rz(θ) = rx(θ)·H | post_v2.py:358 注释 | PASS | 1.9e-16 |
| rz(θ) 与 cz 对易（a/b 两比特） | post_v2.py:1228 | PASS（精确） | 0.0 |
| ZSXXZ 模板 = u3(θ,φ,λ) 模相位 | post.py:28 | PASS（随机 50 组 + 特殊点） | 8.6e-16 |
| ZSXXZ 在 θ=π（φ-λ≠0 mod π） | post.py:38 | PASS（此前担心的退化点，数学仍成立） | 2.7e-16 |
| 零角省略变体（strip 后 4 门/3 门） | post.py:59 | PASS | 4.3e-16 |
| θ≈0 对角特例 rz(φ+λ) | post.py:36 | PASS | 5.0e-11（θ=1e-10 时，误差 O(θ) 符合声明） |
| H 特例（3 门）/ X 特例（1 门） | post.py:31-35 | PASS | 1.7e-16 |
| _u3_to_rzsx_min 全部候选（网格 23 例含退化点） | post.py:43 | PASS | 最大 8.6e-16 |
| **已删除的 ZXZXZ 候选复核** | post.py:50-54 | 确认删除正确：Rz(φ+π)·X·Rz(θ+π)·X = Rz(φ-θ)（非对角=0，vs u3 偏差 0.78-0.93） | — |
| _mat_to_u3_params 往返（100 随机 SU(2) + 4 分支） | post_v2.py:203 | PASS | 2.4e-15 |
| merge_1q_runs 实现（30 随机游程） | post_v2.py:230 | PASS | 4.9e-16 |
| cx 同控/同靶对易、交叉不对易、cz 全对易、cz²=I | post_v2.py:1384 | PASS | 0.0 / 0.968(不对易,应) |
| _cancel_adjacent_cz / _cancel_cz_global / commute_rz_through_cz(_rev) / _merge_sx_pairs+_strip_identity / slide_1q_through / slide_x_through_cz / _sort_segment_1q / _preslide_1q / commute_cancel_cx（各 30 随机电路） | post_v2.py | PASS | 全部 ≤ 7.3e-16 |
| _absorb_h 实现（含 H·rz·H 模式随机电路） | post_v2.py:355 | PASS | 6.1e-16 |
| push_swaps（含 _swap_past 重写规则，30 随机电路） | post.py:122 | PASS | 0.0 |
| absorb_trailing + final_layout 一致性（compact 判据，30 随机） | post.py:161 | PASS | 0.0 |
| fold_only 语义（幽灵位随机电路，40 例，终态 Π 比较） | post_v2.py:1309 | PASS | 1.6e-16 |
| KMS 链 d=1..8 == cx(0,d)；方向翻转映射 | lnn.py:31 | PASS | 0.0 |
| magic 基性质 + _so4_to_su2 往返 | synthesis.py:140 | PASS | 虚部 2.4e-16；l/r 3.1e-16 |
| fit_block_torch + block_params_to_insts 往返（ov≥1-1e-7: 3/3） | synthesis.py:374 | PASS（残差 1.9e-6，B2 阈值内） | — |
| _to_cz_basis 往返 | post_v2.py:64 | PASS | 7.9e-16 |
| noise_aware_relabel 语义（U1 == Π·U0·Π†，20 随机） | post_v2.py:938 | PASS | 0.0 |
| B2 误差界反例 | 见任务 2 | 文档界被打破（因子差 √2） | — |

**B1 结论**：所有代数精确 pass 的恒等式均数值验证通过，未发现同类 ZXZXZ 错误。
特别说明（三处文档措辞与数学不完全一致，均无语义影响）：

1. **post_v2.py:69 注释 "H·ry(θ)·H = rz(θ)（到全局相位）" 为假**：实测 H·ry(θ)·H = ry(-θ)
   （对 rz 的偏差 = 1.0，对 ry(-θ) 的偏差 = 2.4e-16）。`_to_cz_basis` 的 flush() 不依赖该
   恒等式（矩阵合并后按对角性分支输出 rz/u3），通用分支安全——注释与"4 u3 + 4 rz"输出
   结构描述失真，但代码输出经往返验证正确（7.9e-16）。**P3**。
2. **post.py:59 `_strip_zero` 的判据 "rz 角度 ≡ 0 mod 2π"**：RZ 平台门周期是 **4π** 而非 2π
   （RZ(2π) = -I ≠ I）。strip 掉 RZ(2π) 引入的 -1 是全局相位，语义无害（本次验证的 strip
   变体全部模相位精确），但 docs B1 表 "_strip_identity | rz(θ≡0 mod 2π) = I" 的表述
   不准确（θ=2π 时应为 -I）。**P3**（纯文档）。
3. **post.py:23 docstring "U3 = RZ(λ)·SX·RZ(θ+π)·SX·RZ(φ+3π)"**：本审查对随机 50 组
   + θ=π/θ=0/θ=2π 退化点全部验证精确成立（含与 qiskit 2.5 OneQubitEulerDecomposer
   'ZSXX' 逐门对拍一致），无问题。**无需修改**。

---

## 二、任务 2：B2 数值守卫误差界审查

### 2.1 "ov ≥ 1-1e-7 → 酉偏差 ≤ 2√(1-ov)" 推导错误（P2，文档）

正确推导：d 维酉 U、V，ov = |⟨V,U⟩|/d，相位对齐后
‖V-U‖_F² = 2d(1 - Re⟨V,U⟩/d) ≤ 2d(1-ov)，即 **‖V-U‖_F ≤ √(2d)·√(1-ov)**。
d=4 时系数为 **2√2 ≈ 2.828**，文档的 2 少 √2 倍。

数值反例（review_identities.py T20）：取 SU(4) 的
U = diag(e^{iδ},e^{iδ},e^{iδ},e^{-3iδ})，δ = √(1e-7/1.5)：
ov = 0.999999900，‖U-I‖_F = **8.944e-04**；文档界 2√(1-ov) = 6.325e-04 **被打破**；
正确界 2√2√(1-ov) = 8.944e-04（达到紧界）。

对态保真度的影响：|⟨ψ|V†U|ψ⟩| ≥ 1-‖V-U‖₂ ≥ 1-2√2√(1-ov)，
单块 fid 偏差上限 ≈ 2·2√2·√(1e-7) ≈ **1.8e-3**（文档称 6.3e-4，同样差 √2×2）。
因此文档关键推论 "fid 出现 e-5 级偏离 = 存在恒等式错误" 的阈值应放宽到
**e-4~e-3 级**：单块在守卫边缘（ov=1-1e-7）合法替换即可产生 ~1.8e-3 的 fid 下降，
e-5 级偏离不能排除"守卫边缘的合法拟合"。（实践中拟合收敛到 ov≈1-1e-12，
见 2.2，故该推论在观测上未造成误判。）

### 2.2 fid=1.000000 观测与 B2 自洽性

实测数据（scripts/verify_e2e_production.txt，2026-11-11 04:04，修复后代码）：
h16 / h25 端到端 fid = **1.000000**（六位小数），两线路各含 ~20-30 个 A2 块替换。
自洽性推导：若块拟合均收敛到训练停止阈值 loss<1e-12（即 ov ≈ 1-3e-7 之上），
每块状态偏差 ~1e-6，全线路 fid 下降 < 1e-5 → 六位小数显示 1.000000 ✓。
本审查独立拟合 3 个随机 SU(4) 块（T19）：ov 全部 ≥ 1-1e-7，重建偏差最大 1.85e-6，
与"每块 ~1e-6 偏差"一致。Tuccil 类运行的观测（verify_final_min3.txt 0.999999、
exp_tuccil_lock_results 0.999998）对应 1e-6~2e-6 级 fid 下降，同样落在该口径内，
**不构成恒等式错误的证据**（docs 的 e-5 探测器阈值下这些数据会误报）。

### 2.3 fit_block_torch 的 ov 归一化口径（正确）

- `fit_block_torch`（synthesis.py:438）：`ov = |vdot(C,U)| / (‖C‖·‖U‖)`，C 为 4×4 酉
  → ‖C‖=‖U‖=2 → **ov = |⟨C,U⟩|/4**，归一化正确（精确匹配时 =1）。
- 训练损失（synthesis.py:421）：`(ov.abs()/4)²`，其中 ov 为展平 vdot →
  |⟨C,U⟩|²/16 ✓ 与返回口径一致。
- `fit_block_cz` / `fit_block_tuccil` 返回 `|vdot|/4` ✓ 同口径。
- 恒等块删除守卫 `|tr(U)|/4 > 1-1e-7`（post_v2.py:759）✓ 同口径。
- 不一致点（P3）：`fit_block_torch` docstring 称 "收敛验证 > 1-1e-9"，
  实际调用方守卫是 `ov < 1-1e-7` 拒绝（post_v2.py:777），docstring 过时。

---

## 三、任务 3：紧凑子空间验证方法论逻辑审查

对象：`scripts/verify_e2e_production.py`（生产全路径权威验证）。

### 3.1 数学前提检查

- 触碰集收集：`qubits = {final_qc.ops 全部比特}` + `all_q = qubits ∪ {fin[i]}` ✓ 完整
  （含 final_layout 值域，含链化中间位——中间位必在 ops 触碰集内）。
- "其余比特恒等 |0⟩"：C'（产物）只触碰 qubits ✓ 由构造成立；原线路经 fin 嵌入只触碰
  fin 值域 ⊆ all_q ✓。压缩映射 cmap 对两侧同构应用 ✓。
- **判据方向（重要）**：脚本实际比较的是 `U(C')|0⟩ == U(C 经 final_layout 嵌入)|0⟩`。
  这是**交付语义判据**（在真机执行 C'、按 final_layout 读回 = 原线路从 |0⟩ 的读回分布），
  与初始布局 π **完全无关**。本审查用真实路由器做了数值验证（review_router_semantics.py
  A 组，3 条随机 4 比特线路 × 非平凡初始布局，tianyan287 真实 spec）：
  fin 嵌入 fid = **1.000000 / 1.000000 / 1.000000**（最大缺口 4.4e-16），
  而 **π（初始布局）嵌入 fid = 0.000000 / 0.000000 / 0.250000**（3/3 置换非平凡，≠1）。
- **docs/post_pipeline_correctness.md §A 的 I-语义表述错误（P2，文档）**：
  "U(C')|0…0⟩ = U(C)|0…0⟩（C 嵌入初始布局 π）" 对任何发生净置换的路由都不成立
  （正确关系是 U(C')|0⟩ = Π·U(C_π)|0⟩，Π 为 π→fin 的坐标置换）。`verify_e2e_production.py`
  的 docstring ① 同样写着"嵌入初始布局"，但实现用的是 final_layout——**实现是对的，
  两处文档是错的**。建议将 I-语义改写为交付判据（fin 嵌入），并说明初始布局只是
  路由器内部选择、对交付正确性不可见。

### 3.2 可能漏报语义错误的场景

1. **measure 完全不校验（P2，验证盲区仍在）**：`to_qc` 只翻译 ops、忽略 `final_qc.measures`；
   `check_legality` 不查 measure 的位置/禁用位/越界。若管线任一步骤使 measure 标签与
   final_layout 脱节（原始代码确实存在该类缺陷，见 D 组实测；工作区已修三组件），
   fid=1.000000 照样通过。benchpress 基准无 measure → 对当前 587 交付集合无实际影响；
   建议提交 P1-1 修复时同步在验证脚本补 measure 检查。
2. **只验证 |0⟩ 输入**：态矢量相等仅在 |0…0⟩ 上检验，不等价于酉相等。benchpress 评分
   从 |0⟩ 出发 → 判据恰为所需；若评委口径含任意输入或 process fidelity 则不足。P3。
3. **final_layout 双射性不显式检查**（`fin.get(i, i)` 兜底）：若 fin 缺键/碰撞，
   压缩两侧错位通常被 fid≠1 抓住，但建议显式断言 fin 是 {0..n-1} 上的置换。P3。
4. **同向整体置换盲区（非漏洞，说明）**：若某 bug 同时整体置换产物电路标签与
   final_layout（同向），压缩比较仍=1——但该情形的交付读回语义（电路+读回映射同步）
   不变，不是真实错误。

### 3.3 与路由器空位穿越的独立性

压缩判据只依赖 (产物, final_layout) 对，不依赖路由过程；空位穿越（逻辑比特经空物理位
搬移）产生的中间位置都被触碰集自然覆盖（见任务 4 实测：fold 在幽灵位随机电路上
语义精确 1.6e-16）。

---

## 四、任务 4：路由器空位穿越语义审查

### 4.1 空位穿越 × fold 自洽性（PASS）

review_router_semantics.py / review_identities.py T16：随机构造"逻辑比特经空物理位搬移"
的 40 条路由形态电路（n_logic=3 嵌入 n_phys=6，swap 可落在任意空位），fold_only 的
终态不变量 `U(folded)|0⟩ = Π_final·U(routed)|0⟩` 最大偏差 **1.57e-16**——arr 坐标跟踪
对幽灵位严格自洽，无幽灵标签污染（arr 值恒在 0..spec.n-1 内，只是坐标置换）。

### 4.2 防卡死逃逸路径（router.py:451-524）语义

- **回卷逻辑**：对 `reversed(batch)` 逐条 pop + pos/inv 复原——本审查用与源码逐字相同的
  逻辑做了 50 轮随机模拟（B 组实验，含空位参与 swap 的边界情形）：
  回卷后 pos/inv 与推进前**逐位一致** PASS。
- **BFS 推进**：moves = 左半 [(p_i,p_{i+1})] + 右半 [(p_e,p_{e-1})]（e 从尾倒序），
  共 L-1 条边、中间边 (p_split, p_split+1) 恰好空缺——两端内容终位
  (p_split, p_split+1) **相邻** ✓（B 组模拟验证：两端标签终位索引差恒为 1，PASS；
  偶数 L 时两端标签在终点对上调换位置，不影响"相邻"目标）。推进产生的 swap 端点
  全为路径相邻对 ✓；中间位若被其他逻辑比特占据，pos/inv 同步更新 ✓。
- 未发现语义缺陷（P3 备注：回卷只弹 swap 不回滚已执行门——语义上无需回滚 ✓；
  `since_exec` 重置 ✓）。

### 4.3 "fold_and_relink 保留 swap 端点可能非相邻"（P2 → 已修）

代码事实（原 post_v2.py:1174-1181）：逐 swap 判定**不吸收**时，保留的 swap 用
`(arr[a], arr[b])` 重标记（fold 坐标），该对是"两个内容的原始坐标"——**没有相邻性
保证**（耦合图相邻性在坐标置换下不保持）。后续 `chain_long_gates` 跳过 swap、
`decompose_to_platform` 会把 swap 展开成 cz(a',b')——若 (a',b') 非相邻，产物 I-合法
被破坏，且 **mapper 生产路径没有任何最终合法性检查**（合法性只有外部验证脚本做）。

实测触发情况（C 组）：h16 身份布局与随机布局两条真实路由产物上，
`fold_and_relink` 均返回 arr=None（**未采用**——折叠+链化不劣化门槛挡住），
"保留 swap"分支不可达；外部合法性扫描（714 条）也未捕获任何非法产物——
理论风险、无实测危害。

**修复（工作区，2026-11-13，未提交）**：保留 swap 前检查 `spec.adj[qa, qb]`，
非相邻则强制吸收（arr 同步、不输出 swap）——fold 吸收是代数恒等，强制吸收
语义精确；下游 2Q 分支经 noise_aware_chain 链化保证合法（无路径回退原线路）；
成本前瞻被放弃的罕见情形由 fold 整体非劣化开关兜底。修复形态正确。

### 4.4 重标记组件（C1/L3）

noise_aware_relabel：移动仅针对"无 2Q 门孤立位"之外的已用位 → 空位，且每步合法性检查
（新端点必须相邻）✓；语义 = 标签置换共轭（T21 实测 Π·U0·Π† 偏差 0.0）✓；
final_layout 同步 ✓。learned_relabel：legal(π) 检查全部 2Q 门 ✓；apply_pi 同步
final_layout ✓。**measures 同步：原始代码缺失（P1-1），工作区已修（2026-11-13），
静态与 D1 行为双重确认**。

---

## 五、任务 5：实验结论审查（评委角色）

### 5.1 时间线核查（修复 vs 实验）

git 定位：**"外比特 1Q 硬边界"修复 = c881af8（2026-11-11 01:57）**；ZXZXZ 删除 =
post.py 05:30（文件未提交）。scripts/ 下实验文件全部生成于 **02:06-04:04**，
即全部在语义修复之后 → **"实验结论被误吞 bug 污染"的怀疑排除**（此前的担心不成立，
但注意 bench_h_family_v7_final.txt 03:01 与 noslide.txt 03:24 之间 synthesis.py 有
一次 03:17 的 mtime 变化、内容无 diff，两臂代码内容一致）。

### 5.2 "Tuccil 零收益"结论的支撑证据（P2，实验设计缺陷）

- **ablate_tuccil_e2e.py / ablate_tuccil_multi.py 的消融从未消融**：两脚本都调用
  `decompose_to_platform_v2(routed, spec=spec, resynth_every=1)`，而 Tuccil 只在
  `final_tuccil=True` 时才启用（post_v2.py:1525；默认 False，生产 `_v5_pipeline`
  调用点 mapper.py:260 也未传）。(a) 臂的 monkeypatch `fit_block_tuccil` 打在一个
  **永远不会被调用的死路径**上——(a)/(b) 两臂是同一配置的两次数值噪声样本
  （h25 臂间 1Q 差 2、cz 相同即此来源）。
- **真正打开 Tuccil 的实验只有 verify_final_min3.txt（03:19，修复后，n=1 线路）**：
  (E) 终min3+Tuccil：1Q=1397 cz=197 est_fid=8.75e-5 vs (A) 基线 1Q=1294 cz=200
  est_fid=1.37e-4——Tuccil 省 3 cz 但 +103 1Q，est_fid 恶化 36%。方向支持"负收益"，
  但样本量 n=1。
- 结论修订建议：把"Tuccil 零收益"的引证从 ablate_tuccil_e2e/multi 改为
  verify_final_min3，或在 ablate 脚本补 `final_tuccil=True` 重跑。

### 5.3 "re=1 最优"（P3，样本量）

exp_resynth_sweep.txt（02:11，修复后，单线路 h16）：re=1→6 门数单调变差
（1500/1602/1626/1621/1670/1824），re=1 占优方向明确；生产配置（mapper.py:260
resynth_every=1）另有 verify_re1_prod.txt（语义 fid=1.000000）与 verify_e2e_production
（fid=1.000000、twoq 与 qiskit O3 同口径）背书。缺陷：单一线路、无重复方差。
结论可用，但建议表述为"h16 上 re=1 占优"。

### 5.4 "preslide 有害"（P3，可接受但有保留）

- 比较对：bench_h_family_v7_final.txt（03:01，preslide 开）vs
  bench_h_family_v7_noslide.txt（03:24，monkeypatch 关闭）——同代码、同种子、单因子，
  三线路方向一致（比值 0.810/0.710/0.766 → 0.821/0.713/0.775）。
- 保留点：(1) 每配置单次运行，无方差估计；对照 bench_h_family_v7_new.txt（02:43，
  同配置另一轮）比值 0.807/0.703/0.749——轮间噪声 ~0.5-1%，与组间差 1-1.5% 同量级，
  结论稳健性一般；(2) "保真度"是 est_fid（校准误差积），不是语义 fid。
- 生产侧佐证：preslide 移除后 verify_e2e_production（04:04）h16/h25 语义
  fid=1.000000、N16 twoq=72 与 noslide.txt 完全一致 → 移除无回归。结论可保留，
  建议补一次带方差的复跑。

### 5.5 其他结论核验

- "min3 最终再合成无增益"（verify_final_min3：E 比 A 差）✓ 支持。
- "gate_guard 不利"（bench_h_family_v7_guard：0.789/0.669/0.723 全面差于
  final/noslide）→ 生产默认 gate_guard=False ✓ 支持（3 线路一致）。
- "通勤轮数/循环上限无增益"（exp_commute_rounds、exp_maxrounds：12=24=36 轮全同）✓。
- "587 全合法"：tables/final_report*/legality_rows_*.jsonl 合计 722 行、0 非法、
  **8 行 timeout 未验证**——表述应为"已验证子集（714 条）全合法 + 8 条超时未覆盖"。
  且该扫描是外部验证脚本，生产 map_circuit 内部无合法性自检（见 4.3）。P3。
- exp_tuccil_lock_results 的 "lock=1 re=3 语义 fid=0.999998"：2e-6 级偏离落在
  B2 拟合残差口径内（见 2.2），非恒等式错误；lock 机制对输出零影响
  （lock=0/1 产物全同）——该保护机制当前是无效保护（被保护的场景未出现）。P3。

### 5.6 外比特 1Q 误吞 bug 的污染排查结论

修复（c881af8，01:57）早于全部实验文件（02:06 起）→ **无污染**。
唯一例外：post.py 的 ZXZXZ 删除（05:30）晚于部分实验——但该候选自平局规则以来
从未被选中（post.py:50-54 注释，本审查 T10 复核其确为对角坍缩），
对实验结论无影响。

---

## 六、发现汇总（按严重级别）

### P0（语义错误）：未发现

所有 B1 恒等式、pass 实现、路由器契约、fold 语义均数值验证通过。
（ZXZXZ 候选删除正确；compact 判据实现正确。）

### P1（已发现，且审查期间已在工作区修复——待提交与回归验证）

1. **measure 标签与 final_layout 脱节（读回错误，实测复现 → 已修）**：
   - 原始缺陷：absorb_trailing（post.py:161）吸收尾部 swap 时更新 final_layout
     但不动 `circ.measures`；noise_aware_relabel（post_v2.py:995）/
     learned_relabel（post_v2.py:1066）重标记 ops 与 final_layout 但不动
     measures——路由器在路由末端位置发的 measure 停留在吸收前位置，按 fin⁻¹
     落到别的逻辑比特槽位（D1 实测：路由末端 [x(0), swap(0,1)]、measure 在位置 1
     （logical 0 的测量），吸收后 fin={0:0,1:1}——logical 0 的 cbit 读走位置 1
     的内容，logical 0 的 x|0⟩=|1⟩ 输出丢失）。
   - 修复（工作区，2026-11-13，未提交）：post.py absorb_trailing 增加 measure
     随 a/b 交换同步（实测：修复后 measure 正确跟随 1→0，D1 判据转一致）；
     post_v2.py noise_aware_relabel（p→p_new 时 measure 同步）与
     learned_relabel apply_pi（measure 随 π 重标记）同步修复。
   - 遗留：verify_e2e_production.py 的 to_qc / check_legality 仍完全忽略
     measure（验证盲区未消除）——建议补 measure 位置/禁用位/越界检查后提交。
   - 影响面（修复前）：benchpress 587 条无 measure → 无实际影响；
     含 measure 的电路（若竞赛题含测量指令且评审按线路内 measure 读回）读回错误。
2. **fold_and_relink 保留 swap 端点相邻性无保证（潜在硬件非法 → 已修）**：
   - 原始缺陷：post_v2.py:1174-1181"不吸收"分支保留的 swap 以 fold 坐标重标记，
     无相邻性检查；后续分解为 cz 后可能非法；生产无最终合法性自检。
     实测（C 组）：h16 两个布局下 fold_and_relink 均未采用（arr=None），分支
     不可达——理论风险，无实测危害。
   - 修复（工作区，2026-11-13，未提交）：保留 swap 前检查 `spec.adj[qa, qb]`；
     非相邻则强制吸收（arr 同步、不输出 swap）。语义精确性：fold 吸收本就是
     代数恒等（此前保留仅为成本考量），强制吸收语义仍精确；合法性：下游 2Q
     分支经 noise_aware_chain 链化保证相邻（无路径时回退原线路）。
     代价备注（P3）：强制吸收放弃成本前瞻，罕见情形下链化门数可能膨胀——
     非劣化开关（fold 产物 n_2q ≥ orig_cost 则整体回退）仍兜底。
   - 遗留：提交 + 触发该分支的回归用例。

### P2（方法论/文档）

3. docs §A 与 verify_e2e_production.py docstring 的 I-语义表述错误（初始布局嵌入应为
   final_layout 嵌入；正确关系含 Π 置换）——文档口径与实际判据矛盾。
   （数值证据：A 组 3/3 线路 π 嵌入 fid=0.00/0.00/0.25，fin 嵌入 fid=1.000000。）
4. docs §B2 误差界因子差 √2（2√2 正确，8.94e-4 vs 6.32e-4 数值反例）；"e-5 级偏离
   =恒等式错误"推论阈值应放宽至 e-4~e-3。
5. Tuccil 消融实验未消融（final_tuccil 未开启，两臂同配置）；结论改引
   verify_final_min3（n=1）或重跑。
6. fold_and_relink 保留 swap 无相邻性守卫 + 生产无内部合法性自检（理论风险，
   实测未触发，见 4.3）。

### P3（建议）

7. post_v2.py:69 注释 "H·ry(θ)·H = rz(θ)" 为假（实为 ry(-θ)）；docs B1
   "rz(θ≡0 mod 2π)=I" 表述不准确（RZ(2π)=-I，全局相位无害）。
8. fit_block_torch docstring 守卫阈值 1e-9 vs 实际 1e-7 不一致。
9. KAK 收敛慢（iters=20 往返偏差可达 0.16；生产仅作初值 + torch 精修 + ov 守卫，
   语义安全）；docstring "往返测试锁定"措辞过强。
10. 紧凑验证不查 final_layout 双射性；只验 |0⟩ 输入；measure 盲区（并入 P1-1）。
11. 实验结论样本量小（sweep/消融多为单线路、单次运行）；合法性扫描 8 条 timeout
    未覆盖。
12. slide_1q_through 允许 1Q 门穿越 barrier（调度语义，非酉语义）。

---

## 七、真机可实现性声明可信度评估

**结论：当前交付口径（benchpress、无 measure、|0⟩ 输入）下，"产物可在真机执行"
的声明可信，但有条件。**

- **I-语义**：≤26 比特权威验证（紧凑子空间法）实现正确、判据 = 交付语义 ✓
  （A 组实测 fin 嵌入 fid=1.000000、π 嵌入 fid=0 的对照证明判据方向正确）；
  实测 h16/h25 fid=1.000000 ✓；更大规模的保证依赖 B1 代数精确性——本审查逐一数值
  验证了 B1 全部恒等式（31 项 0 失败），该链条目前站得住。
- **I-合法**：714 条已扫描全合法 + 8 条超时未覆盖；生产内部无自检；
  fold 保留 swap 非相邻缺陷已在工作区修复（P1-2）。
- **盲区**：measure 语义缺陷（P1-1）已在工作区修复三组件，但**验证脚本仍不
  检查 measure**（to_qc/check_legality 忽略 measure）——修复提交前应补上；
  |0⟩-输入之外的等价性未被任何验证覆盖——若竞赛口径含测量指令或以酉等价
  评分，现有证据不覆盖，需补验证。
- 实验结论（Tuccil/re=1/preslide）方向均有支持，但证据强度为"单线路/少样本"级
  （且 Tuccil 消融存在实验设计缺陷），作为默认参数决策可用、作为对外声明需
  降格表述。

## 八、必须修复清单（按优先级；标注状态）

1. [P1-1，**已修待提交**] measure 全链路重标记一致性：工作区已修
   absorb_trailing / noise_aware_relabel / learned_relabel 三处。
   **提交前补**：verify_e2e_production.py / verify_all_legality.py 补
   measure 位置/禁用位/越界检查（当前验证盲区仍在）。
2. [P1-2，**已修待提交**] fold_and_relink 保留 swap 相邻性守卫：工作区已修
   （非相邻强制吸收）。提交 + 补触发分支的回归用例；生产可选内部合法性自检。
3. [P2] 修正 docs §A I-语义表述与 verify_e2e_production.py docstring（final_layout
   口径 + Π 置换说明）。
4. [P2] 修正 docs §B2 误差界因子（2√2）与探测器阈值（e-4~e-3）。
5. [P2] 重跑 Tuccil 消融（final_tuccil=True）或删除无效引证。
6. [P3] 修正 _to_cz_basis 注释、_strip 判据注释、fit_block_torch docstring、
   KAK docstring；合法性扫描超时行补覆盖。

---

## 附录：验证脚本与关键输出

- `scripts/review_identities.py`：B1 全部恒等式 + pass 实现随机电路测试（31 项检查，
  0 失败）；输出 `scripts/review_identities.out`。**在修复后工作区（含
  synthesis.py 的 3cx 高斯-牛顿替换）重跑仍 31/31 通过**（T19 重建偏差 1.03e-6，
  ov 3/3 ≥ 1-1e-7）。
- `scripts/review_router_semantics.py`：路由器契约/嵌入方向/逃逸路径/fold 相邻性/
  measure 一致性实验；输出 `scripts/review_router.out`。关键输出摘录：

```
== A. 路由器契约与嵌入方向 ==
  trial 0: fin嵌入 fid=1.000000 | π嵌入 fid=0.000000 | 置换非平凡=True | swaps=7
  trial 1: fin嵌入 fid=1.000000 | π嵌入 fid=0.000000 | 置换非平凡=True | swaps=10
  trial 2: fin嵌入 fid=1.000000 | π嵌入 fid=0.250000 | 置换非平凡=True | swaps=12
  结论：fin 嵌入 fid 最大缺口=4.441e-16；π 嵌入在置换非平凡时 fid≠1（3/3）→
  docs A 的 I-语义表述（初始布局嵌入）不成立，紧凑子空间法测的是交付判据
== B. 防卡死逃逸（回卷 + BFS 推进）语义 ==
  回卷恢复 pos/inv + BFS moves 语义一致: PASS
== C. fold_and_relink 语义 + 保留 swap 相邻性 ==
  h16 身份布局路由: 1076 门 swap=52
  fold_and_relink 未采用（无收益），跳过（保留 swap 分支不可达）
== D. measure 标签一致性（修复后） ==
  D1 吸收后: fin={0:0,1:1,2:2,3:3}；measure 原属 logical 0（路由末端位置 1），
     现位置 0 按 fin^-1 属 logical 0 → 修复后行为：measure 跟随 content ✓ 一致
```

补充实验：
- B 组（50 轮回卷模拟 + BFS 推进断言）PASS：回卷恢复 pos/inv 逐位一致；
  两端标签终位相邻（偶数 L 时两端在终点对上调换位置，不影响"相邻"目标）。
- C 组补充：h16 随机布局路由（1073 门 swap=49）fold_and_relink 仍不采用。
- 合法性扫描统计：tables/final_report*/legality_rows_*.jsonl 共 722 行、
  0 非法、8 行 timeout 未验证。
- 修复核查：工作区 P1-1（absorb_trailing/noise_aware_relabel/learned_relabel
  measure 同步）与 P1-2（fold 保留 swap 相邻性守卫）均已落地；D1 实测
  measure 跟随正确；T19 实测 3cx 高斯-牛顿替换语义不变。
