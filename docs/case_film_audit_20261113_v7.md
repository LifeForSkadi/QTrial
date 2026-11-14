# 案例演示片 v9 自审记录（2026-11-13）

## 交付物

- 评审草稿：`promo_video/out/qtrial-case-draft-v9.mp4`
- 规格：78 s（2340 帧 @30 fps）、1920×1080 设计、`--scale=0.5` 渲染为 960×540、无音轨
- Composition：`QTrialCase`（`promo_video/src/CaseFilm.tsx`）
- 旧成片 `QTrialDemo`、`决赛提交/`、PPT、报告书均未改动；本片尚未嵌入 PPT。

## 对用户 6 点意见的处理

1. **0–9 s 逻辑线路**
   - 换用 20 比特混合门负载：H / X / Y / Z / S / T / RX / RY / RZ + CX，共 45 个 CX、450 个 1Q 门。
   - 2Q 线在中间 1Q 门处断开，1Q 盒保持完整；长程 CX 红色虚线带光晕，不再被 1Q 门整段遮住。
   - 左侧比特标签加固定遮罩栏，滚动门不会覆盖 q0–q19。
   - 图例先出现（0.4 s），滚动 1.2 s 后才开始；右侧统计与红/蓝含义提前。
2. **9–16 s 交互图 + 拓扑**
   - 右侧布局重排：20/45/450 三个事实移到 QASM 代码块右侧；去掉“已知最优”。
   - 去掉目标芯片文本框，拓扑放大；新增右侧“105 比特 / 168 可用耦合器 / 4 禁用”竖排事实。
   - 交互图先全屏缩放落位，标题/代码/统计在落位后出现，避免文字被大图压住。
   - 场景从 12.7 s 缩短到 7 s。
3. **16–27 s 布局竞技**
   - 删除左侧文字刷新；先展示“不优化、直接平铺”的真机拓扑与 103 SWAP。
   - 13 个候选布局全部上墙（4 列 × 4 行），每个给出 SWAP / 深度 / 保真度。
   - 胜者 #3 高亮，左侧放大胜者布局；右下角对比卡：−47 SWAP、深度 163→86、保真 0.31→0.43。
4. **27–43 s 路由同步**
   - 以线路演化为导向：右侧大线路按深度层步进，左侧小拓扑同步搬 token。
   - 原有 1Q / CX 门真实出现并执行；遇到不相邻 CX 插入 SWAP 层，后续门在画面中整体后移。
   - 状态条逐层说明：当前层、事件（CX/SWAP/1Q）、物理位与逻辑位、进度、CX/SWAP 计数。
   - 拓扑使用上一页胜者布局；遇到空位交换时线路底部有“空位”行。
5. **43–57 s 后处理**
   - 左侧线路窗口放大到 1100×556；11 个可折叠 SWAP 逐个高亮、淡出、后续门左移。
   - 右侧 6 张 pass 卡：A1 `fold_and_relink`、A2 `collect_resynthesize_2q_blocks`（Weyl/VW）、A3 `commute_cancel_cx`、B `decompose_to_platform_v2` fixpoint、C1/L3 重标记、非退化门。
   - 底部流水线实时显示 路由 56 → A1 45 → A2 44 → 平台基 44；最终 2Q 深度 49、深度 127、保真 0.275。
6. **结果总结**
   - 删除 78–90 s 的 PPT 式总结页；真机数据表保留，后接 3 s 字标收尾。

## 时间线

| # | 帧 | 时间 | 场景 | 关键数据 |
|---|---|---|---|---|
| 01 | 0–270 | 0–9 s | 逻辑线路 | 20 比特；45 CX、450 1Q；38/45 个 CX 在平凡布局下不相邻 |
| 02 | 270–480 | 9–16 s | 交互图 + 拓扑 | 20/45/450；105 比特、168 可用耦合器 |
| 03 | 480–810 | 16–27 s | 布局竞技 | seed 7；13 候选；胜者 #3 56 SWAP vs 平铺 103 SWAP |
| 04 | 810–1290 | 27–43 s | 路由同步 | 56 SWAP、45 CX；86 层 |
| 05 | 1290–1710 | 43–57 s | 后处理 | 11 次折叠 56→45；A2 45→44；A3 对消 4 CX；最终 44 SWAP |
| 06 | 1710–1890 | 57–63 s | 最终线路 | 44 SWAP；2Q 深度 49；平台基深度 127；保真 0.275 |
| 07 | 1890–2250 | 63–75 s | 真机验证 | x5 0.6170、ghz5 0.5943、ghz10 0.2209、qft5 0.9539 |
| — | 2250–2340 | 75–78 s | 字标收尾 | QTrial |

## 数据来源

- 线路：`data/video_case/mixed20.qasm`
  - CX 交互骨架取自 `data/Queko/BIGD/20QBT_45CYC_.5D1_.1D2_0.qasm`（45 个 CX）。
  - 450 个 1Q 门替换为 H/X/Y/Z/S/T/RX/RY/RZ 混合门，仅用于门级可视化。
  - 38/45 个 CX 在平凡布局 0–19 下不相邻（按 `topo_real` 可用边判定）。v6 的 40 是按 |i−j|>1 估算；按真实耦合器判定为 38，本版以 38 为准。
- 生成脚本：
  - `promo_video/tools/dump_case_video_data_mixed.py` → `case_logic_mixed.json`
  - `promo_video/tools/dump_mixed_case.py` → `case_mixed.json`、`case_trace_mixed.json`
- 候选池：`promo_video/src/data/case.json` 中 RL 多起点 + 局部搜索为同一交互图生成的 13 个布局，在 mixed20 上重新路由实测打分。
- 胜者规则：保真度 ≥ 最优 98% 的候选中取 SWAP 最少者；胜者 #3。
- 后处理：对 56 个路由 SWAP 逐个做 `fold_and_relink` 的前瞻判定，11 个可吸收；应用 absorb 步后 45 个 SWAP；`collect_resynthesize_2q_blocks` → 44；`decompose_to_platform_v2`（含 A3/B）得到最终 44 SWAP、深度 127、2Q 深度 49、保真 0.275。
- 真机：`tables/real_machine_v8/summary.md`；31/31 为发布前合法性/语义审计，真机执行只有 4 条。

## 诚实性说明（必须保留在审计/报告中）

1. **mixed20 是演示负载，不是 QUEKO 原始线路**：CX 结构来自 QUEKO BIGD，1Q 层替换为混合门。视频中的映射/路由/后处理数字均由 QTrial 在该文件上实际运行得到，但不应把它当作 QUEKO 基准成绩引用。
2. **A1 是部分折叠**：当前生产 `fold_and_relink` 在存在“保留 SWAP”时会整体回退（P0 守卫）；本片按逐 SWAP 前瞻判定只应用 11 个可吸收步，这是语义等价的合法子集，但不是生产函数在該案例上的完整输出。
3. **C1/L3 未应用于最终指标**：片中作为候选评估展示，最终 44/127/0.275 来自 A1（部分）+ A2 + A3 + B 路径。若用户要求“完整生产 v5 路径”，需要重跑 C1/L3 并重新计算最终指标。
4. **真机只有 4 条**：表中 x5/ghz5/ghz10/qft5 为 tianyan176 真实执行；31/31 是发布前审计，不是 31 条真机执行。tianyan-287 机时不足。

## 自审检查

- **数据一致性**：S1 45/450/38；S3 56/103、深度 86/163、保真 0.428/0.307；S4 56 SWAP、45 CX、86 层；S5 56→45→44、A3 4 CX、最终 44/49/127/0.275；S6 44/49/127/0.275、门数 176/340/163/10；S7 四条真机数据——均与 `case_mixed.json`、`case_trace_mixed.json`、`tables/real_machine_v8/summary.md` 一致。
- **排版**：540p 逐场景静帧检查；修复 S1 标签遮挡、2Q/1Q 叠压、S2 标题被缩放图压住、S3 候选卡过小（改 4 列 + 两行指标）、S4 面板模糊入场、S5 卡片与底部流水线重叠、S6 右侧空白（改门数卡）。
- **时序**：S2 缩到 7 s；S4 480 帧（路由在 431 帧结束，留 1.6 s 完成态）；S5 420 帧（折叠在 242 帧结束，留最终流水线）；S8 总结删除。
- **可读性**：540p 下标题/状态条/关键数字可读；S5 卡片正文与 S6 门标签偏小，1080p 定稿会更清楚。

## 已知限制（下一轮候选）

- S4 第一层 1Q 门很密，540p 下线路像“门墙”；可考虑只显示与当前/未来 2Q 相关的行，或增加行聚焦。
- S5 A1 的 11 个可折叠 SWAP 多数在尾部/空位，线路窗口较稀疏；可加一个 A1 机制 inset（SWAP→置换）补足直观性。
- C1/L3 未真正应用；若需要完整生产路径，需重跑并接受最终指标可能变化。
- S2 交互图先全屏再落位，标题延迟到落位后出现；若用户希望标题从头就在，需要改回并接受短暂遮挡。
- 无音轨；未嵌入 PPT；未做 1080p 终渲与独立终检。

## 复现

```bash
cd promo_video
npx tsc --noEmit
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v9.mp4 --scale=0.5 --concurrency=4
npx remotion still src/index.ts QTrialCase out/case_audit/v8_check.png --frame=840 --scale=0.5
```

数据重新生成：

```bash
python promo_video/tools/dump_case_video_data_mixed.py
python promo_video/tools/dump_mixed_case.py
```

## 新增/修改文件

新增：
- `promo_video/tools/dump_mixed_case.py`
- `promo_video/tools/dump_case_video_data_mixed.py`
- `promo_video/src/components/TopoView.tsx`
- `promo_video/src/case/LayoutArena.tsx`
- `promo_video/src/data/case_mixed.json`
- `promo_video/src/data/case_trace_mixed.json`
- `promo_video/src/data/case_logic_mixed.json`
- `data/video_case/mixed20.qasm`
- `docs/case_film_audit_20261113_v7.md`（本文件）

重写：
- `promo_video/src/case/LogicCircuit.tsx`
- `promo_video/src/case/RouteSync.tsx`
- `promo_video/src/case/PassDetail.tsx`
- `promo_video/src/case/FinalCard.tsx`
- `promo_video/src/shots/S05_CaseCircuit.tsx`
- `promo_video/src/CaseFilm.tsx`

未修改：
- `promo_video/src/Demo.tsx`、`Promo.tsx`、`决赛提交/`、旧 PPT/报告书/视频。
