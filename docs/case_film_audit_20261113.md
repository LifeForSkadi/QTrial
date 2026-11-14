# 案例演示片 v6 自审记录（2026-11-13）

## 交付物

- 草稿：`promo_video/out/qtrial-case-draft-v6.mp4`
- 规格：90 s、1920×1080 设计、当前以 `--scale=0.5` 渲染为 960×540 评审版、30 fps、无音轨
- Composition：`QTrialCase`（`promo_video/src/CaseFilm.tsx`），共 2700 帧
- 关键帧：`promo_video/out/case_audit/v6_*.png`、`v4_*.png`
- 旧成片 `QTrialDemo` 与 `决赛提交/` 未改动；本片尚未嵌入 PPT

## 场景表

| # | 时间 | 场景 | 数据源 |
|---|---|---|---|
| 01 | 0–8 s | 逻辑线路：40/45 个 CX 跨距 >1 | `case_logic.json`（QUEKO 20 比特） |
| 02 | 8–20.7 s | 交互图 + 目标芯片拓扑 | 复用 `S05_CaseCircuit`、`case.json`、`topo_real.json` |
| 03 | 20.7–32.3 s | 13 个候选布局、seed 7、路由实测 63 vs 120 | 复用 `S06_CaseLayout`、`case_trace.json` |
| 04 | 32.3–44 s | 拓扑 + 逻辑行线路同步：28 次逻辑交换 | 新 `RouteSync.tsx`、`case_trace.diagram_ops` |
| 05 | 44–58 s | 后处理：14 步折叠、63→49、VF2++ 重标记 0 | `PassDetail.tsx`、`TraceFilm`、`case_trace.json` |
| 06 | 58–66 s | 最终线路与指标：49 SWAP / depth 268 / 2Q 65 / 0.173 | `FinalCard.tsx`、`case.json` |
| 07 | 66–78 s | 真机 4 条线路：QFT5 0.954、GHZ10 0.221 | `tables/real_machine_v8/summary.md` |
| 08 | 78–90 s | 结果总结 + 收尾 | `Summary.tsx`、`S12_Outro` |

## 审计检查

- **数据一致性**：S1 的 45 CX/40 长程与 QASM 一致；S3 的 13 候选、63/120 与 `case_trace` 一致；S4 的 63 总交换、28 逻辑交换与 `diagram_ops` 一致；S5 的 14 折叠、49 保留、relink=0 与 `case_trace` 一致；S6 的 49/268/65/0.173 与 `case.json` 一致；S7 的 4 条真机数据与 `summary.md` 一致；S8 的 583/0.227/1.21/0.954 与 `tables/submission/summary.md` 一致。
- **知识准确性**：S1 讲“非相邻双比特门”，不讲“坏比特”；S4 区分物理交换与逻辑标签交换；S5 的 VF2++ 只作为候选，不虚构为必经步骤；S7 明确 31/31 是发布前审计、真机只执行 4 条。
- **转场**：S2–S8 开头加 14 帧章节卡，消除硬切到近黑帧；S1 片头同样有章节卡。
- **排版**：关键帧逐张检查；修复 S3 图例换行、S3 机制说明换行、S5 卡片 3 文案过长、S6 QCIS 片段与 P 行不一致、S7 审计徽章“真机合法”歧义、S4 相机包围盒未覆盖交换端点。
- **文字**：字幕 54–56 px；图内正文 ≥18 px（1080 设计）；未发现溢出/重叠（S3 候选列表与 #12 行仍偏挤，见下）。

## 已知限制（下一轮优先）

1. **VF2++ 未被案例触发**：`case_trace.relink=0`，S5 只能诚实展示“候选 0 次”；若必须看到 VF2 过程，需要换一条 VF2 胜出的线路，或补一条概念候选对比。
2. **A2 2Q 块重合成未单独可视化**：S5 卡片 3 只说明“按需”，没有逐块前后对比。
3. **S3 偏挤**：候选列表 13 行 + 三个代表卡 + 胜者/平铺对比，信息密度高；下一轮可缩成 Top-6 + “其余候选”计数。
4. **S5 胶片偏小**：14 步折叠的“并入置换”标签仅 12 px；下一轮可放大胶片或改为逐门高亮。
5. **S4 为 28 次逻辑交换的逐步展示**，63 次中含空位的交换快进；字幕已说明。
6. 当前为 540p 评审版；定稿需 1080p、独立终检、重嵌 PPT、更新 SHA/提交包。
7. 无音轨；若需要 BGM/音效，按 video-shotcraft 声音流程另做。

## 待用户评价

- 案例选 QUEKO-20（63 SWAP，轨迹丰富）而不是 QFT5（5 SWAP）：是否接受？真机段已用 QFT5/GHZ10 补正确性。
- 90 s 是否合适？若要压回 75 s，可砍 S3 候选列表或 S5 胶片。
- S3/S5 要“更简”还是“更细”？下一轮按反馈决定。

## 复现

```bash
cd promo_video
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v6.mp4 --scale=0.5 --concurrency=4
npx remotion still src/index.ts QTrialCase out/case_audit/v6_frame.png --frame=1050
```
