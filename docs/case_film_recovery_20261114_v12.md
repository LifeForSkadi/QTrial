# 案例演示片 v12 恢复点（2026-11-14）

> 当前状态：84.9 s / 540p 评审草稿 v12 已完成并自查；等待用户观看与下一轮意见。
> 审计：`docs/case_film_audit_20261114_v12.md`
> 文本：`docs/case_film_text_v12.md`

## 1. 当前产物

- `promo_video/out/qtrial-case-draft-v12.mp4`
  - 84.9 s（2546 帧 @30 fps）、960×540、无音轨、8.8 MB。
- Composition：`QTrialCase`（`promo_video/src/CaseFilm.tsx`），设计 1920×1080。
- 旧 `QTrialDemo`、`决赛提交/`、PPT、报告书未动；本片未嵌入 PPT。

## 2. 本版改动

- S1：跨距 CX 常驻光晕；96–112 帧蓝→红插值 + 抖动 + 发光脉冲。
- S2：误差色改蓝→淡紫；线宽/透明度恒定；节点不透明亮青 + 暗描边，压在连线之上。
- S3：场景 330→290 帧；候选池 46 帧入场、胜者 132 帧宣布。
- S4：动画未改；仅同步标题文字（补“逐”）。
- S5：
  - A1 修复 `q`→`lq` 行绑定；改为 13 层真实分层线路，折叠时后续层左移、已折叠 SWAP 移除；空位↔空位 SWAP 画在“空位”行并注明 5 个。
  - A3 加 P88/P95 的 1Q 邻居暗背景、层 1–8 列头。
  - B 加左右 1Q 邻居暗背景、合并前/后列头。
  - 全部删除 v5 字样，改“采用优化后线路 / fidelity(opt)”。
- S6：底栏说明放大；加 `SWAP → 3×CZ + 1Q`、`44 × 3 = 132 CZ`；4 个指标缩为 2×2。
- 全片文本按用户 v11 改稿更新；S4/S7 两处补字，S2 图例拆行。

## 3. 关键文件

- `promo_video/src/CaseFilm.tsx`：时间线；S3 时长 290。
- `promo_video/src/case/LogicCircuit.tsx`：S1 光晕/颜色插值/抖动。
- `promo_video/src/components/TopoView.tsx`：蓝紫误差色、`nodeStyle`、节点描边。
- `promo_video/src/case/LayoutArena.tsx`：S3 节奏与标题。
- `promo_video/src/case/RouteSync.tsx`：S4 标题。
- `promo_video/src/case/PassStudio.tsx`：A1 分层线路、A3/B 邻居、Guard 去 v5。
- `promo_video/src/case/FinalCard.tsx`：S6 底栏重构。
- `promo_video/src/case/RealMachine.tsx`：S7 副标题补“一”。

## 4. 数据

- 主案例：`data/video_case/mixed20.qasm`；45 CX + 450 1Q；13 候选；胜者 #3 56 SWAP vs 平铺 103。
- 后处理：A1 11 折叠 56→45；A2 45→44；A3 −2 CX；最终 44 SWAP / 2Q 49 / 深度 127 / 保真 0.275。
- A1：11 步中 5 步为 `lq=-1` 的空位↔空位。
- S6：44×3=132 CZ；A2 后非 SWAP 2Q 46，A3 对消 2，余 44；132+44=176。
- 真机：v9 7 条 v8 + 3 条 O3；4 条补跑 O3 仍排队。

## 5. 自查

- `npx tsc --noEmit` 通过。
- 成片 2546 帧 / 84.8667 s / 960×540 / 8.8 MB。
- 抽查 12 张成片帧，无溢出、重叠、黑帧、场景错位。
- 540p 下 A1 单门标签、S7 小字仍偏小；1080p 会改善。

## 6. 待用户确认

1. S1 的蓝→红抖动/光晕强度是否合适。
2. S2 蓝紫误差色与节点亮度是否满意。
3. S3 压缩后的节奏是否合适（胜者约 18.9 s 出现）。
4. A1 分层线路是否解决了“全堆在最底部”。
5. A3/B 的 1Q 邻居暗背景是否够看但不抢戏。
6. S6 底栏等式是否一眼能懂。
7. 文本建议：S3 “103 次”→ “103 个”；S3 路由打分句式；S6 “导致不同表示的产生”；S7 补“287 规格 / 176 真机”口径。

## 7. 复现

```bash
cd promo_video
npx tsc --noEmit
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v12.mp4 --scale=0.5 --concurrency=4
```
