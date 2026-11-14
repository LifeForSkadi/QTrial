# 案例演示片 v11 上下文恢复点（2026-11-14）

> 当前状态：86.2 s / 540p 评审草稿 v11 已完成；等待用户评价。
> 主审计：`docs/case_film_audit_20261114_v11.md`
> 文本汇总：`docs/case_film_text_v11.md`

## 1. 当前产物

- `promo_video/out/qtrial-case-draft-v11.mp4`
  - 86.2 s（2586 帧 @30 fps）、960×540、无音轨。
- Composition：`QTrialCase`（`promo_video/src/CaseFilm.tsx`），设计 1920×1080。
- 旧 `QTrialDemo`、`决赛提交/`、PPT、报告书未动；本片未嵌入 PPT。

## 2. 时间线

| # | 场景 | 本地帧 | 全局约 | 时长 |
|---|---|---|---|---|
| 01 | 逻辑线路 | 0–270 | 0–9.0 s | 9.0 s |
| 02 | 交互图 + 拓扑 | 0–210 | 8.3–15.3 s | 7.0 s |
| 03 | 布局竞技 | 0–330 | 14.5–25.5 s | 11.0 s |
| 04 | 路由同步 | 0–480 | 24.8–40.8 s | 16.0 s |
| 05 | 后处理工作台 | 0–780 | 40.1–66.1 s | 26.0 s |
| 06 | 最终线路 | 0–220 | 65.3–72.7 s | 7.3 s |
| 07 | 真机验证 | 0–360 | 71.9–83.9 s | 12.0 s |
| — | 字标收尾 | 0–90 | 83.2–86.2 s | 3.0 s |

- 每页 22 帧 PPT 式上推转场；无全屏章节小标题。
- S5 六幕：A1 30–280、A2 280–410、A3 410–510、B 510–610、C1 610–700、非退化门 700–780。

## 3. 关键文件

- `promo_video/src/CaseFilm.tsx`：转场与时间线。
- `promo_video/src/case/LogicCircuit.tsx`：S1 混合门线路、跨距 CX 卡、静态→高亮→滚动。
- `promo_video/src/shots/S05_CaseCircuit.tsx`：S2 交互图/拓扑/误差色。
- `promo_video/src/case/LayoutArena.tsx`：S3 候选墙、胜者延迟标绿。
- `promo_video/src/case/RouteSync.tsx`：S4 线路主导路由、SWAP 聚焦、规则卡。
- `promo_video/src/case/PassStudio.tsx`：S5 六幕后处理工作台。
- `promo_video/src/case/FinalCard.tsx`：S6 两张压缩线路地图（44 SWAP vs 176 CZ）。
- `promo_video/src/case/RealMachine.tsx`：S7 v9 真机 7+3 表/对照。
- `promo_video/src/components/TopoView.tsx`：拓扑视图 + 误差色。
- `promo_video/src/data/case_mixed.json`、`case_trace_mixed.json`、`case_logic_mixed.json`、`real_machine_v9.json`。

## 4. 数据

- 线路：`data/video_case/mixed20.qasm`；45 CX + 450 混合 1Q；38/45 平凡布局下不相邻。
- 候选：13 个 RL 布局，mixed20 重新路由；胜者 #3 56 SWAP vs 平铺 103。
- 后处理：A1 11 次折叠 56→45；A2 45→44；A3 −2 CX；最终 44 SWAP / 2Q 49 / 深度 127 / 保真 0.275。
- 真机 v9：v8 x5 0.2540、ghz5 0.3576、ghz10 0.1428、ghz15 0.0406、qft5 0.9011、qft6 0.8370、qaoa12 0.2999；O3 qft5 0.6614、qft6 0.5982、qaoa12 0.3294。
- 31/31 为发布前审计；7 条 v8 + 3 条 O3 为真机执行；4 条补跑 O3 仍排队。

## 5. 诚实性口径

1. mixed20 是演示负载，不是 QUEKO 原始成绩。
2. A1 是部分折叠（11 个可吸收步）；生产函数存在保留 SWAP 时整体回退。
3. C1/L3 未应用于最终指标；B 是机制示意。
4. 真机 x5/ghz 低值来自 v9 新批次实测，未挑选。

## 6. 已知问题

- A1 尾部窗口偏稀疏；A3 画面较空；C1 是候选映射表；S6 是压缩地图。
- S5 卡片正文、S6 地图行号、S7 小字在 540p 偏小。
- 上推转场 0.73 s，较 PPT 快；若用户要更慢可把 T 调到 30。
- 无音轨、未嵌 PPT、未做 1080p 终渲与独立终检。

## 7. 复现

```bash
cd promo_video
npx tsc --noEmit
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v11.mp4 --scale=0.5 --concurrency=4
```

## 8. 待用户评价

- 上推转场是否合适；是否要更慢/改淡入淡出。
- S1 静态→跨距卡→红线→滚动的节奏。
- S2 误差色是否清楚。
- S3 胜者延迟标绿是否自然。
- S4 规则卡放大与 SWAP 聚焦。
- S5 六幕动画是否足够；A1/A3 画面偏空是否要加局部放大。
- S6 两张压缩地图是否解决“看不出 44 SWAP”。
- S7 7+3 数据布局。

## 9. 用户最新反馈（2026-11-14）

> “目前有挺多问题都需要更改的，boss你不满意的地方也很有道理，不过目前要先压缩上下文，压缩完了我再来具体说说我的看法。”

- 用户认为 v11 仍有**较多问题**，将逐条说明；在用户给出具体意见前**不要自行开工**。
- 压缩后先读本文件 + `docs/case_film_audit_20261114_v11.md` + `docs/case_film_text_v11.md`，再按用户意见改。
- 旧成片 `QTrialDemo`、`决赛提交/`、PPT、报告书仍未动。
