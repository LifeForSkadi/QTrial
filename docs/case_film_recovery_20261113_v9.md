# 案例演示片 v9 上下文恢复点（2026-11-13）

> 当前状态：78 s / 540p 评审草稿 v9 已完成；等待用户评价。
> 主审计：`docs/case_film_audit_20261113_v7.md`
> 旧 v6 恢复点：`docs/case_film_recovery_20261113.md`（历史，勿再按它开工）

## 1. 当前产物

- `promo_video/out/qtrial-case-draft-v9.mp4`
  - 78 s（2340 帧 @30 fps）、960×540、无音轨。
- Composition：`QTrialCase`（`promo_video/src/CaseFilm.tsx`），设计尺寸 1920×1080。
- 旧 `QTrialDemo`、`决赛提交/`、PPT、报告书均未动；本片未嵌入 PPT。

## 2. 时间线

| # | 帧 | 时间 | 场景 | 关键数据 |
|---|---|---|---|---|
| 01 | 0–270 | 0–9 s | 逻辑线路 | 20 比特；45 CX、450 1Q；38/45 个 CX 在平凡布局下不相邻 |
| 02 | 270–480 | 9–16 s | 交互图 + 拓扑 | 20/45/450；105 比特、168 可用耦合器 |
| 03 | 480–810 | 16–27 s | 布局竞技 | 13 候选；胜者 #3 56 SWAP vs 平铺 103 SWAP |
| 04 | 810–1290 | 27–43 s | 路由同步 | 56 SWAP、45 CX；86 层 |
| 05 | 1290–1710 | 43–57 s | 后处理 | 11 折叠 56→45；A2 45→44；A3 对消 4 CX；最终 44 SWAP |
| 06 | 1710–1890 | 57–63 s | 最终线路 | 44 SWAP；2Q 深度 49；平台基深度 127；保真 0.275 |
| 07 | 1890–2250 | 63–75 s | 真机验证 | x5 0.6170、ghz5 0.5943、ghz10 0.2209、qft5 0.9539 |
| — | 2250–2340 | 75–78 s | 字标收尾 | QTrial |

## 3. 数据与生成

- 线路：`data/video_case/mixed20.qasm`
  - CX 骨架来自 QUEKO BIGD `20QBT_45CYC_.5D1_.1D2_0`（45 CX）；1Q 层替换为 H/X/Y/Z/S/T/RX/RY/RZ 混合门（450 个）。
  - 38/45 个 CX 在平凡布局 0–19 下不相邻（按 `topo_real` 可用边判定）。
- 生成：
  - `python promo_video/tools/dump_case_video_data_mixed.py` → `case_logic_mixed.json`
  - `python promo_video/tools/dump_mixed_case.py` → `case_mixed.json`、`case_trace_mixed.json`
- 候选：`case.json` 的 13 个 RL 布局，在 mixed20 上重新路由实测；胜者 #3。
- 指标：路由 56 SWAP / 深度 86 / 保真 0.428；A1 后 45；A2 后 44；平台基 44 SWAP / 深度 127 / 2Q 深度 49 / 保真 0.275。
- 真机：`tables/real_machine_v8/summary.md`；31/31 为发布前审计，真机仅 4 条。

## 4. 关键实现

- `promo_video/src/components/TopoView.tsx`：可复用拓扑视图（全芯片 / 候选缩略 / 路由 token），屏幕像素视觉参数。
- `promo_video/src/case/LogicCircuit.tsx`：混合门线路、2Q 线在 1Q 盒处断开、固定标签栏、图例先行。
- `promo_video/src/case/LayoutArena.tsx`：平铺 103 SWAP → 13 候选墙（4 列）→ 胜者 #3。
- `promo_video/src/case/RouteSync.tsx`：按 86 个深度层步进，8 列线路窗口，后续门后移，拓扑同步，状态条。
- `promo_video/src/case/PassDetail.tsx`：1100×556 线路窗口，11 个折叠步，6 张 pass 卡，底部流水线。
- `promo_video/src/case/FinalCard.tsx`：平台基门序窗口、44/49/127/0.275、门数统计。
- `promo_video/src/shots/S05_CaseCircuit.tsx`：交互图先全屏落位，再出标题/QASM/统计/大拓扑。
- `promo_video/src/CaseFilm.tsx`：7 场景 + 收尾，2340 帧。

## 5. 诚实性口径（见审计第 3 节）

1. mixed20 是演示负载，不是 QUEKO 原始线路成绩。
2. A1 是逐 SWAP 判定后的**部分折叠**（11 个可吸收步）；生产 `fold_and_relink` 在存在保留 SWAP 时会整体回退。
3. C1/L3 未应用于最终指标；最终 44/127/0.275 来自 A1（部分）+ A2 + A3 + B。
4. 真机仅 4 条；31/31 是发布前审计。

## 6. 已知问题

- S4 第一层 1Q 门很密，540p 下线路像“门墙”；可考虑行聚焦。
- S5 A1 的折叠 SWAP 多数在尾部/空位，线路窗口偏稀疏；可加 A1 机制 inset。
- S5 卡片正文、S6 门标签在 540p 偏小；1080p 定稿会好转。
- S2 交互图先全屏再落位，标题延迟出现；若用户要标题全程在，需要改。
- 无音轨、未嵌 PPT、未做 1080p 终渲与独立终检。

## 7. 复现

```bash
cd promo_video
npx tsc --noEmit
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v9.mp4 --scale=0.5 --concurrency=4
npx remotion still src/index.ts QTrialCase out/case_audit/v8_check.png --frame=840 --scale=0.5
```

## 8. 待用户评价的重点

- S1 换混合门负载是否接受（vs 保持 QUEKO 原始 X-only）。
- S2 7 s 是否合适；交互图先全屏落位是否喜欢。
- S3 4 列候选墙 + 指标两行是否清晰。
- S4 线路主导、拓扑缩小、状态条解释是否达到“看得出路由怎么实现”。
- S5 6 张 pass 卡 + 底部流水线是否足够；是否需要 A1 机制 inset。
- S6/S7 的篇幅与信息量。
- 78 s 总长是否合适。
