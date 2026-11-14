# 案例演示片上下文恢复点（2026-11-13）

> 用途：上下文压缩后快速恢复“案例演示片重做”项目状态。
> 当前状态：已产出 90 s / 540p 评审草稿 v6；用户反馈“问题很大”，具体改进方向待用户压缩后说明。**不要自行开始下一轮改动。**

---

## 1. 当前产物

- 评审草稿：`promo_video/out/qtrial-case-draft-v6.mp4`
  - 90 s（2700 帧 @30 fps）、960×540 评审版（以 1920×1080 设计、`--scale=0.5` 渲染）、无音轨。
- 自审记录：`docs/case_film_audit_20261113.md`
  - 场景表、数据一致性、已修复项、已知限制、复现命令。
- 关键帧：`promo_video/out/case_audit/v6_*.png`、`v4_*.png`、`v3_*.png`。
- 旧成片 `QTrialDemo`、`决赛提交/`、PPT 未改动；本片尚未嵌入 PPT。

## 2. 新增/修改文件

新增：
- `promo_video/src/CaseFilm.tsx`：90 s 时间线 + 章节卡 + 全局字幕
- `promo_video/src/case/LogicCircuit.tsx`：S1 逻辑线路
- `promo_video/src/case/RouteSync.tsx`：S4 拓扑 + 逻辑行线路同步
- `promo_video/src/case/PassDetail.tsx`：S5 后处理
- `promo_video/src/case/FinalCard.tsx`：S6 最终线路
- `promo_video/src/case/RealMachine.tsx`：S7 真机表
- `promo_video/src/case/Summary.tsx`：S8 结果总结
- `promo_video/src/data/case_logic.json`：QUEKO 逻辑线路展示数据
- `tools/dump_case_video_data.py`：生成 `case_logic.json`
- `docs/case_film_audit_20261113.md`：自审记录
- `docs/case_film_recovery_20261113.md`：本文件

修改：
- `promo_video/src/Root.tsx`：注册 `QTrialCase` composition（2700 帧）
- `promo_video/src/shots/S05_CaseCircuit.tsx`：Kicker 编号改为 `02 / 08 交互图 + 拓扑`
- `promo_video/src/shots/S06_CaseLayout.tsx`：Kicker `03 / 08`；机制说明单行；图例缩短；候选 footer 加 `seed 7`
- `promo_video/src/shots/S07_CaseRoute.tsx`：Kicker `04 / 08`；导出 `TraceFilm`；新增“路由规则”卡；QCIS 片段改为门集；措辞“真机拓扑上实际运行”

未修改：
- `promo_video/src/Demo.tsx`、`promo_video/src/Promo.tsx`、`决赛提交/`、旧 PPT/视频/报告书。

## 3. 时间线与数据

| # | 帧 | 时间 | 场景 | 关键数据 |
|---|---|---|---|---|
| 01 | 0–240 | 0–8 s | 逻辑线路 | QUEKO 20 比特；45 CX，40 个跨距 >1；450 个 1Q |
| 02 | 240–620 | 8–20.7 s | 交互图 + 拓扑 | 20 比特、45 个 2Q；目标芯片 15×7（真实拓扑 105 节点） |
| 03 | 620–970 | 20.7–32.3 s | 布局竞技 | seed 7；13 个候选；胜者 63 SWAP vs 平铺 120 |
| 04 | 970–1320 | 32.3–44 s | 路由同步 | 63 次总交换；28 次逻辑 token 间交换逐步展示；其余含空位交换快进 |
| 05 | 1320–1740 | 44–58 s | 后处理 | 14 个 SWAP 折叠；63→49；VF2++ 重标记 0 次（relink=0）；A2/A3/B 按需 |
| 06 | 1740–1980 | 58–66 s | 最终线路 | 49 SWAP；depth 268；2Q depth 65；fid 0.173；平台基 RZ/X2P/CZ/M |
| 07 | 1980–2340 | 66–78 s | 真机验证 | tianyan176 4 条：x5 0.6170、ghz5 0.5943、ghz10 0.2209、qft5 0.9539 |
| 08 | 2340–2700 | 78–90 s | 结果 + 收尾 | 583 条；0.227；1.21× pytket；0.954 真机；对感知 O3 0.924×、22.5% 胜出 |

补充：
- 案例数据源：`promo_video/src/data/case_trace.json`（winner_layout 20、63 swap_trace、14 absorb、49 keep、relink=0、diagram_ops 558）、`case.json`（13 候选、final/v5 指标、qcis_pick）、`topo_real.json`（105 节点真实拓扑）。
- 真机数据源：`tables/real_machine_v8/summary.md`；31/31 为发布前合法性/语义审计，真机执行只有 4 条；tianyan-287 机时不足。
- 结果数据源：`tables/submission/summary.md`（583 完成、0.227/0.246/0.187、胜场 130/469/531）。

## 4. 关键决策（可能被用户推翻）

1. **主案例用 QUEKO-20，不用 QFT5**：
   - 仓库已有真实路由/折叠轨迹（63 SWAP、14 步折叠），S3–S5 可用真实数据；
   - QFT5 真机只有 5 SWAP，轨迹细节少；QFT5/GHZ10 放在 S7 真机段。
   - 用户尚未确认此决策。
2. **90 s**：6 分钟 PPT 的 75 s 版本扩到 90 s；若用户要压回 75 s，需砍 S3 或 S5。
3. **540p 评审版**：按用户要求继续 540p；定稿再 1080p。
4. **章节卡**：S1–S8 开头 14 帧章节卡，掩盖硬切到近黑帧；用户可能觉得像 PPT，需评估。
5. **S4 以逻辑行展示线路**：20 行 L0–L19；物理 token 在左，逻辑行在右；28 次逻辑交换逐步，63 次含空位快进。
6. **VF2++ 未触发**：`relink=0`，S5 只写“候选 0 次”，不虚构过程。
7. **A2 未单独可视化**：S5 卡片 3 只说明“按需”，无逐块前后对比。

## 5. 已知问题 / 用户可能不满意的点

自审已列（详见 `docs/case_film_audit_20261113.md`）：
- S3 信息密度高：13 行候选 + 三个代表卡 + 胜者/平铺对比。
- S5 胶片偏小：14 步折叠的“并入置换”标签仅 12 px，右侧卡片承担主要解释。
- S4 只展示 28 次逻辑交换，63 次中含空位交换快进；用户可能想要完整 63 次。
- VF2++ 没有过程画面；A2 没有逐块画面。
- 章节卡可能显得“PPT 化”，不是电影感转场。
- 90 s 对 6 分钟 PPT 偏长；用户可能想压回 75 s。
- 540p 评审版清晰度有限；无音轨；未嵌入 PPT。
- 旧 S05/S06 镜头复用，S3 的排版/密度可能不符合用户“每一帧布局”的要求。

**用户原话**： “问题很大，还是继续生成 540p 的评审版，不过在此之前需要先压缩一下上下文……整理压缩完我告诉 boss 该如何改进。”
→ 压缩后先等用户说明改进方向，不要自行猜测开工。

## 6. 复现 / 渲染命令

```bash
cd promo_video
npx tsc --noEmit
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v6.mp4 --scale=0.5 --concurrency=4
npx remotion still src/index.ts QTrialCase out/case_audit/v6_frame.png --frame=1050
```

- 数据生成：`python tools/dump_case_video_data.py`（从 `data/Queko/BIGD/20QBT_45CYC_.5D1_.1D2_0.qasm` 生成 `promo_video/src/data/case_logic.json`）。
- 旧成片：`QTrialDemo` 仍为 75 s；不要覆盖。
- 当前工作目录曾切换到 `promo_video`；命令注意相对路径。

## 7. 环境与 git

- 本会话未提交 git；新增/修改文件见上。
- 旧 `docs/session_recovery_20261113.md` 仍描述决赛交付包（88 MB、报告 61 页、PPT 9 页、视频 v4 75 s），与本视频重做项目并行，不要混淆。
- 无运行中的 Remotion/Chrome 进程（渲染已结束）。
