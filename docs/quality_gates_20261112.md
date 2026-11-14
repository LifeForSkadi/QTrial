# 决赛交付质量门记录（2026-11-12）

四项质量门按顺序执行，证据可复核。

## ① qinyan-nature-review 投稿前预审

- 报告：`docs/review_final_report_20261112.md`
- 一致性校验：`python .claude/skills/qinyan-nature-review/scripts/review_consistency.py docs/review_final_report_20261112.md`
  → `records=17 valid=true errors=[] warnings=[]`
- 结论：无 P0；2 项 P1 共识风险（零 qiskit 边界、复现材料），其余为 P2。
- 已整改：见报告末尾"整改记录（2026-11-12）"；包括
  零 qiskit 表述收窄、复现材料补包、587/583 统一、胜场分母、
  bootstrap/Wilson 95% CI、测试失败披露、结论"全面第一"过度声称修正、
  Web 对照默认关闭并加注。
- 复检：PDF 全文检索"全面第一/全面领先/基本持平/值得注意"均为 0；
  `作品报告书_决赛.pdf` 61 页。

## ② video-shotcraft 独立终检

- 成片：`promo_video/out/qtrial-demo.mp4`（75s，1920×1080，30fps，无音轨）
- 关键帧：`promo_video/out/review/f0000.png` … `f2249.png`（18 张）
- 审查输入：`promo_video/out/review/BRIEF.md`、`references/final-review.md`、
  `references/aesthetic-rules.md`、`src/Demo.tsx`
- 审查方式：干净上下文 subagent（未参与制作），只读；本机若图片读取失败，
  以 ffmpeg 抽帧 + OCR/像素探针取证。
- v1 问题：每镜头独立 10 帧淡出到 0，镜头交界出现约 0.3s 近黑帧；
  已改为镜头间硬切，仅保留片头 10 帧淡入与片尾 15 帧淡出。
- v2 成片：`promo_video/out/qtrial-demo.mp4`（75s，20.3MB）；
  PPT 内嵌媒体已同步为 v2（`ppt/media/media1.mp4` 20,268,551 bytes）。
- 第一轮独立审查 agent 在 OCR 取证阶段无输出（7 分钟），已停止；
  v2 重新派独立 agent，限定 13 张关键帧、仅用 Read/Pillow，
  不调用 PowerShell OCR，10 分钟时限。
- v2 独立终检结论（干净上下文 agent，62 次工具调用）：
  必须修复 3 项（字幕 34px 低于 56px 规范；S02 白底管线图与深色片冲突；
  审计小字 18px），建议 7 项（Web 勾选态前后不一致、假光标未落在按钮、
  打字速度过快、辅助字号偏小、结果图数值偏小、"祖冲之三号同款"表述、
  图注数字重复）。
- v3 整改：字幕 56px 并缩短文案；新增 `fig_pipeline_dark.png` 深色管线图；
  审计小字 32px 并缩短；Web 勾选态统一为未勾选、光标移到"执行映射"按钮、
  打字放慢至 2.4 帧/字符、终端/标题/收尾字号上调、深色结果图字号加大；
  Web 副标题改为"祖冲之三号同款架构"。
- 复核：v3 关键帧 `promo_video/out/review/v3_*.png`；主会话逐帧确认
  字幕 56px、深色管线图、审计 32px、勾选态、光标、打字速度、结果图字号均生效。
- 报告：`promo_video/out/review/REPORT.md`（v2 报告 + v3 整改表；
  独立 agent 闭环复核待回复后补记）。
- v3 成片：75s，20.6MB；独立复核确认字幕/深色管线 ✓，但审计披露被删、
  S07 字幕"零依赖"扩大化 → v4 恢复"含 2 项已知断言"、字幕改
  "映射内核零 qiskit"、第三卡去孤字、Web 字幕中性化。
- v4 成片：75s，20.6MB；独立 agent 最终确认两项必须修复 ✓、
  无新增必须项（非阻断："环境性"可补、第一卡断行、同款架构待核实）。
- 交付包已同步 v4：`决赛提交/` 88MB，PPT 内嵌 media1.mp4 20.6MB，
  SHA256SUMS 已更新。

## ③ research-writing-skill 中文润色 / 去 AI 味

按 `references/cn-prompt-templates.md` 的"润色中文"与"去 AI 味（Word 中文）"
原则，对摘要、5.4.2/5.4.5/5.4.6、结论做定向修改：

- 删除/替换模糊与渲染性表达："值得注意""代价巨大""高度契合""全面领先"
  "全面第一""基本持平"（保留有数据支撑的"接近"）。
- 减少机械破折号：5.4.2/5.4.5/5.4.6 与结论的解释性"——"改为冒号/逗号/句号。
- 修正歧义：大线路 2Q 深度"高于噪声感知 O3"改为
  "比噪声感知 O3 高约 34%（2Q 深度越低越好）"。
- 修正过度声称：结论"布局贡献实验…保真度全面第一"改为
  "高于盲目 qiskit 管线、噪声感知 O1 与 pytket；与噪声感知 O3 的差距见 5.4.3"。
- 统一口径："基本持平"改为"接近但略低"（小层 vs 感知 O3，含 bootstrap CI）。
- 数字、引用、公式未改；PDF 复核 0.735/0.146/3031/130-578/95% CI 均在。

## ④ office-academic-skill PPT 全检

- 溢出检查（PowerPoint COM，`inspect_pptx_overflow.ps1`）：
  `overflow_count=0`（修复了第 9 页卡片正文超宽一处）。
- 文本扫描（`scan_pptx_text.py`）：9 页，`bad_hits=[]`
  （无 TODO/占位/旧"零 qiskit 依赖"表述）。
- 字体：全部为"微软雅黑"（中文可读；英文/数字使用同一字族，属可接受偏差）。
- 备注：9/9 页均有演讲者备注（79–232 字）。
- 视频：第 6 页存在 MEDIA 形状（`qtrial-demo.mp4`），媒体已嵌入 PPT 包。
- 图片：第 3/7 页图片显示纵横比与原始纵横比误差 0.0%。
- 页序：封面→问题→管线→RL 布局→路由/后处理→演示→结果→验证→价值，符合 6 分钟分镜。
- 导出：`out/qa` 与临时目录逐页 PNG 目检通过；第 9 页两行正文无溢出。

## 交付包

`决赛提交/`（87MB）：
- `01_源码/QTrial/`：v8 源码 + 生产权重 + LAUREL 权重 + 基线逐线路行 +
  机制实验数据 + 复现脚本 + 报告构建链
- `02_报告书/`：`作品报告书_决赛.docx/.pdf`（61 页）
- `03_PPT/`：`作品展示PPT_决赛.pptx`（9 页，内嵌 75s 视频，含备注）
- `04_演示视频/`：`演示视频.mp4`（75s）
- `05_讲稿/`：`讲稿_6分钟.md`
- `SHA256SUMS.txt`

待提交人处理：封面单位/姓名/手机号；一式六份 A4 双面打印；U 盘与报到材料。
