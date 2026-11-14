# 会话恢复点（2026-11-13）

> 本文件用于上下文压缩后快速恢复。决赛交付已完成，四项质量门全部执行并闭环。

## 当前状态

- **交付包**：`决赛提交/`，88 MB，已含源码、报告书、PPT、视频、讲稿、SHA256SUMS。
- **报告书**：`作品报告书_决赛.pdf` 61 页 A4；`作品报告书.docx` 同源。
- **PPT**：`作品展示PPT_决赛.pptx` 9 页，第 6 页内嵌 v4 演示视频，逐页演讲者备注。
- **视频**：v4，75 s，1920×1080，30 fps，无音轨；镜头间硬切；Web 实拍结果
  4 SWAP / 2Q 32 / 深度 99 / 2Q 深度 28 / 保真度 0.8237 / 0.815 s。
- **源码**：git HEAD `6f4fdc7`（v8 全量重评 + 审计 P0 修复），包内附生产权重、
  LAUREL 权重、基线逐线路行、机制实验数据与复现脚本。
- **质量门**：全部完成（见下）。

## 交付物校验（SHA256）

- `决赛提交/02_报告书/作品报告书_决赛.docx` 1.85 MB `2239f2bd56c2755704fc2602b7d5ec8dd43982ef01227ace04e1d8aa2432f2b4`
- `决赛提交/02_报告书/作品报告书_决赛.pdf` 2.20 MB `5d548f53b5949d8031b9ca4d5a7f5ad7e652c3e13ada6d4e479c656225ac4927`
- `决赛提交/03_PPT/作品展示PPT_决赛.pptx` 21.34 MB `e60ba7ea2c4a0ddfbd1840cdd5c790b86766bb70fb9994a4dd6f7eae7b3d1093`
- `决赛提交/04_演示视频/演示视频.mp4` 20.57 MB `a6b3a8c814c6a154ef109c3fd35719a9ba65ba4fb831fee9010dad44c5b56371`
- `决赛提交/05_讲稿/讲稿_6分钟.md` 4.7 KB `1f04d8e51ab632b96b877b786b8866eb6de40b78431635a49d1a09c1bd8993bb`

## 质量门证据

- `docs/review_final_report_20261112.md`：qinyan 预审 17 条、2 项 P1 共识与整改记录。
- `docs/quality_gates_20261112.md`：四项质量门执行记录。
- `promo_video/out/review/REPORT.md`：video-shotcraft 独立终检（v2 问题 + v3/v4 整改 + 独立确认 ✓）。
- `promo_video/out/review/v4_*.png`：v4 关键帧证据。
- `promo_video/out/review/BRIEF.md`：视频简报/分镜/tokens。

## 关键数字（对外口径）

- 583 条完成线路（587 基准，4 条 900 s 超时）。
- 总体估计保真度 0.227；噪声感知 O3 0.246（0.924×）；pytket 0.187（1.21×）。
- ≤10：0.735，O3 0.761（0.965×，bootstrap 95% CI 0.957–0.973）。
- 11-50：0.146，O3 0.163（0.895×）；pytket 0.106（1.37×）。
- 51-105：2Q 深度 3031，pytket 3025（持平），O3 2263（高约 34%，越低越好）。
- 审计：真机合法 31/31、measure 31/31、端到端保真度 1.000000、232 项测试通过 + 2 项已知环境性断言。

## 待用户处理

1. 报告书封面填单位/姓名/手机号。
2. 报告书一式六份 A4 双面打印（≤80g，勿过度装裱）。
3. U 盘拷贝 `决赛提交/` + 承诺书/保险等报到材料。
4. 现场提前试播 PPT 第 6 页（视频已内嵌；`04_演示视频/演示视频.mp4` 为备份）。

## 非阻断项（视频终检遗留）

- 审计卡"含 2 项已知断言"可补"环境性"（当前 32px 已可读）。
- S07 第一卡正文"复/用"处断行（可改文案，非必须）。
- "祖冲之三号同款架构"出处无法从仓库核验（项目记忆支持，保留）。

## 重建命令

```bash
# 报告书（JSON 为构建真相源；build.py 已关闭内部自检附录）
python scripts/thesis/build.py
# Word COM 导出 PDF（脚本在 %TEMP%/docx_to_pdf.ps1）

# PPT（内嵌视频 + poster + 9 页备注）
python scripts/thesis/make_ppt_final.py promo_video/out/qtrial-demo.mp4 promo_video/out/qa_demo/f800.png

# 视频（Remotion，已改用自带 chrome-headless-shell；remotion.config.ts.bak 为旧 Edge 配置）
cd promo_video && npx remotion render src/index.ts QTrialDemo out/qtrial-demo.mp4

# 提交包 + 校验和
python scripts/build_submission_package.py
```

## 环境与 git

- Web 演示服务器已停止；重启：`python -m qtrail.web`（固定 8000）。
- Web 实拍：`python tools/capture_web_demo.py`（Edge CDP，需先启动 web）。
- git HEAD `6f4fdc7`；工作树 27 项修改/未跟踪（README、web、thesis content/build、
  三张报告图、review/quality 文档、submission 脚本、make_ppt_final.py、`决赛提交/`、
  `promo_video/` 等）。**本次会话未提交 git**；`决赛提交/` 为自包含交付物。
