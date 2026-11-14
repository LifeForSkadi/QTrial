# 案例演示片 v16 恢复点（2026-11-14）

> 当前状态：按用户要求整片重写，文稿走 kaikou 第 5 步并通过 `lint_script.py`；
> 画面按音频反推拉长；无 BGM 校对版已渲染（2026-11-14 13:33）。
> 成片：`promo_video/out/qtrial-case-draft-v16-voice-nobgm.mp4`（4003 帧 / 133.44 s / 960×540 / 17.9 MB）。
> 文稿：`docs/case_film_text_v16.md`；配音稿：`promo_video/kaikou/script_v5.txt`。

## 1. 本版做了什么

用户原话：逐点改效率太低，直接重写字幕+配音。要求：
1. 不看视频，仅凭音频能完整知晓视频在展示什么；
2. 音频与视频动画一一对应，可调节各环节演示时间；
3. 音频不应该有明显的停顿或空闲时间。

本版：
- 21 句连续讲解，开场先给路线图（20 比特、45 个 CX；布局、路由、后处理、真机验证四关）。
- 按 kaikou 第 5 步改文案：短句、双字词、语气词、问题带路、口语接句；`lint_script.py` 无问题。
- 场景时长按每段音频反推：S1 650 / S2 444 / S3 501 / S4 590 / S5 780 / S6 429 / S7 591 / S8 172 帧。
- 画面节奏同步调整：
  - S1：跨距 CX 红线与 38/45 说明延后到 380 帧，正好接 L03；线路滚动延长到 630 帧。
  - S2：目标芯片延后到 170 帧出现，正好接 L05。
  - S3：平铺计数、候选墙、胜者揭晓按新时长放大（胜者 330–390 帧）。
  - S4：路由步进整体放慢 1.3 倍，路线走完接 L11 的“走到头”。
  - S7：结果表、QFT 对照卡、审计徽章分别延后到 20–92 / 150–210 / 330 帧。
  - S8：片尾淡出按 172 帧总长收束。

## 2. 读音

- CX → `C叉`（字幕 CX）
- SWAP → 英文 `swap`（字幕 SWAP；不再是“斯瓦普”）
- Qiskit → `Q斯kit`（字幕 Qiskit）
- 重标记 → `重新标记`（chóng）
- 保真度 → 配音 `保真程度`，字幕仍写“保真度”
- 天衍 → 配音 `天演真机`，字幕“天衍真机”
- v8 → 配音 `v 八`，字幕 v8
- QTrial → 配音 `Q Trial`，字幕 QTrial

## 3. 文件

| 内容 | 路径 |
|---|---|
| 配音稿 | `promo_video/kaikou/script_v5.txt` |
| 逐句成品 | `promo_video/kaikou/vo/v5/lines/L001..L021.wav` |
| 整条旁白 | `promo_video/kaikou/vo/v5/narration_all.wav`（128.87 s） |
| 贴片时间轴 | `promo_video/src/data/narration_v5.json` |
| 公共音轨 | `promo_video/public/audio/voice/v5/L001..L021.wav` |
| 时间轴脚本 | `promo_video/kaikou/timeline_v5.py` |
| 全片文本 | `docs/case_film_text_v16.md` |

## 4. 复现命令

```bash
cd promo_video/kaikou
# 全量 21 句
PYTHONIOENCODING=utf-8 python voice_torch.py script_v5.txt vo/v5 --ref ref/clean/S4_fast.wav --takes 4
# 只重配某句（例：第 8 句）
PYTHONIOENCODING=utf-8 python voice_torch.py script_v5.txt vo/v5 --ref ref/clean/S4_fast.wav --takes 4 --only 8
# 重新挑遍 + 拼整条
PYTHONIOENCODING=utf-8 python rescore.py script_v5.txt vo/v5
PYTHONIOENCODING=utf-8 python assemble.py script_v5.txt vo/v5
# 重排时间轴 + 复制 public（改场景入点后重跑）
PYTHONIOENCODING=utf-8 python timeline_v5.py
# 渲染无 BGM 校对版
cd ..
npx remotion render src/index.ts QTrialCaseVoice out/qtrial-case-draft-v16-voice-nobgm.mp4 --scale=0.5 --concurrency=4
```

## 4. 渲染后核对

- 成片：4003 帧 / 133.44 s / 960×540 / 17.9 MB / AAC 48 kHz，峰值 -2.9 dBFS。
- L08/L09/L016 开头的低电平前导已裁掉（最长 L09 0.9 s），`vo/v5/lines_raw/` 保留原文件。
- L10 由 1670 提前到 1655 帧，S4 两处停顿从 1.3 s 压到 0.8/0.9 s。
- 剩余 >0.8 s 的间隔都在页面/子步骤切换处（S3→S4、S4→S5、S5 各幕、S5→S6），属自然换气。

## 5. 下一步

1. 用户看 v16，按 L01–L21 提意见。
2. 改 `script_v5.txt` → 只重配改动句 → `rescore.py` → `assemble.py` → `timeline_v5.py`（必要时改 `S5_LOCAL`）→ 重渲。
3. 字幕定稿后做 3 个 BGM 盲听选项（`house-vibez` 用户不喜欢），再出 v16 无声版 + 旁白版。
4. 更新 `case_film_text_v16.md`、`case_film_audit_20261114_v16.md`、本恢复点和记忆。

## 6. 注意

- 声音本身已通过，不再重录/重选参考。
- 本版不设时长限制；先保证完整，再考虑压缩。
- 对外材料不声明“之前是 bug/修复”；当前数字真实可复现。
