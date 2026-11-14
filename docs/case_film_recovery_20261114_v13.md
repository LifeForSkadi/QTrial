# 案例演示片 v13 恢复点（2026-11-14）

> 当前状态：84.9 s / 540p，无声版 + BGM/旁白版均已完成并自查；等待用户观看。
> 审计：`docs/case_film_audit_20261114_v13.md`
> 文本：`docs/case_film_text_v13.md`
> 音频：`docs/audio_credits_v13.md`

## 1. 当前产物

- `promo_video/out/qtrial-case-draft-v13-silent.mp4`
  - 84.9 s / 2546 帧 / 960×540 / 无音轨 / 8.75 MB。
- `promo_video/out/qtrial-case-draft-v13-voice.mp4`
  - 84.9 s / 2546 帧 / 960×540 / AAC 音轨 / 12.0 MB。
- Composition：
  - `QTrialCase`（voice=false）
  - `QTrialCaseVoice`（voice=true）
  - 均在 `promo_video/src/Root.tsx`，画面共用 `CaseFilm.tsx`。

## 2. 本版改动

- S1：填充式单比特门 44×28→30×17；旋转门 56×28 不变；CX 40×20 不变；颜色不变；2Q 断口随小门收紧。
- A1：折叠节奏改为 50 帧起步、前 4 次 26 帧、其余 16 帧，最后一次约 47.6 s 完成；45–47 s 不再静止。
- 后处理转场：左侧 8 帧淡出 + 8 帧淡入；右侧卡片 0.32/0.72/1 三级亮度平滑插值；底部公式卡 0.4↔1。
- 字幕：删除 4 处总结式大字幕；新增 22 句口语字幕，底部 31 px，淡入 6/淡出 8。
- 文本：S3 `103 个`、S3 路由打分句式、S6 两种表示表述（采纳建议 1–3）；建议 4 不采纳。
- 音频：SAPI Huihui 旁白 22 句 + Mixkit `House Vibez` BGM；旁白增益 1.25，BGM 0.035。

## 3. 关键文件

- `promo_video/src/CaseFilm.tsx`：两版、Narration、BGM、22 句 Audio Sequence。
- `promo_video/src/components/Narration.tsx`：底部口语字幕。
- `promo_video/src/data/narration.json`：from / dur / text / audio。
- `promo_video/tools/narration_src.json`：旁白文本与起始帧。
- `promo_video/tools/tts_narration.ps1`：SAPI 离线合成。
- `promo_video/tools/make_narration.mjs`：ffprobe 测长、生成 JSON、检查重叠。
- `promo_video/public/audio/voice/n01–n22.wav`：旁白。
- `promo_video/public/audio/bgm/house-vibez.mp3`：BGM。
- `promo_video/src/case/LogicCircuit.tsx`：S1 小门。
- `promo_video/src/case/PassStudio.tsx`：A1 节奏、淡入淡出、卡片亮度。

## 4. 旁白时间轴

见 `docs/case_film_text_v13.md` 第 0 节。最晚一句 n22 83.4–84.9 s。

## 5. 自查

- tsc 通过；两版均 2546 帧 / 84.9 s。
- 无声版无音轨；voice 版 AAC，时长 84.885 s。
- 22 句旁白逐句 RMS 均有语音；BGM 空段 0.008–0.013；峰值 0.852。
- 抽查 7 个关键帧，无溢出/双图/黑帧。
- 540p 下 S5 单门标签、S7 小字仍偏小；1080p 会改善。

## 6. 待用户确认

1. S1 缩小后的单比特门是否够小、CX 是否够突出。
2. 45–47 s 折叠节奏是否不再停滞。
3. 后处理左侧淡出淡入、右侧卡片 0.72 亮度是否合适。
4. 口语字幕的语气、长度、断句是否满意。
5. 旁白音色（SAPI Huihui）与 BGM（House Vibez）是否接受；若要换，请给方向。
6. S3/S6 定稿文字是否满意。

## 7. 复现

```bash
cd promo_video
npx tsc --noEmit
powershell -NoProfile -ExecutionPolicy Bypass -File tools/tts_narration.ps1
node tools/make_narration.mjs
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v13-silent.mp4 --scale=0.5 --concurrency=4
npx remotion render src/index.ts QTrialCaseVoice out/qtrial-case-draft-v13-voice.mp4 --scale=0.5 --concurrency=4
```
