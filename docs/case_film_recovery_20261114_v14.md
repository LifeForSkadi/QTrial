# 案例演示片 v14 恢复点（2026-11-14）

> **已被 v15 取代：见 `docs/case_film_recovery_20261114_v15.md`。**
> v14 保留作历史记录；v15 修了读音、后处理对点、页面过渡和审计句。

> 当前状态：v13 画面 + **新克隆旁白已贴进视频**，无 BGM 校对版已渲染：
> `promo_video/out/qtrial-case-draft-v14-voice-nobgm.mp4`（84.885 s，960×540）。
> **现在等用户按字幕内容逐条提修改；拿到意见前不要动字幕、不要选 BGM。**
> 续接时先读本文件 + `docs/kaikou_voice_v2_recovery.md` + `docs/case_film_text_v14_draft.md`。

## 1. 现在在哪一步

- v13 的画面/字幕/双版本已完成：`promo_video/out/qtrial-case-draft-v13-silent.mp4`、`...-voice.mp4`。
- v13 的 AI 旁白（SAPI Huihui）和 BGM 用户都不满意。
- 已安装 `kaikou` skill，用用户本人录音克隆了 17 句新旁白：
  - 主力参考：S4（中英数字），加快 8%。
  - 全片旁白 80.33 s，视频 84.87 s，装得下。
  - 用户已听 `sample.wav` / `narration_all.wav`，确认**声音没问题**。
- 下一步不是继续调声音，而是**贴进视频**，让用户对着字幕提修改。

## 2. 必读文件

1. 本文件。
2. `docs/kaikou_voice_v2_recovery.md`：17 句、时长、建议 from 帧、复现命令、环境。
3. `docs/case_film_recovery_20261114_v13.md`：v13 画面/时间线/文件。
4. `docs/case_film_text_v13.md`：v13 全片文本（旧字幕）。
5. `promo_video/kaikou/script_v2.txt`：新的 17 句稿。
6. `promo_video/kaikou/vo/v2/lines/`：新的 17 句成品。

## 3. 贴片步骤（等用户回来后再做）

已准备：

- 17 个 WAV 已复制到 `promo_video/public/audio/voice/v2/`。
- 草稿 `promo_video/src/data/narration_v2.json` 已生成（17 句、建议 from/dur、最后一句结束在 2546 帧）。

接下来：

1. ✅ `src/components/Narration.tsx` 已改 import `narration_v2.json`。
2. ✅ `src/CaseFilm.tsx` 已改读 `narration_v2.json` 的 `audio/voice/v2/Lxxx.wav`；新增 `bgm?: boolean`，默认 false，本轮无 BGM。
3. ✅ 已渲染无 BGM 的 voice 版：`out/qtrial-case-draft-v14-voice-nobgm.mp4`（84.885 s，960×540，AAC 48 kHz，最大 -3.2 dBFS）。逐句入点与计划时间轴对齐（误差 < 50 ms）。
4. ⏳ 等用户按字幕提修改 → 改 `script_v2.txt` → `voice_torch.py --only 改过的句子` → `rescore.py` → `assemble.py` → 更新 `narration_v2.json`。
5. 字幕定稿后，做 3 个 BGM 盲听选项，再渲染 v14 无声版 + 旁白版。
6. 最后更新 `case_film_text_v14.md`、`case_film_audit_20261114_v14.md`、本恢复点和记忆。

## 4. 用户最新原话

> “先做进视频，我根据字幕内容来说明需要修改的地方，声音听起来没有问题。不过在开始视频制作之前，我可能需要先帮boss你压缩一下上下文，你准备一下吧 *wink*”

- 声音已通过；**不要重新录、不要重新选参考、不要重做声音**。
- 用户会按**字幕内容**提修改；等他们给意见。
- 压缩后如果用户还没给字幕意见，**不要自行改字幕或渲染**；先把环境准备好、读文件、等指令。

## 5. 产物路径

- **v14 字幕校对版（新旁白，无 BGM）**：`promo_video/out/qtrial-case-draft-v14-voice-nobgm.mp4`
- v13 无声：`promo_video/out/qtrial-case-draft-v13-silent.mp4`
- v13 旁白：`promo_video/out/qtrial-case-draft-v13-voice.mp4`
- 新旁白整条：`promo_video/kaikou/vo/v2/narration_all.wav`
- 新旁白试听：`promo_video/kaikou/vo/v2/sample.wav`
- 新旁白逐句：`promo_video/kaikou/vo/v2/lines/L001..L017.wav`
- 新稿：`promo_video/kaikou/script_v2.txt`

## 6. 环境提醒

- 录音/盲听服务已停；需要时再起：
  - 录音：`python .claude/skills/kaikou/tools/recorder/server.py --list promo_video/kaikou/list_qtrial.json --out promo_video/kaikou/ref/raw --no-asr`
  - 盲听：`python .claude/skills/kaikou/tools/audition/server.py promo_video/kaikou/vo/audition/round1 --port 8766`
- `ffmpeg` 在 `C:\Users\SKADI\bin`，PATH 可用。
- Qwen3-TTS / Whisper 模型已缓存，不需要重新下载。
