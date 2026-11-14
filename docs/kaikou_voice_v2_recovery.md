# 案例片 v14 · 克隆旁白恢复点（2026-11-14）

> 状态：**用户已确认声音没问题**，但还没贴进视频；用户要先压缩上下文，回来后根据字幕内容逐条提修改。
> 视觉仍用 v13 的 `CaseFilm`；本文件只管“换声音 + 换字幕”的语音链路。

## 1. 结论

- 录音 24 条，房间底噪 -85 dBFS，参考条 SNR 42–50 dB，无削波。
- 盲听用户选 **S4（中英数字参考）**。
- 用 S4 加快 8% 的参考，生成 17 句正式旁白，每句 4 遍，Whisper 回听 + 数字归一后挑最好的一遍。
- 全片旁白 **80.33 s**，视频 84.87 s，装得下。
- 用户已听 `sample.wav` / `narration_all.wav`，确认**声音没问题**。
- 下一步：贴进视频，让用户按字幕提修改；之后再做 BGM 盲听。

## 2. 当前文件

| 内容 | 路径 |
|---|---|
| skill | `.claude/skills/kaikou/` |
| 录音清单 | `promo_video/kaikou/list_qtrial.json` |
| 原始录音 | `promo_video/kaikou/ref/raw/R1_1.wav`、`S1_1..S5_3.wav`、`V_1..V_6.wav`、`F_1/F_2.wav` |
| 清理参考 | `promo_video/kaikou/ref/clean/S1..S5.wav` + `.txt` |
| 主力参考 | `promo_video/kaikou/ref/clean/S4_fast.wav`（S4 的 atempo=1.08） |
| 17 句稿 | `promo_video/kaikou/script_v2.txt` |
| 逐句成品 | `promo_video/kaikou/vo/v2/lines/L001..L017.wav` + `.json` |
| 全部 take | `promo_video/kaikou/vo/v2/takes/` |
| 整条旁白 | `promo_video/kaikou/vo/v2/narration_all.wav`（80.33 s，-23 LUFS） |
| 40 秒试听 | `promo_video/kaikou/vo/v2/sample.wav`（40.72 s） |
| 真声对照 | `promo_video/kaikou/vo/v2/real.wav`（V_2，6.76 s） |
| 盲听结果 | `promo_video/kaikou/vo/audition/round1/picks.json`（chosen=S4） |

## 3. 17 句与时长

| # | 秒 | ASR 吻合 | 字幕（当前 show 文本） |
|---|---|---|---|
| 01 | 6.63 | 0.984 | 先看这条量子线路。四十五个 CX 当中，有三十八个两端的比特根本不相邻。 |
| 02 | 6.51 | 1.000 | 把线路变成交互图，再放进真实芯片：一百零五个比特，一百六十八条耦合器。 |
| 03 | 5.70 | 1.000 | 直接平铺，要补一百零三个 SWAP；十三个候选里，三号只用了五十六个。 |
| 04 | 6.93 | 0.954 | 布局只是把起点摆好。线路往前走一层，遇到不相邻的 CX，就补一个 SWAP。 |
| 05 | 4.05 | 1.000 | 补进来的 SWAP 会占掉一层，后面的门整体往后让。 |
| 06 | 4.84 | 1.000 | 路由结束，线路里留下五十六个 SWAP，接下来交给后处理。 |
| 07 | 3.51 | 1.000 | 后处理从线路尾部开始，先折叠 SWAP。 |
| 08 | 3.71 | 1.000 | 这一步折叠了十一个，五十六降到四十五。 |
| 09 | 4.84 | 1.000 | 再把相邻的两比特门块重新合成，又能抵消一对 CX。 |
| 10 | 2.01 | 1.000 | 单比特门合并到平台基。 |
| 11 | 4.55 | 1.000 | 重标记只在新映射保真度不降时才采用。 |
| 12 | 2.38 | 1.000 | 最后一道保险：不退化才通过。 |
| 13 | 5.31 | 1.000 | 四十四个 SWAP 去哪了？平台基没有 SWAP，每个展开成三个 CZ。 |
| 14 | 3.99 | 1.000 | 七条 v8 线路和三条 O3，都上了天衍真机。 |
| 15 | 3.76 | 0.947 | QFT5、QFT6 都高于 Qiskit O3。 |
| 16 | 4.54 | 0.957 | 三十一项发布前审计全部通过，不是三十一次真机执行。 |
| 17 | 1.12 | 1.000 | 这就是 QTrial。 |

> ASR 偶尔把 CX 写成 CXX/CHX、SWAP 写成 CWAP、Qiskit 写成 Kiskit、天衍写成天眼，这是识别问题；用户已确认声音没问题。

## 4. 建议的贴片时间轴（2546 帧 @30 fps）

| # | from 帧 | 约秒 | 计划 dur 帧 | 说明 |
|---|---|---|---|---|
| 01 | 15 | 0.5 | 207 | S1 |
| 02 | 264 | 8.8 | 202 | S2 |
| 03 | 468 | 15.6 | 179 | S3 |
| 04 | 708 | 23.6 | 216 | S4 |
| 05 | 930 | 31.0 | 124 | S4 |
| 06 | 1056 | 35.2 | 148 | S4→S5 |
| 07 | 1206 | 40.2 | 112 | S5 A1 |
| 08 | 1320 | 44.0 | 118 | S5 A1 |
| 09 | 1440 | 48.0 | 151 | S5 A2/A3 |
| 10 | 1593 | 53.1 | 67 | S5 B |
| 11 | 1662 | 55.4 | 142 | S5 B/C1 |
| 12 | 1806 | 60.2 | 80 | S5 C1/Guard |
| 13 | 1950 | 65.0 | 168 | S6 |
| 14 | 2124 | 70.8 | 124 | S7 |
| 15 | 2250 | 75.0 | 118 | S7 |
| 16 | 2370 | 79.0 | 139 | S7 |
| 17 | 2511 | 83.7 | 35 | 收尾 |

- 这些是建议值，等用户按字幕提修改后，以最终稿为准。
- `dur` 计算：`min(下一句 from - 本句 from - 2, 音频帧 + 8)`；最后一句取音频帧 + 4。

## 5. 环境

- Windows 11 + RTX 5090 24GB；PyTorch 2.11.0+cu128，CUDA 可用。
- `qwen-tts 0.1.1`（PyTorch 版），模型 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 已缓存。
- `transformers 4.57.3` + `openai/whisper-large-v3-turbo` 已缓存。
- `soundfile / librosa / pyloudnorm / pypinyin` 已装。
- `ffmpeg 9.0.2` 在 `C:\Users\SKADI\bin`，PATH 已可用。
- `faster-whisper` 装了但 PyAV 不兼容，**没有使用**；ASR 用 Transformers pipeline。

## 6. 复现命令

```bash
cd promo_video/kaikou
# 重新生成全部 17 句（约 10–15 分钟）
PYTHONIOENCODING=utf-8 python voice_torch.py script_v2.txt vo/v2 --ref ref/clean/S4_fast.wav --takes 4
# 数字归一后重新挑遍
PYTHONIOENCODING=utf-8 python rescore.py
# 拼接整条 + 40 秒试听
PYTHONIOENCODING=utf-8 python assemble.py
# 只重配某一句（例：第 4 句）
PYTHONIOENCODING=utf-8 python voice_torch.py script_v2.txt vo/v2 --ref ref/clean/S4_fast.wav --takes 4 --only 4
```

## 7. 已准备 / 下一步

已准备：

- `vo/v2/lines/L001..L017.wav` 已复制到 `promo_video/public/audio/voice/v2/`（17 个）。
- 已生成草稿 `promo_video/src/data/narration_v2.json`：17 句、建议 from/dur、audio 指向 `audio/voice/v2/Lxxx.wav`、text 先用 show 文本，最后一句正好结束在 2546 帧。

下一步（等用户回来后再做）：

1. ✅ `Narration.tsx` 已改 import `narration_v2.json`；`CaseFilm.tsx` 已改用 `audio/voice/v2/Lxxx.wav`（`bgm` 默认 false）。
2. ✅ 已渲染 **无 BGM 的 voice 版**：`promo_video/out/qtrial-case-draft-v14-voice-nobgm.mp4`（84.885 s，960×540）；字幕校对表在 `docs/case_film_text_v14_draft.md`。现在等用户按字幕提修改；不要先花时间选 BGM。
3. 用户改字幕 → 改 `script_v2.txt` → `voice_torch.py --only 改过的句子` → `rescore.py` → `assemble.py` → 更新 `narration_v2.json`。
4. 字幕定稿后做 3 个 BGM 盲听选项（当前 `house-vibez` 用户不喜欢），再渲染 v14 无声版 + 旁白版。
5. 最后更新 `docs/case_film_text_v14.md`、`case_film_audit_20261114_v14.md`、`case_film_recovery_20261114_v14.md` 和记忆。

## 8. 用户最新原话

> “先做进视频，我根据字幕内容来说明需要修改的地方，声音听起来没有问题。不过在开始视频制作之前，我可能需要先帮boss你压缩一下上下文，你准备一下吧 *wink*”

- 声音已通过，**不要重新录/重新选参考**。
- 视频还没开始改；先压缩上下文，等用户回来。
- 用户会按**字幕内容**提修改，不是按声音。
