# 案例演示片 v13 自审记录（2026-11-14）

> 成片：
> - 无声版 `promo_video/out/qtrial-case-draft-v13-silent.mp4`
> - BGM + 旁白版 `promo_video/out/qtrial-case-draft-v13-voice.mp4`
> 规格：84.9 s / 2546 帧 / 960×540 / 30 fps；voice 版含 AAC 音轨
> 文本：`docs/case_film_text_v13.md`
> 音频：`docs/audio_credits_v13.md`
> 恢复点：`docs/case_film_recovery_20261114_v13.md`

## 1. 用户 4 条意见的落实

### 1.1 S1 缩小单比特门，突出 CX

- 填充式单比特门（H/X/Y/Z/S/T）从 44×28 缩到 30×17，圆角 3.5，字号 12。
- 带角度的旋转门（RX/RY/RZ）保持 56×28，避免角度标签被挤掉。
- CX 符号保持 40×20，竖线 3.4/4.2 px、红色高亮 5 px，颜色未改。
- 2Q 线穿过 1Q 门时的断口同步从 ±14 改为 ±8.5（旋转门仍 ±14），断口更贴合小门。

### 1.2 45–47 s 停滞与后处理转场

1. **A1 折叠节奏**
   - 旧节奏：18 帧开始，前 4 次 24 帧，其余 7 帧；最后一次约 45.2 s 完成，之后静止约 2.9 s。
   - 新节奏：50 帧开始，前 4 次 26 帧，其余 16 帧；最后一次约 47.6 s 完成。
   - 45–47 s 现在仍在持续折叠（45.0 s 约 7/11，46.0 s 约 9/11，47.0 s 约 10/11）。
2. **左侧线路淡入淡出**
   - 每幕结束前 8 帧淡出到面板底色，下一幕开始后 8 帧淡入；不再双图叠加。
   - 幕标题与算法名同步淡出/淡入。
3. **右侧卡片亮度衔接**
   - 当前卡 0.32→1，上一张 1→0.72，更早的保持 0.72，未到的保持 0.32。
   - 边框、背景、标题色、算法色随 focus 插值；不再瞬间跳变。
   - 已看过的卡片保持 0.72（旧版 0.55），方便回头阅读。
4. **底部公式卡**：在幕切换时降到 0.4 再升回 1，避免文字硬切。

### 1.3 文字建议

- **采纳 1**：S3 主标题 `103 次` → `103 个`。
- **采纳 2**：S3 副标题改为 `由路由器实测插入的 SWAP 数与估计保真度，再打分排名`。
- **采纳 3**：S6 副标题改为 `平台基没有 SWAP 门，必须把 SWAP 展开为 CZ + 1Q，于是同一条线路出现两种表示`。
- **不采纳 4**：不增加“287 规格 / 176 真机”的口径说明。

### 1.4 口语字幕 + 两版输出

- 删除 S1/S2/S3/S7 的总结式大字幕。
- 新增全片统一口语字幕 22 句，底部居中，31 px，淡入 6 帧/淡出 8 帧。
- 字幕文本按“创作者面对面讲解”重写，不再是页面注释。
- 输出两版：
  - `qtrial-case-draft-v13-silent.mp4`：无音轨。
  - `qtrial-case-draft-v13-voice.mp4`：BGM + 旁白。
- 旁白选择：Windows SAPI5 `Microsoft Huihui Desktop`（本机唯一中文离线声音），Rate 4。
- BGM 选择：`House Vibez` — Lily J — Mixkit Stock Music Free License。

## 2. 旁白时间轴

见 `docs/case_film_text_v13.md` 第 0 节。22 句覆盖 S1–S7，最晚一句 83.4–84.9 s。

- 每句音频由 `tools/tts_narration.ps1` 生成为 22.05 kHz / 16-bit / mono WAV。
- `tools/make_narration.mjs` 测量时长，生成 `src/data/narration.json`（from / dur / text / audio）。
- 字幕与音频共用同一 `from` / `dur`，逐句对齐；相邻句无语音重叠。
- 旁白增益 1.25，BGM 约 0.035；实测旁白段 RMS 约 0.035–0.072，BGM 空段约 0.008–0.013，约 13–15 dB 差。
- 峰值 0.852，无削波。

## 3. 时间线（未变）

| # | 场景 | 本地帧 | 全局约 | 时长 |
|---|---|---|---|---|
| 01 | 逻辑线路 | 0–270 | 0–9.0 s | 9.0 s |
| 02 | 交互图 + 拓扑 | 0–210 | 8.3–15.3 s | 7.0 s |
| 03 | 布局竞技 | 0–290 | 14.5–24.2 s | 9.7 s |
| 04 | 路由同步 | 0–480 | 23.5–39.5 s | 16.0 s |
| 05 | 后处理工作台 | 0–780 | 38.7–64.7 s | 26.0 s |
| 06 | 最终线路 | 0–220 | 64.0–71.3 s | 7.3 s |
| 07 | 真机验证 | 0–360 | 70.6–82.6 s | 12.0 s |
| — | 字标收尾 | 0–90 | 81.9–84.9 s | 3.0 s |

## 4. 自查执行

- `npx tsc --noEmit` 通过。
- 无声版：2546 帧 / 84.8667 s / 8.75 MB / 无音轨。
- voice 版：2546 帧 / 84.8850 s / 12.0 MB / AAC 音轨。
- 抽查成片帧：3.0 s（S1 小门）、20.6 s（S3 定稿文字）、47.9/48.0/48.2 s（A1→A2 淡出淡入）、54.5 s（A3 对消）、65.5 s（S6）、80.0 s（S7 审计字幕）。
- 音频逐句 RMS 检查：22 句全部有语音能量；BGM 空段有稳定低电平；无削波。
- 未发现字幕溢出、双图叠加、黑帧或场景错位。

## 5. 仍存在的限制

1. 旁白是 Windows SAPI Huihui，音色偏“播报”，不如云端神经 TTS 自然；本机无网络，无法换 edge-tts。
2. 540p 评审版里 S5 单门标签、S7 表头仍偏小；1080p 终渲会改善。
3. A1 仍是 13 层局部窗口，不是 86 层全景；底部压缩条提供全局位置。
4. A3 的邻居门只取层 1–8。
5. B 仍是机制示意。
6. 未做 1080p 终渲、未嵌入 PPT、未做独立终检。
7. BGM 许可为 Mixkit 免署名可商用，发布前建议再核对。

## 6. 复现

```bash
cd promo_video
npx tsc --noEmit
# 若改旁白文本：
powershell -NoProfile -ExecutionPolicy Bypass -File tools/tts_narration.ps1
node tools/make_narration.mjs
# 两版：
npx remotion render src/index.ts QTrialCase out/qtrial-case-draft-v13-silent.mp4 --scale=0.5 --concurrency=4
npx remotion render src/index.ts QTrialCaseVoice out/qtrial-case-draft-v13-voice.mp4 --scale=0.5 --concurrency=4
```
