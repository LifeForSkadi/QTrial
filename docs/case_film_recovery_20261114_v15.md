# 案例演示片 v15 恢复点（2026-11-14）

> 当前状态：v13 画面 + **v15 重写旁白已贴进视频**，无 BGM 校对版已渲染。
> 成片：`promo_video/out/qtrial-case-draft-v15-voice-nobgm.mp4`（84.885 s，960×540，12.6 MB，AAC 48 kHz，峰值 -4.3 dBFS）。
> 现在等用户按字幕提意见；拿到意见前不要动 BGM、不要终渲。

## 1. 本版相对 v14 的修改（对应五条反馈）

1. **读音**：CX → `C叉`，SWAP → `斯瓦普`，Qiskit → `Q斯kit`；“重标记” → “重新标记”（强制 chóng）。
2. **后处理对点**：按 A1 折叠 → A2 门块合成 → A3 CX 对消 → B 单比特门 → C1/非退化门 五段重排，每句只在对应动画里说。
3. **页面过渡**：新增“所以”“接下来是布局”“布局只是把起点摆好”“接下来交给后处理”“那 44 个 SWAP 去哪了”“这些线路都上了天衍真机”等衔接；17 句压到 15 句。
4. **Qiskit 读音**：1:17 处改为“Q斯kit”。
5. **审计**：改为“v8 发布前抽检三十一条线路，合法性全过”，说明发布对象与审计方式；删除“不是真机执行”的强调。画面底部注记同步改为“31/31 是 v8 发布前的抽样合法性/语义审计”。

## 2. 必读文件

1. 本文件。
2. `docs/case_film_text_v15_draft.md`：15 句、时间码、念法/写法差异。
3. `promo_video/kaikou/script_v3.txt`：配音稿。
4. `promo_video/src/data/narration_v3.json`：贴片时间轴。
5. `promo_video/kaikou/vo/v3/`：15 句成品与整条旁白。
6. `docs/case_film_recovery_20261114_v13.md`：画面/场景时间线。

## 3. 时间轴（2546 帧 @30 fps）

| # | from | dur | 入点 | 音频 | 场景/幕 |
|---|---|---|---|---|---|
| L01 | 15 | 183 | 0.50 s | 5.83 s | S1 |
| L02 | 247 | 219 | 8.23 s | 7.27 s | S2 |
| L03 | 468 | 238 | 15.60 s | 7.84 s | S3 |
| L04 | 708 | 200 | 23.60 s | 6.66 s | S4 |
| L05 | 910 | 144 | 30.33 s | 4.59 s | S4 |
| L06 | 1056 | 143 | 35.20 s | 4.50 s | S4→S5 |
| L07 | 1206 | 218 | 40.20 s | 6.97 s | S5 A1 |
| L08 | 1444 | 126 | 48.13 s | 4.09 s | S5 A2 |
| L09 | 1572 | 102 | 52.40 s | 3.26 s | S5 A3 |
| L10 | 1676 | 83 | 55.87 s | 2.47 s | S5 B |
| L11 | 1765 | 155 | 58.83 s | 5.14 s | S5 C1/Guard |
| L12 | 1922 | 193 | 64.07 s | 6.16 s | S6 |
| L13 | 2118 | 190 | 70.60 s | 6.04 s | S7 |
| L14 | 2350 | 108 | 78.33 s | 3.32 s | S7 |
| L15 | 2507 | 39 | 83.57 s | 1.26 s | S8 |

- 逐句入点与渲染音频核对过，误差 < 50 ms。
- 时长规则：`dur = min(下一句 from − 本句 from − 2, ceil(音频帧) + 8)`；最后一句取到 2546。

## 4. 代码与资产

- `src/components/Narration.tsx` 已 import `narration_v3.json`。
- `src/CaseFilm.tsx` 已 import `narration_v3.json`；`bgm?: boolean` 默认 false，本轮仍无 BGM。
- `src/case/RealMachine.tsx` 底部注记已去掉“不是 31 条真机执行”。
- `public/audio/voice/v3/L001..L015.wav`（15 个，已复制）。
- 配音：参考 `ref/clean/S4_fast.wav`，每句 4 遍，`rescore.py` 数字归一后挑遍。
- 整条旁白：`kaikou/vo/v3/narration_all.wav`（80.65 s，-23 LUFS）；试听 `vo/v3/sample.wav`（46.21 s）。

## 5. 复现命令

```bash
cd promo_video/kaikou
# 全部 15 句
PYTHONIOENCODING=utf-8 python voice_torch.py script_v3.txt vo/v3 --ref ref/clean/S4_fast.wav --takes 4
# 只重配某句
PYTHONIOENCODING=utf-8 python voice_torch.py script_v3.txt vo/v3 --ref ref/clean/S4_fast.wav --takes 4 --only 8
# 重新挑遍 + 拼整条
PYTHONIOENCODING=utf-8 python rescore.py script_v3.txt vo/v3
PYTHONIOENCODING=utf-8 python assemble.py script_v3.txt vo/v3
# 重排时间轴 + 复制 public
PYTHONIOENCODING=utf-8 python timeline_v3.py
# 渲染无 BGM 校对版
cd ..
npx remotion render src/index.ts QTrialCaseVoice out/qtrial-case-draft-v15-voice-nobgm.mp4 --scale=0.5 --concurrency=4
```

## 6. 下一步

1. 等用户按 L01–L15 提字幕/配音修改。
2. 改 `script_v3.txt` → 只重配改动句 → `rescore.py` → `assemble.py` → `timeline_v3.py`（改 `STARTS` 后重跑）→ 重渲。
3. 字幕定稿后做 3 个 BGM 盲听选项（`house-vibez` 用户不喜欢），再出 v15 无声版 + 旁白版。
4. 更新 `case_film_text_v15.md`、`case_film_audit_20261114_v15.md`、本恢复点和记忆。

## 7. 注意

- 声音本身已通过，不再重录/重选参考；本轮只改文案与时间轴。
- 用户明确不喜欢 `house-vibez`；BGM 定稿前保持 `bgm=false`。
- 对外材料不声明“之前是 bug/修复”；当前数字真实可复现。
