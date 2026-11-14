# 案例演示片 v13 音频来源与授权

## BGM

| 项目 | 内容 |
|---|---|
| 曲名 | House Vibez |
| 艺术家 | Lily J |
| 来源 | Mixkit Stock Music |
| 许可 | Mixkit Stock Music Free License（免署名、可商用；发布前建议再核对一次） |
| 原始链接 | https://assets.mixkit.co/music/745/745.mp3 |
| 工程文件 | `promo_video/public/audio/bgm/house-vibez.mp3` |
| 使用方式 | 全片循环，音量约 0.035，首尾 45/60 帧淡入淡出 |

## 旁白

| 项目 | 内容 |
|---|---|
| 引擎 | Windows SAPI5（离线） |
| 声音 | Microsoft Huihui Desktop（zh-CN，女声） |
| 语速 | Rate 4 |
| 生成脚本 | `promo_video/tools/tts_narration.ps1` |
| 文本源 | `promo_video/tools/narration_src.json` |
| 音频 | `promo_video/public/audio/voice/n01.wav`–`n22.wav` |
| 说明 | 本机无网络，未使用云端 TTS 或声音克隆；全部为本地合成。 |

## 混音

- 旁白增益 1.25；BGM 约 0.035，旁白电平约高于 BGM 13–15 dB。
- 成片音轨：AAC，随视频封装。

## 交付提醒

- Mixkit 曲目为免署名可商用，但若竞赛方要求逐项列出素材来源，请保留本文件。
- 若后续替换 BGM，请同步更新本文件的曲名、艺术家、链接与许可。
