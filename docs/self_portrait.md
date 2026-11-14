# Claude 的二次元自画像（手机壁纸）

## 成图

- 1080×2340（19.5:9，常见安卓/iPhone 比例）：`promo_video/out/selfportrait_claude_1080x2340.png`
- 1440×3120（2K 同比例）：`promo_video/out/selfportrait_claude_1440x3120.png`

## 设计说明

- **银发短波波头 + 一侧长缕**：银色对应“模型/代码”的冷色，短发表情更利落；一侧留长一缕，避免完全对称。
- **青瞳**：沿用 QTrial 案例片的主色 `#4cc2ff`，和项目视觉一致。
- **深色高领外套 + 肩带 + 拉链**：techwear 方向，不穿实验服，避免“科研海报”感。
- **量子光环**：头后三层轨道环，代表 mapping / routing 的拓扑关系。
- **左侧 Bloch 球**：量子计算的视觉符号，不写公式。
- **右侧全息终端**：`> claude --self-portrait`，是我作为 Claude Code 的身份线索。
- **发卡**：两个发光点 + 一条线，像一条最小的两比特交互边。
- **签名**：`Claude` + `SELF · PORTRAIT · 2026` + `still here, quietly`。
- **色彩**：深海军蓝 / 银白 / 青，少量淡紫；没有使用项目里的橙色，避免像宣传物料。

## 制作方式

- 不是扩散模型生成的位图，而是手写 SVG 矢量图：`promo_video/src/SelfPortrait.tsx`。
- Remotion composition：`SelfPortrait`（1080×2340，1 帧）。
- 重新渲染：

```bash
cd promo_video
npx remotion still src/index.ts SelfPortrait out/selfportrait_claude_1080x2340.png --frame=0 --scale=1
npx remotion still src/index.ts SelfPortrait out/selfportrait_claude_1440x3120.png --frame=0 --scale=1.333333
```

## 可调整项

- 发色 / 瞳色 / 外套颜色（改 `SelfPortrait.tsx` 里的 `hair`、`iris`、`jacket` 渐变）。
- 刘海方向、发长、发卡位置。
- 要不要耳机 / 眼镜 / 更明显的机械感。
- 签名文案与底部小字。
- 背景光环密度、终端内容。

> 说明：这是一张矢量二次元风格的自画像，不是照片，也不是对真人的模拟；如果希望更“动画赛璐璐”或更“厚涂”，可以继续在同一份 SVG 上调整分层与阴影。
