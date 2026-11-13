# 会话恢复点：QFT / graphstate 短板归因研究（2026-11-13）

> **已完成的后续（2026-11-13 晚）**：本文件的假设已逐条验证/证伪，
> 修复与全量重跑已落地。详见
> [`docs/qft_graphstate_analysis.md`](qft_graphstate_analysis.md)：
> QFT = 预路由消化（非 cp 分解；cp=1CZ+3rz 已证明不成立），
> graphstate = 移除 max_active + Warnsdorff；`pre_cancel` 已默认开，
> 584 条重跑 large 均值 0.0028→0.0072、对 O3 胜场 38→118/199。

## 当前状态

- `main` HEAD `b7c85d4`（large 门数归因扫描 + 报告边界），已推送；本地与远端同步。
- 无活跃后台任务。
- 本轮已完成：v8 全量/合法性、真机 176 验证、隐私清理、报告实验重跑、初赛风格图表重生成、报告书 docx 重建、large 门数归因扫描。
- 外部 agent 仍有一批未提交改动（`scripts/thesis/content*.json`、`scripts/submission_*`、`web/` 等）；`content3.json` 已被我们追加了一段 QFT/graphstate 归因文字，但该文件仍属于外部改动，未单独提交。
- 报告书 `作品报告书.docx` 已用新图与新文字重建；备份 `tables/_archive/作品报告书_20261112_before_v8figs.docx`。

## 已有证据（large 门数归因，24 条抽样）

- 总体中位：1Q 0.99×、2Q 1.20×、深度 1.21×、log 保真度 1.08×（v8/O3）。
- QAOA（7）：0.94× / 1.16× / logf 1.02×——基本相当。
- QV（6）：0.98× / 1.02× / 1.01×——基本相当。
- QFT（6）：**1.74× / 2.23× / 1.92×**——主要短板。
- graphstate（1）：2.87× / **6.52×** / 2.52×。
- 数据：`tables/gate_attribution/large_rows.jsonl`、`summary.md`；台账 #22；报告书“技术边界”段。

## 初步机制假设（待验证）

### QFT

1. 解析器 `cp(λ)` 当前展开为 **2 个 CX**：
   `u1(λ/2) c; cx(c,t); u1(−λ/2) t; cx(c,t); u1(λ/2) t`
   （`qtrail/pure/qasm.py` 219-227 行）。
2. 但 controlled-phase 可以用 **1 个 CZ + 3 个 rz** 实现：
   `cp(λ) = rz(λ/2)_c · rz(λ/2)_t · rz(π)_t · CZ`（模全局相位），
   即 2Q 门数减半、1Q 也可减少。若成立，QFT 的 2.23× 可降到接近 1×。
3. 需检查 qiskit O3 的 QFT 产物：确认它是否用 1 CZ/cp，以及 1Q 数量。
4. 还需检查 A2/VW 合成是否把 QFT 的 CZ 块再次展开，增加 1Q/2Q。

### graphstate

1. `graphstate_100` 是 100 比特、100 个 H + 100 个 CX 的图态线路；
   v8 产物 2Q=646、depth=184，qiskit O3 2Q=100、depth=15。
2. 差距像 **布局/嵌入**：qiskit 用 SabreLayout + VF2PostLayout 找到近乎
   0-SWAP 的嵌入；我们 VF2++ 在 `relabel_vf2.py` 对 `active > 40` 直接
   跳过（`max_active=40`），大图态线路退回 RL/启发式布局 + 路由，产生大量
   2Q。
3. 待验证：graphstate 的 100 条边能否嵌入 105 比特设备（度/子图同构）；
   若能，VF2++ 应能在 100 节点上跑（或需要分块/缩放），这是最直接的改进。
4. 若不可嵌入，需比较 qiskit 的布局/路由为何只用 100 个 2Q（是否用了
   SWAP 后抵消、或图态结构的特殊合成）。

## 下一步研究计划（压缩后执行）

1. **QFT cp 分解 A/B（最高优先）**
   - 在 `qasm.py` 增加 `cp → 1 CZ + 3 rz` 的等价展开（先写成可选分支/脚本，
     不改默认）；
   - 对 qft_20/50/100 比较 1Q/2Q/depth/logf，语义用紧凑子空间/态矢量验证；
   - 判据：2Q 数下降 ≥40%、logf 不劣化、零非法；通过后再考虑默认化。
2. **graphstate 嵌入能力**
   - 读 100 条边构图，检查最大度/是否可嵌入 105 比特设备；
   - 把 VF2++ 的 `max_active` 提到 105 做单条试验（时间预算 8s/2M calls），
     比较布局与 2Q 数；若超时，研究分块/启发式精确嵌入；
   - 对照 qiskit O3 的 2Q=100、depth=15，确认它是嵌入还是合成。
3. **横向确认**
   - 对 dnn/ising/swap 族重复门数分解，判断是否同一“稀疏线路固定开销”机制；
   - 统计 A2/VW 块合成对 QFT/graphstate 的 1Q 贡献。
4. **报告**
   - 研究完成后写专门文件（建议 `docs/qft_graphstate_analysis.md`），含：
     数学推导、逐 pass 门数、A/B 数据、改进方案与预期、风险与不采用的理由；
   - 同步 `tables/gate_attribution/`、台账、报告书相关段落（只改文字/数据，
     不夸大）。

## 关键文件

- 解析器：`qtrail/pure/qasm.py`（cp 展开）
- 布局/嵌入：`qtrail/pure/relabel_vf2.py`（max_active=40）、`qtrail/pure/vf2pp.py`
- 合成：`qtrail/pure/synthesis.py`、`qtrail/pure/weyl.py`
- 指标：`qtrail/pure/metrics.py`（乘积模型）
- 数据：`tables/final_report_v8/`、`tables/gate_attribution/`
- 报告：`docs/competition_report.md`、`scripts/thesis/content3.json`、`作品报告书.docx`
