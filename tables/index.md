# tables/ 目录说明

本目录存放全部实验结果。每个文件/子目录的用途、数据口径、生成脚本如下。
**口径总览**：项目有两条管线——qiskit 支撑版（内部参照，路由用 qiskit
SabreSwap）与 pure 版（竞赛交付，零 qiskit，自研路由器 + 后处理栈）。
带 `qt` 列的原始数据（jsonl）记录了每条线路的逐项指标，是所有汇总表的来源。

## 最终交付数据

| 路径 | 内容 | 生成脚本 |
|---|---|---|
| `final_report/` | **最终分层性能报告 v2**：完整 Benchpress ≤105 比特 + MQTBench + QUEKO（≈590 条），QTrial pure（fidelity 规则）vs 盲目O1/O3、感知O1/O3、pytket；均值/截尾(去10%)/中位数；逐线路原始数据 jsonl | `scripts/final_report_bench.py` |
| `report.md` | qiskit 支撑版总报告（混合竞技 v2 时代的完整结果汇总） | `scripts/make_tables.py` 等 |
| `research_contributions.md` | 研究贡献清单（八项贡献 + 诚实负结果记录） | 手工撰写 |
| `reference.json` | 复现基准（10 线路 × 3 种子参考值） | `scripts/reproduce.py` |
| `speed_large_vs_o3.jsonl` + `.summary.txt` | 大比特线路（51-105q × 18 条）QTrial pure vs 感知O3 单线路耗时对比原始数据（增量路由 + Numba 内核版） | `scripts/speed_large_vs_o3.py` |

## 对比实验证据（qiskit 支撑版口径，不受 pure 版修复影响）

| 路径 | 内容 | 生成脚本 |
|---|---|---|
| `cqlib_inject/` | RL 布局注入天衍平台 Cqlib（MCTS）管线 vs Cqlib 原生 / tket / SABRE 分层对比（含超时部分结果） | `scripts/cqlib_inject_bench.py` |
| `defect_sweep/` | 缺陷强度扫描（×0/×2/×3/×5）：QTrial 保真度对缺陷强度不变 vs 盲目 O1 退化 1.56×——噪声规避能力证据 | `scripts/defect_sweep.py` |
| `cross_topo/` | 跨拓扑泛化（6×6 / heavy-hex-115 / Sycamore-53）：同一权重跨设备零重训 | `scripts/cross_topo_bench.py` |
| `noise_aware/` | 噪声感知 O1/O3 基线实验（qiskit Target 注入误差）分层原始数据 | `scripts/noise_aware_o1_bench.py` |
| `paper_8x8/` | CO-MAP 论文同设置对照（8×8 网格 + MQTBench n=15 + Queko-20） | `scripts/paper_parity.py` |
| `target_post/` | Target 后处理管线实验（RL 布局 + 噪声感知 O1/O3 预设，置换吸收 92→0 SWAP） | `scripts/target_post_bench.py` |
| `goat/` | 5 编译器（QTrial/O1/O3/tket/MQT QMAP）同台对比 + GOAT（calibration-aware RL）调研结论 | `scripts/benchmark_goat.py` |

## 单文件记录

| 路径 | 内容 |
|---|---|
| `paper_parity_analysis.md` | 移动基线分析：今日 SABRE 比论文时代强 2.6×，论文 65-95% 减少率无法重现的原因 |
| `two_stage_analysis.md` | 两阶段（布局+路由）误差分解分析 |
| `mqtbench_n15.{md,csv}` + `_rows.jsonl` + `_summary.json` | MQTBench 15 比特论文对齐评测 |
| `queko_BIGD.{md,csv}` + `_rows.jsonl` + `_summary.json` | QUEKO BIGD（20 比特 × 360）全量评测 |
| `queko_BNTF.{md,csv}` + `_rows.jsonl` + `_summary.json` | QUEKO BNTF（16 比特 × 180）全量评测 |
| `ab_postfid.json` | 后处理感知保真度奖励微调 A/B（qiskit 版） |
| `ab_pure_models.json` | pure 管线上三模型 A/B 门禁（c0.05 vs postfid_ft/mt） |
| `ablation_noise.json` | 噪声感知消融（布局级 2Q 误差 -5.4% 等） |

## 历史与冗余数据

已被后续实验取代或仅用于中间调试的历史结果，已备份至项目外
`f:\Study\Quantum Computing\信安竞赛\_QTrial_archive\tables_archive\`：
`pure_stratified/`（早期 pure 分层，deps 污染口径作废）、
`goat_stratified{,_pure}/`（分层三规则，同上）、`stratified/`（早期 qiskit 分层）、
`final_stratified/`、`hybrid_v2/`、`queko_noise/`、`rl_lexiroute.jsonl`。
