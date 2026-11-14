# QTrail 决赛提交包

2026 年度中国青年科技创新「揭榜挂帅」擂台赛 · 量子+AI 双向赋能榜题
（天衍-287 量子线路映射优化）

## 包内结构

| 目录 | 内容 | 说明 |
|---|---|---|
| `01_源码/QTrial/` | 完整交付源码 | 交付管线 `qtrail/`、配置、预训练权重、示例、测试、关键实验脚本、数据与图件 |
| `02_报告书/` | `作品报告书_决赛.docx` / `.pdf` | 61 页，A4；打印前请在封面填写单位、姓名、手机号 |
| `03_PPT/` | `作品展示PPT_决赛.pptx` | 9 页，6 分钟节奏；第 6 页内嵌演示视频，含逐页演讲者备注 |
| `04_演示视频/` | `演示视频.mp4` | 75 秒，1920×1080；PPT 内嵌版的单独备份 |
| `05_讲稿/` | `讲稿_6分钟.md` | 逐页口播稿与时间控制表 |

## 源码快速验证

```bash
cd 01_源码/QTrial
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m qtrail.pure_cli examples/ghz10.qasm
```

期望输出包含 `swap_count`、`depth`、`est_fidelity`、`wall`，并在 `out/` 下生成
`ghz10_mapped.qasm`、`ghz10.qcis`、`ghz10_metrics.json`。

启动 Web 演示：

```bash
python -m qtrail.web
# 浏览器打开 http://127.0.0.1:8000
```

## 正确性与复现

- 全量测试：`python -m pytest tests/ -q`（232 项通过 + 2 项已知环境性断言：
  `test_fit_block_torch_exact` 容差、`test_platform_verifier_needs_key` 本机密钥存在性）
- 发布前审计：`python scripts/audit_v8_release.py`（31 条抽样：真机合法 31/31、
  measure 语义 31/31、端到端语义保真度 1.000000）
- 分层数据：`tables/submission/summary.md`（583 条完成线路的均值/截尾/中位三口径，
  逐线路原始数据见 `tables/submission/per_circuit.jsonl`）
- 基线逐线路原始行：`tables/final_report/{small,medium,large}_fidelity_rows.jsonl`
- 机制实验数据：`tables/defect_sweep/`、`tables/cross_topo/`、
  `tables/compare_models/layout_contribution_*.jsonl`
- 运行时间测试集：`data/scaling/`；图件：`tables/report_figures/thesis/`

## 口径说明

- **零 qiskit 的边界**：映射内核与 CLI 交付管线（`python -m qtrail.pure_cli`）
  全链路不导入 qiskit；Web 演示层（`python -m qtrail.web`）与对照实验使用 qiskit
  做 QASM 解析与基线编译。
- 估计保真度为逐门乘积代理；51-105 比特层乘积下溢，跨方法比较以 2Q 深度/深度为准。
- 587 条基准中 11-50 层 4 条线路在 900 秒预算内超时，统计基于 583 条完成线路。

## 复现实验（对应报告 5.4.3–5.4.5）

```bash
# 分层主表（基线读 tables/final_report，主结果可重建）
python scripts/aggregate_v8.py
python scripts/submission_aggregate.py

# 机制实验与图件
python scripts/defect_sweep.py
python scripts/cross_topo_bench.py
python scripts/bench_layout_contribution.py --bench benchpress --limit 30
python scripts/make_report_figures.py
python scripts/make_runtime_compare.py
```

说明：Benchpress / MQTBench / QUEKO 原始数据集体积较大（数百 MB），未随包分发，
可按报告 5.1 节来源获取；包内 `tables/final_report/` 已含基线逐线路结果，可直接复核主表。

## 数据与权重

- `checkpoints/tianyan-287_gat_combined_calib_dep0.1_t0.5_c0.05_best.pt`：天衍-287 生产权重
- `checkpoints/multi7topo_gat_combined_calib_dep0.1_t0.5_best.pt`：多拓扑迁移权重
- `checkpoints/demo_laurel/*.pt`：报告 5.4.3 引用的 LAUREL 89 轮收敛权重
- `examples/`：可直接运行的 QASM 示例

## 联系与提交

报到时提交：U 盘（本包 + 答辩材料）+ 报告书装订本一式六份（A4 双面，≤80g，勿过度装裱）。
