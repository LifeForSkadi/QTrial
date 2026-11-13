# 真机 v9 实验记录（天衍-176）

## 目的

展示 QTrial v8 管线在**真实超导硬件**上的可执行性与保真度模型校验
（非竞赛主表口径；竞赛主表仍是天衍-287 规格的模拟基准）。

## 设置

- 设备：tianyan176（66 比特、94 条耦合器、2 个禁用比特；提交时
  空闲/可运行）；shots = 2000 / 作业。
- v8 臂（7 个）：`x5`、`ghz5`、`ghz10`、`ghz15`、`qft5`、`qft6`、
  `qaoa12`；Qiskit 感知 O3 臂（3 个）：`qft5`、`qft6`、`qaoa12`。
- 产物：`out/real_machine_v9/tianyan176/`（QCIS + 指标 JSON）；
  qid 见 `jobs.json`。

## 分析口径

- 理想分布：qiskit `Statevector`（逻辑序）；
- 经典保真度 = `(Σ √(p·q))²`（Bhattacharyya 平方）；
- TVD；峰值态（支撑 ≤4）命中率；
- GHZ 规模扩展（5/10/15）实测命中 vs 乘积模型预测。

## 状态

- **2026-11-13 约 13:20 提交全部 10 个作业；轮询至 16:12 仍无结果
  返回**。期间轮询器多次遇到瞬时 `CqlibRequestError`（疑似平台限流），
  16:12 后单独查询一次成功：`experimentResultModelList: []`——作业
  **仍在排队/运行，不是失败**。已启动低频长轮询（180s 间隔，3h），
  结果返回后自动落盘。未伪造任何数据。
- 若后续返回：把 counts 文件放入本目录，运行

```bash
python scripts/real_machine_v9_analyze.py x5 ghz5 ghz10 ghz15 qft5 qft6 qaoa12
python scripts/make_real_machine_v9_figure.py
```

生成 `summary.md`、`rows.json` 与 `tables/report_figures/图13_真机对比.png`。

## qid

| 作业 | qid |
|---|---|
| x5_v8 | 2121108697134555138 |
| ghz5_v8 | 2121109930360594433 |
| ghz10_v8 | 2121109938258468865 |
| ghz15_v8 | 2121109945908879362 |
| qft5_v8 | 2121109953341186049 |
| qft6_v8 | 2121109960375033858 |
| qaoa12_v8 | 2121109968415514625 |
| qft5_o3 | 2121110028677910530 |
| qft6_o3 | 2121110036543594497 |
| qaoa12_o3 | 2121110044219170818 |
