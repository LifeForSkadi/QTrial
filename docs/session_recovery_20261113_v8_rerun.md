# 会话恢复点：v8 全量重跑 + 真机对比实验（2026-11-13 晚）

## 状态速览

- `main` HEAD **aa2ce69**，已推送；`origin/main` 同步。
- 未提交变更 32 项：外部 agent 的 thesis/web/submission 文件（勿动）+
  本轮真机脚本/合法性脚本的小改动（见下）。
- 后台任务（压缩后可能仍存活，进程消失则按本文件续跑）：
  1. **真机 v9 轮询**：`scripts/real_machine_v9_batch.py`（逐条 30s 轮询）
     或 `scripts/real_machine_v9_poll.py`；`tables/real_machine_v9/jobs.json`
     保存 10 个 qid，可断点续跑。
  2. **合法性复验**：`bash scripts/run_legality_v8_parallel.sh`
     （6 分片，输出 `tables/final_report_v8/legality_rows*.jsonl`）。

## 已完成（本轮）

1. **QFT/graphstate 归因修复**（提交 9300435 / ff4bc70 / 095f5be）：
   - `pre_cancel` 预路由消化（默认已开，映射器 `PureMapper.pre_cancel`）；
   - VF2++ 移除 `max_active=40`、Warnsdorff 排序；
   - 分析文件 `docs/qft_graphstate_analysis.md`、台账 #23。
2. **v8 全量重跑**（提交 991e35d）：584 条（small 127 / medium 258 /
   large 199），**0 错误**。新口径：
   - large 均值保真度 **0.0028→0.0072**；对感知 O3 胜场 **38→118/199**；
     深度 8207→7242、SWAP 3872→3286；
   - medium 0.1319→0.1446；small 0.7337；
   - QFT：N055 2Q 3333 / logf −21.68（O3 −25.51）、N080 5826/−39.59、
     N097 7660/−62.05；graphstate_100 100/−2.10；ghz_n78 77/−1.91；
     ising_n98 194/−3.68。
   - 修复输入级 `id` 门崩溃：`post_v2._oneq_matrix` 补单位阵。
3. **图表全部重生成**（`scripts/make_report_figures.py` + runtime 5.7/5.8），
   报告书 docx 已重建（备份 `tables/_archive/作品报告书_20261113_before_v8rerun.docx`）。
4. **reproduce.py PASS**（pre_cancel 默认开后基准未漂移）。
5. 测试 233 passed（3 个既有环境性失败：CLI GBK / torch 容差 / 真机 key）。
6. 真机 v9 实验**已提交 10 个作业**（tianyan176，2000 shots）：
   - v8：x5, ghz5, ghz10, ghz15, qft5, qft6, qaoa12
   - O3：qft5, qft6, qaoa12
   - QCIS 已生成于 `out/real_machine_v9/tianyan176/`；
     qid 在 `tables/real_machine_v9/jobs.json`。**尚未返回**（队列慢）。

## 待办（压缩后继续）

1. **等真机结果**：轮询 jobs.json；结果落盘后运行
   `python scripts/real_machine_v9_analyze.py x5 ghz5 ghz10 ghz15 qft5 qft6 qaoa12`
   → `tables/real_machine_v9/summary.md` + `rows.json`；
   再运行 `python scripts/make_real_machine_v9_figure.py`
   → `tables/report_figures/图13_真机对比.png`（2×2：实测保真度 /
   GHZ 规模扩展 / 2Q 门数 / 预测-实测校验）。
   若队列仍无返回：保留作业与脚本，报告"已提交待返回"，不要伪造数据。
2. **合法性复验**：`run_legality_v8_parallel.sh` 完成后检查
   `legality_rows*.jsonl` 的 legal 列（目标 0 illegal）；把
   `scripts/verify_all_legality.py`（dev 判定修复）与
   `scripts/run_legality_v8_parallel.sh` 提交。
3. **收尾提交**：真机分析 summary/rows/图13 + 合法性脚本/结果。
4. 可选：把图13 与真机结论补进报告书（`scripts/thesis/build.py` 重跑），
   但 `content*.json` 属外部 agent 未提交改动，谨慎。

## 关键文件

- 分析：`docs/qft_graphstate_analysis.md`、台账 #23
- 基准：`tables/final_report_v8/{report.md,summary_v8.md,*_fidelity_rows.jsonl}`
- 图：`tables/report_figures/`（图3/4/8、thesis/图5.1-5.10）
- 真机：`scripts/real_machine_v9_{map,submit,batch,poll,analyze}.py`、
  `scripts/make_real_machine_v9_figure.py`、`tables/real_machine_v9/`、
  `out/real_machine_v9/tianyan176/`
- 合法性：`scripts/verify_all_legality.py`、`scripts/run_legality_v8_parallel.sh`
- 基础设施：`scripts/final_report_bench_v8.py`（--dev + 桶缓存）、
  `scripts/run_bench_v8_parallel.sh`
