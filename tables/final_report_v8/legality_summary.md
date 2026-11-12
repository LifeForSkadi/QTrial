# v8 全量合法性重扫（2026-11-13）

脚本：`scripts/verify_all_legality.py --out-dir tables/final_report_v8`
（11 分片，生产 mapper v5、routing_seeds=16、timeout 900s、修复后解析器）。

## 结果

| 项 | 数值 |
|---|---|
| 可解析线路（扫描覆盖） | **584 / 587** |
| 合法 | **581** |
| 非法（门集/相邻/禁用/越界/swap 残留/measure） | **0** |
| 布局非单射 | **0** |
| 超时（未完成，非非法） | **3** |

超时线路：`bp_feynman_hwb11`、`bp_feynman_hwb12`、
`bp_vqe_uccsd_n28_vqe_uccsd_n28`（与 v8 主跑同口径的已知最难线路；
主跑另有 transpiled 变体也超时）。

## 与 587 清单的差额（3 条）

`bp_bb84_n8_bb84_n8`、`bp_bb84_n8_bb84_n8_transpiled`、
`bp_seca_n11_seca_n11_transpiled` 含 **mid-circuit measure**
（测量后同一比特再作用量子门）。独立审计 P2-3 修复后，解析器显式拒绝
此类输入（此前 `Circuit.measures` 在路由后统一末尾发射，语义不等价）。
这 3 条在 v8 主跑时被静默“测量后移”，其结果不可信，已在
`*_fidelity_rows*.jsonl` 中标记为
`qt_error="unsupported: mid-circuit measure"` 并从统计中剔除。

## v8 分层完成/错误（标记后）

| 层 | 完成 | 错误 |
|---|---|---|
| ≤10 | 127 | 2（bb84 两条，mid-circuit） |
| 11-50 | 254 | 5（4 条 900s 超时 + seca mid-circuit） |
| 51-105 | 199 | 0 |
| 合计 | **580** | **7** |

## 结论

v8 可映射且在时限内完成的产物**未发现任何真机合法性违规**（含
final_layout 单射性）。合法性的剩余边界是：3 条超时未完成、3 条
mid-circuit measure 输入被显式拒绝（不是产物非法，而是无法在本管线
语义下安全映射）。
