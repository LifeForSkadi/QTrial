# 《作品报告书_决赛》投稿前预审报告

评审日期：2026-11-12。评审对象：`作品报告书_决赛.pdf`（61 页，A4）及其配套提交包
`决赛提交/`（源码、权重、数据、PPT、视频）。

## Review setup

- **输入范围**：全文 61 页（摘要、第 1–5 章、结论、附录、参考文献）；配套源码与数据包；
  未提供原始实验日志（仅有逐线路 JSONL 与汇总表）。
- **评审边界**：以稿件自身声称与包内可核验材料为准；不评价竞赛评审规则、不预测获奖；
  未对真机结果做独立复测。
- **核心命题**：面向天衍-287 的量子线路映射，RL 布局 + 自研路由/后处理 + 噪声感知，
  在保真度上超过 pytket 与盲目基线、小规模接近 qiskit 感知 O3，且输出真机可执行。
- **可见证据**：583 条完成线路分层表（均值/截尾/中位）、逐线路 JSONL、发布前审计
  （31/31 合法、measure 31/31、端到端 fid=1.000000）、232 项测试、图表 5.1–5.10。
- **影响可信度的缺失材料**：LAUREL 89 轮权重、基线逐线路原始行、缺陷扫描/跨拓扑/
  布局贡献/运行时间实验的脚本与数据未随包提交（仅主结果与 5.2/5.3/5.9 图可复现）。

## Lens A — Conceptual significance

**Overall assessment**：问题定位清楚（映射是量子编译的核心瓶颈），工程闭环完整；
但"创新"表述与真实贡献类型之间存在张力，且"零 qiskit"的边界需要收紧。

**Major strengths**：以真机可执行性为第一约束（门集/相邻/禁用比特/无 swap 残留），
评测口径统一到平台基，且如实披露大线路层与感知 O3 的差距；这在同类工作中不常见。

**Major concerns**：

Concern ID: A-M1
Issue key: novelty-framing-01
Severity: P2
Axis: 概念
Claim pointer: 结论 "在方法创新上，本文的主要创新可归纳为以下四个方面"；摘要 "方案按…三个阶段组织"
Evidence pointer: 结论第 2 段（"自主复现 SABRE 路由器"）、第 3 段（"引入 LAUREL…QuEst"）
Concern: 四项"创新"中，第二项是 SABRE 机制的自主复现 + 工程加速，第三项是 LAUREL/QuEst 的接入与改造，均非新算法；与第一项（噪声感知 RL 布局）和第四项（评测/验证体系）混称为"创新"。
Why it matters: 评审可能质疑创新性含金量；主动区分"新提出/复现改造/系统集成"反而增强可信度。
Resolution test: 将"主要创新"改为"主要工作与贡献"，每项显式标注类型（新提出 / 复现并工程加速 / 集成改造 / 评测体系），且摘要与结论用词一致。

Concern ID: A-M2
Issue key: claim-scope-qiskit-01
Severity: P1
Axis: 概念
Claim pointer: 1.1 "零 qiskit 依赖的自研映射优化管线"；1.3 "零 qiskit 依赖量子线路映射管线"；结论 "交付系统零 qiskit 依赖"
Evidence pointer: `qtrail/web/app.py` 第 18 行导入 `qtrail.utils.qasm_io`，后者第 9 行 `from qiskit import QuantumCircuit, qasm2`；`qtrail/pure_cli.py` 与 `qtrail/pure/` 无 qiskit 导入
Concern: 映射内核与 CLI 管线确实零 qiskit；但 Web 演示层（视频中展示的界面）通过 `qasm_io` 依赖 qiskit 做 QASM 解析与对照基线，包 README 又写成 "qtrail.pure_cli / qtrail.web 零 qiskit 依赖"，边界被扩大。
Why it matters: 评委若在 Web 环境未装 qiskit 时启动失败，会直接质疑核心卖点；主张与实现不一致是硬伤。
Resolution test: 所有对外材料统一为"映射内核与 CLI 交付管线零 qiskit；Web 演示层与对照实验使用 qiskit 做解析/基线"，并同步报告、PPT、视频字幕与包 README；或在 Web 层改用 `qtrail.pure.qasm` 解析并移除 qiskit。

**Minor concerns**：

Concern ID: A-M3
Issue key: learning-baseline-gap-01
Severity: P2
Axis: 概念
Claim pointer: 1.x "学习型布局"相关工作；结论 "将 QAP 求解从静态拓扑优化推进到噪声现实下可迁移的构造式决策"
Evidence pointer: 参考文献 [7] 仅出现在讨论段；实验部分无学习型映射方法的对照
Concern: 未与 QAP-Router 等学习型方法做实验对比，学习型定位的"相对先进性"无法从现有证据建立。
Why it matters: 审稿人可能要求与同类学习方法对比；不声明边界会被视为回避。
Resolution test: 在限制段明确"未与学习型方法做实验对比，原因是其代码/权重不可获得或规模不匹配"，或补一组小规模对照。

## Lens B — Technical integrity

**Overall assessment**：主结果的计算链路（pure product 模型 vs qiskit `estimate_fidelity`）公式一致，基线公平性设计充分；但复现材料与统计不确定性不足。

**Major strengths**：同一平台基分解、同一种子、同一指标实现；发布前审计把"语义等价"与"硬件可执行"分开验证；对 4 条超时线路如实披露。

**Major concerns**：

Concern ID: B-M1
Issue key: reproducibility-assets-01
Severity: P1
Axis: 复现
Claim pointer: 5.4.3 "本方案侧使用随仓库分发的 LAUREL 权重（89 轮收敛版）"；5.4.4 运行时间实验；5.4.3 缺陷扫描/跨拓扑
Evidence pointer: 提交包 `01_源码/QTrial/checkpoints/` 仅含 `tianyan-287…c0.05_best.pt` 与 `multi7topo…best.pt`；`scripts/` 未含 `make_report_figures.py`、`make_runtime_compare.py`、`defect_sweep.py`、`cross_topo_bench.py`、`bench_layout_contribution.py`；`data/scaling`、`tables/defect_sweep`、`tables/cross_topo`、`tables/compare_models/layout_contribution_*.jsonl` 均未入包
Concern: 报告引用 5.4.3–5.4.4 与图 5.4–5.8、5.10 的实验，但对应权重、脚本、数据不在提交包内，读者无法按包复现。
Why it matters: 报告明确承诺"可复现"，但关键支撑实验不可复现，削弱整体可信度。
Resolution test: 将 `checkpoints/demo_laurel/*.pt`、上述脚本、`data/scaling/`、`tables/defect_sweep/`、`tables/cross_topo/`、`tables/compare_models/layout_contribution_*.jsonl` 补入包，并在 5.4.3/5.4.4 标注对应脚本路径。

Concern ID: B-M2
Issue key: reproducibility-baseline-data-01
Severity: P1
Axis: 复现
Claim pointer: 摘要/5.1 "原始数据逐线路落盘 JSONL"；5.2 "所有方法共享同一指标实现"
Evidence pointer: 包内 `tables/submission/per_circuit.jsonl` 仅含 v8（qt_*）行；基线（aw_o3/aw_o1/b_o3/b_o1/tket）逐线路行仅存在于主仓库 `tables/final_report/`，未入包
Concern: 主表比值与胜场由基线行与 v8 行配对计算，但包内缺少基线逐线路原始行，读者只能看到汇总表，无法复核逐线路胜场 130/469/531。
Why it matters: "可复核"是报告的核心承诺；缺基线行使胜场与中位比值不可独立验证。
Resolution test: 将 `tables/final_report/{small,medium,large}_fidelity_rows.jsonl` 或其合并脱敏版补入包；或提供 `submission_aggregate.py` 可重建的完整输入。

Concern ID: B-M3
Issue key: statistical-uncertainty-01
Severity: P2
Axis: 统计
Claim pointer: 5.4.2 "0.965 倍 / 0.895 倍"；5.4.5 "0.924 倍 / 胜 130 场"
Evidence pointer: 表 5.x、图 5.9；全文无置信区间或检验
Concern: 关键比值与胜率均为点估计；小规模 0.965 与 1.0 的差异是否超出采样波动未说明。
Why it matters: "基本持平"的结论方向正确，但缺不确定性使"持平/差距"的边界不可判断。
Resolution test: 对配对线路做 bootstrap（≥1000 次）给出比值 95% CI，胜率给 Wilson 95% CI，并在 5.4.2 补一句；若 CI 跨 1，明确写"不能区分"。

Concern ID: B-M4
Issue key: runtime-single-run-01
Severity: P2
Axis: 统计
Claim pointer: 5.4.4 "自研纯管线全程保持秒级端到端耗时"
Evidence pointer: 图 5.7/5.8；正文注明"逐线路单次运行记录、不取平均"
Concern: 单次计时无法区分算法耗时与系统抖动；"秒级"结论缺少波动范围。
Why it matters: 评委可能追问运行时间稳定性；单次数据不足以支撑"全程秒级"的强表述。
Resolution test: 对代表性 10 条线路重复 3 次，报告 中位数与 IQR；或把结论限定为"单次记录下的典型值"。

Concern ID: B-M5
Issue key: evidence-test-count-01
Severity: P2
Axis: 证据
Claim pointer: 4.3.3 "方案的全部功能由 232 个自动化测试常驻守护"；附录 "232 个测试"
Evidence pointer: 本机 `python -m pytest tests/ -q` → `2 failed, 232 passed`；失败项为 `test_fit_block_torch_exact`（容差）与 `test_platform_verifier_needs_key`（本机存在 TIANYAN_LOGIN_KEY 的反向断言）
Concern: "232 个测试常驻守护"未披露 2 项已知失败；读者会理解为全绿。
Why it matters: 透明度问题；主动披露已知环境性失败比被评委发现更好。
Resolution test: 改为"232 项通过 + 2 项已知环境性断言（容差/本机密钥）"，并在附录列出测试命令与失败项。

Concern ID: B-M6
Issue key: evidence-demo-metric-01
Severity: P2
Axis: 证据
Claim pointer: 视频/Web 演示中的"估计保真度 0.8035"与"与 Qiskit SABRE 对比"
Evidence pointer: `tables/submission/captures/web_result.png`；Web 端点 `qtrail/web/app.py` 的 QTrial 指标来自 pipeline Mapper，基线来自 `sabre_transpile`+`compute_metrics`，基线 1Q 门体积与 QTrial 不同
Concern: Web 结果卡片的 0.8035 是内部估计口径，与报告 5.4.2 的 0.735（小层均值）不是同一统计对象；对照区默认勾选时可能显示 0.219 的基线值，易被误读为"全面领先"。
Why it matters: 演示数字与报告口径不一致会被追问；对照区若展示不可比数字，反而损害可信度。
Resolution test: Web 对照区标注"参考口径，非报告主表"并默认关闭；或统一用 `compute_metrics` 计算两侧指标后在 UI 标注口径；视频中已关闭对照，需在讲稿/备注中说明。

## Lens C — Evidence and communication

**Overall assessment**：主结果表述总体准确、诚实；但 587/583 口径、图注分母、编号体系存在局部不一致。

**Major strengths**：主动写出大线路层差距与感知 O3 的优势；图 5.2 明确标注乘积代理下溢；结论不使用"显著优于"等无数据形容词。

**Major concerns**：

Concern ID: C-M1
Issue key: reporting-counts-01
Severity: P2
Axis: 图文
Claim pointer: 摘要 "在 Benchpress、MQTBench 与 QUEKO 共 587 条线路的分层评测中"
Evidence pointer: 5.1 "587 条…其中 11-50 层 4 条超时"；5.4.2/5.4.5/结论均用 583 条完成
Concern: 摘要仍用 587 条作为评测规模，与正文/结论的 583 条完成不一致。
Why it matters: 同一文档两个规模口径，评委核对时会认为统计不严谨。
Resolution test: 摘要改为"587 条基准中 583 条完成（4 条超时）"，全文统一该表述。

Concern ID: C-M2
Issue key: figure-denominator-note-01
Severity: P2
Axis: 图文
Claim pointer: 5.4.5 "对噪声感知 O3 胜 130 场，对 pytket 胜 469 场，对盲目 O1 胜 531 场"
Evidence pointer: 图 5.9 右图标注 130/578、469/529、531/578
Concern: 正文未给分母，读者无法把 531 与图中 531/578 对齐；不同基线可比线路数不同（578/529/578）也未说明。
Why it matters: 胜场的可比样本数不同会影响解读；补分母是低成本高收益的修正。
Resolution test: 正文改为"对感知 O3 胜 130/578、对 pytket 469/529、对盲目 O1 531/578"，并加一句"分母差异来自基线缺失行"。

Concern ID: C-M3
Issue key: metric-label-consistency-01
Severity: P2
Axis: 图文
Claim pointer: 5.4.2 同段混用保真度均值与深度截尾值
Evidence pointer: 图 5.2 图注"均值口径"；图 5.3 图注"截尾均值口径"
Concern: 正文在保真度处用均值、深度处用截尾值，但未逐句标注口径，读者易把两者混为同一统计量。
Why it matters: 统计口径是本文强调的严谨性卖点，正文标注不完整会削弱该卖点。
Resolution test: 每次引用数值时写明"均值"或"截尾均值"；或在段首统一声明"本段保真度用均值、深度用截尾均值"。

Concern ID: C-M4
Issue key: numbering-consistency-01
Severity: P2
Axis: 图文
Claim pointer: 1.3 "本文的主要工作可归纳为以下五个方面"；结论 "主要创新可归纳为以下四个方面"
Evidence pointer: p10 vs p52
Concern: "五项工作"与"四项创新"数量与编号不对应，读者难以建立映射。
Why it matters: 评审需要快速抓住贡献；编号不一致会造成"贡献到底几项"的困惑。
Resolution test: 统一为"五项工作、四类贡献"并给出对应关系，或全部改为五项。

Concern ID: C-M5
Issue key: cover-placeholder-01
Severity: P2
Axis: 图文
Claim pointer: 封面 "（单位：姓名：手机号：     ）"
Evidence pointer: PDF 第 1 页
Concern: 提交前必须填写的封面信息仍为占位。
Why it matters: 报到材料的形式完整性；占位符会被视为未完成。
Resolution test: 填写单位、姓名、手机号后重新导出 PDF，并核对与报名回执一致。

Concern ID: C-M6
Issue key: claim-scope-qiskit-01
Severity: P1
Axis: 图文
Claim pointer: 结论 "交付系统零 qiskit 依赖"；包 README "qtrail.pure_cli / qtrail.web 零 qiskit 依赖"
Evidence pointer: 同 A-M2（web/app.py → qasm_io.py → qiskit）
Concern: 对外沟通层把"映射内核/CLI 零 qiskit"扩大为"交付系统/Web 零 qiskit"。
Why it matters: 这是评委最容易现场验证的主张；一旦 Web 在无 qiskit 环境启动失败，整份材料的可信度受损。
Resolution test: 报告、PPT、视频字幕、包 README 全部改为"映射内核与 CLI 零 qiskit；Web 演示层与对照实验使用 qiskit"。

Concern ID: C-M8
Issue key: overclaim-layout-first-01
Severity: P1
Axis: 证据
Claim pointer: 结论第 4 段 "本方案布局在 Benchpress、MQTBench 与 QUEKO 三个基准上保真度全面第一"
Evidence pointer: 5.4.3 同节 "与噪声感知 O3 的比值分别为 0.70、0.82、0.73"（本方案低于感知 O3）；图 5.6
Concern: 结论的"全面第一"未把噪声感知 O3 纳入比较，与 5.4.3 的如实披露矛盾；读者会理解为布局在全部基线中第一。
Why it matters: 结论层过度声称与正文自相矛盾，是评审最容易抓住的可信度问题。
Resolution test: 改为"高于盲目 qiskit 管线、噪声感知 O1 与 pytket；与噪声感知 O3 的差距见 5.4.3"，并全文检索无"全面第一"残留。

**Minor concerns**：

Concern ID: C-M7
Issue key: reproducibility-assets-01
Severity: P1
Axis: 图文
Claim pointer: 5.4.3/5.4.4 的"可复现"表述
Evidence pointer: 同 B-M1（包内缺权重/脚本/数据）
Concern: 报告在正文承诺复现，但读者按包无法复现 5.4.3–5.4.4。
Why it matters: 与 Lens B 同一问题在沟通层的体现；评委按包操作会直接遇到缺口。
Resolution test: 同 B-M1。

## Cross-review synthesis

**共同认可的优势**：真机可执行性优先的工程闭环；统一平台基与同种子口径；对大线路差距的如实披露；发布前审计的三层证据链。

**P0 共识风险**：无。核心命题（超过 pytket/盲目基线、小规模接近感知 O3、真机可执行）均有可核验证据支撑。

**P1 共识风险**：
1. `claim-scope-qiskit-01`（A-M2 / C-M6）：零 qiskit 的边界需收紧到映射内核与 CLI，并同步全部对外材料。
2. `reproducibility-assets-01`（B-M1 / C-M7）：补入 LAUREL 权重、实验脚本与数据，或缩小"可复现"的表述范围。

**仅单一视角提出但值得保留的问题**：
- A-M1 创新表述（概念层，建议改"工作与贡献"）；A-M3 学习型基线缺失（明确边界）。
- B-M2 基线逐线路数据缺失（复现层，建议补包）；B-M3 统计不确定性（建议补 bootstrap CI）；B-M4 运行时单次（建议限定表述）；B-M5 测试失败披露；B-M6 Web 指标口径。
- C-M1 587/583 统一；C-M2 胜场分母；C-M3 口径标注；C-M4 编号统一；C-M5 封面占位。

**各视角权重差异**：Lens A 更关注"创新含金量与主张边界"，Lens B 更关注"可复现与统计严谨"，Lens C 更关注"对外一致性与形式完整"。三者对"零 qiskit 边界"与"复现材料"的权重最高。

**最小修复路径（按优先级）**：
1. 收紧零 qiskit 表述（报告/PPT/README/视频字幕）——P1。
2. 补包：`demo_laurel` 权重、`tables/final_report` 基线行、`data/scaling`、`tables/{defect_sweep,cross_topo,compare_models}`、对应脚本——P1。
3. 摘要 587→583 完成口径；正文补胜场分母；统一"五项工作/四类贡献"编号——P2。
4. 补 bootstrap 95% CI 与测试失败披露——P2。
5. 封面信息填写——P2（提交前人工完成）。

**当前材料无法评估的事项**：竞赛评审规则下的"创新性"相对排名；真机 105 比特满占线路的独立复测（287 校准期）；与学习型映射方法的公平实验对比（无可用实现/权重）；生成式 AI 使用声明（竞赛未要求，未评估）。

## 整改记录（2026-11-12）

| Concern | 处置 | 证据 |
|---|---|---|
| A-M2 / C-M6 零 qiskit 边界 | 报告 1.1/1.3/结论加入边界说明；PPT 封面/第 5/9 页改为"映射内核零 qiskit"；视频管线图与工程承诺改为"映射内核与 CLI"；包 README 同步 | `作品报告书_决赛.pdf` p7/p10/p52；PPT 第 1/5/9 页；`promo_video/out/review/check_pipeline.png` |
| A-M1 创新表述 | 结论"主要创新"改为"五项工作可归纳为四类贡献"，与 1.3 节对应 | p52 |
| A-M3 学习型基线 | 保留为"无法评估"边界，未补实验（无可用实现/权重）；在预审边界中说明 | 本报告 Review setup |
| B-M1 / C-M7 复现材料 | 补入 LAUREL 权重、`tables/final_report/` 基线行、`data/scaling/`、`tables/{defect_sweep,cross_topo,compare_models,report_figures}`、7 个实验脚本 | `决赛提交/01_源码/QTrial/` 与包 README"复现实验"节 |
| B-M2 基线逐线路数据 | 同上，基线行已入包 | `tables/final_report/*.jsonl` |
| B-M3 统计不确定性 | 5.4.2 增补 2000 次配对 bootstrap 与 Wilson 95% CI；"基本持平"改为"接近但略低" | p42 |
| B-M4 运行时单次 | 报告原文已注明"逐线路单次记录"；保留并限定结论，未补重复实验 | p47–p48 |
| B-M5 测试失败披露 | 改为"232 项通过 + 2 项已知环境性断言" | p31、附录 |
| B-M6 Web 指标口径 | Web 对照默认关闭并加注"参考口径"；视频中关闭对照 | `qtrail/web/static/index.html` |
| C-M1 587/583 | 摘要改为"587 条基准线路（完成 583 条，4 条超时）"；全文统一 | p2 |
| C-M2 胜场分母 | 正文补 130/578、469/529、531/578 及分母差异说明 | p49 |
| C-M3 口径标注 | 5.4.2 明确"均值/截尾均值"；2Q 深度改为"比感知 O3 高约 34%（越低越好）" | p42 |
| C-M4 编号统一 | 结论改为"五项工作→四类贡献"并标注对应 | p52 |
| C-M5 封面占位 | 保留占位，需提交人填写（无法代填） | p1 |
| C-M8 结论过度声称 | "全面第一"改为"高于盲目管线/感知 O1/pytket；与感知 O3 差距见 5.4.3" | p52 |
