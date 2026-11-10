# Claude Code 权限放行文档（QTrial 攻坚会话 2026-11-13）

## 用户决定：授予所有权限（全量放行）

本会话 Auto 模式的安全审查组件不可用，导致 Bash/WebSearch/WebFetch
被一律拒绝。用户已明确授权**所有权限**。

本构建界面只有 Auto / Plan / Manual / Edit automatically 四种模式，
无 Bypass 档位；且当前会话已锁死在坏的 Auto 模式——界面切换只对
新会话生效。**有效做法 = 手动写 settings.local.json + 重启会话**。

## 方式一（推荐，一步到位）：手动创建 settings.local.json

在 IDE 中新建 `F:\Study\Quantum Computing\信安竞赛\QTrial\.claude\settings.local.json`
（Claude 无法代写此文件——权限设置文件本身受安全审查保护）：

```json
{
  "permissions": {
    "defaultMode": "acceptEdits",
    "allow": [
      "Bash(*)",
      "WebSearch",
      "WebFetch(*)",
      "Read(*)",
      "Edit(*)",
      "Write(*)",
      "Glob(*)",
      "Grep(*)",
      "NotebookEdit(*)"
    ]
  }
}
```

`Bash(*)` = 所有 bash 命令放行；WebSearch/WebFetch 全放行；文件工具
全放行——效果等同"所有权限"。保存后**重启 Claude Code 会话**（新
会话不再沿用坏掉的 Auto 模式，并读取该文件）。重启时若右上角可选，
选 Edit automatically 即可。

## 方式二：界面切换 + 重启

- 右上角权限模式选择器切到 **Edit automatically**（或 Manual）；
- 重启会话；未匹配 allow 规则的命令会弹窗，逐个批准。

## 方式三（临时）：CLI 启动参数

```
claude --dangerously-skip-permissions
```

## 生效后的执行清单（按顺序）

1. **Tuccil merge 跳过消融实验**（已实现，等验证）：
   ```
   python -X utf8 scripts/exp_tuccil_lock.py
   ```
   预期：re=1/3 × lock=on/off 四组 1Q/cz/depth 对比 + lock 产物
   Statevector 语义 fid=1；基线（lock off, re=1）应复现 1Q=1152。

2. **重启全量重跑 11 分片**（会话切换被杀，进度 ~1/3；断点续跑，
   已完成行自动跳过——**不要加 rm**）：
   ```
   python -X utf8 scripts/final_report_bench_v7.py --buckets small --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets medium --shard 0/6 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets medium --shard 1/6 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets medium --shard 2/6 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets medium --shard 3/6 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets medium --shard 4/6 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets medium --shard 5/6 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets large --shard 0/4 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets large --shard 1/4 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets large --shard 2/4 --timeout 900
   python -X utf8 scripts/final_report_bench_v7.py --buckets large --shard 3/4 --timeout 900
   ```

3. **收数字**：全量完成后跑 `three_way_compare.py` 出三层三方法对比，
   评估是否超越 RL+感知O3。

4. **qiskit 本地源码核查**（Tuccil lock 的"已有工作"搜索被权限阻断）：
   ```
   python -c "import qiskit,os;print(os.path.dirname(qiskit.__file__))"
   ```
   定位后 Grep 其 `consolidate_blocks.py` / two-qubit peephole 实现，
   确认"保护已合成块免于再优化"机制是否已有先例（若无则为自主
   创新点，可写入叙事）。

## 边界提醒（与权限无关的既有纪律）

权限全放行 ≠ 无需谨慎：删除类（rm/reset --hard）与对外动作
（git push/网络外发）仍会在动手前向你确认一次——这是不可逆/
对外操作的行为准则，不消耗权限弹窗。
