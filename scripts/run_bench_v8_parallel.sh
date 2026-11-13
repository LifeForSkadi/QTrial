#!/usr/bin/env bash
# v8 全量基准并行重跑（2026-11-13 QFT/graphstate 修复后）。
# 8+8+2 分片并发；CUDA 保留（策略推理），OMP/MKL 降到 1 避免线程超订。
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/bench_v8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

for k in 0 1 2 3 4 5 6 7; do
  python -X utf8 scripts/final_report_bench_v8.py \
    --buckets large --shard "$k/8" --seeds 16 --dev cpu \
    > "logs/bench_v8/large_$k.log" 2>&1 &
done
for k in 0 1 2 3 4 5 6 7; do
  python -X utf8 scripts/final_report_bench_v8.py \
    --buckets medium --shard "$k/8" --seeds 16 --dev cpu \
    > "logs/bench_v8/medium_$k.log" 2>&1 &
done
for k in 0 1; do
  python -X utf8 scripts/final_report_bench_v8.py \
    --buckets small --shard "$k/2" --seeds 16 --dev cpu \
    > "logs/bench_v8/small_$k.log" 2>&1 &
done
wait
echo "ALL_DONE"
