#!/usr/bin/env bash
# v8 全样本合法性复验（pre_cancel 默认开后，2026-11-13）。
# 2 large + 3 medium + 1 small 分片并发，纯 CPU。
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/legality_v8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

for k in 0 1; do
  python -u -X utf8 scripts/verify_all_legality.py \
    --buckets large --shard "$k/2" --out-dir tables/final_report_v8 \
    > "logs/legality_v8/large_$k.log" 2>&1 &
done
for k in 0 1 2; do
  python -u -X utf8 scripts/verify_all_legality.py \
    --buckets medium --shard "$k/3" --out-dir tables/final_report_v8 \
    > "logs/legality_v8/medium_$k.log" 2>&1 &
done
python -u -X utf8 scripts/verify_all_legality.py \
  --buckets small --out-dir tables/final_report_v8 \
  > "logs/legality_v8/small.log" 2>&1 &
wait
echo "ALL_DONE"
