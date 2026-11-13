#!/usr/bin/env bash
# pytket 基线补读出（2026-11-13）：medium 4 分片 + large 8 分片并发。
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/pytket_readout
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

for k in 0 1 2 3; do
  python -u -X utf8 scripts/fix_pytket_readout.py --buckets medium \
    --shard "$k/4" > "logs/pytket_readout/medium_$k.log" 2>&1 &
done
for k in 0 1 2 3 4 5 6 7; do
  python -u -X utf8 scripts/fix_pytket_readout.py --buckets large \
    --shard "$k/8" > "logs/pytket_readout/large_$k.log" 2>&1 &
done
wait
echo "ALL_DONE"
