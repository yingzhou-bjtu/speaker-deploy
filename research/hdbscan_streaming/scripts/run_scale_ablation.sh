#!/usr/bin/env bash
# Scaling (canonical): FlowFish vs FISHDBC on synthetic 1280/2560 streams.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/scale_canonical"
mkdir -p "$OUT"

for n in 1280 2560; do
  data="$ROOT/data/scale/synthetic_scale_${n}.npz"
  dest="$OUT/n${n}"
  mkdir -p "$dest"
  echo "=== n=${n} ==="
  "$ROOT/.venv/bin/python" "$ROOT/scripts/run_benchmark.py" \
    --data "$data" --out "$dest" \
    --methods adaptive_fishdbc_async fishdbc \
    --warmup 20 --tradeoff-warmup 20 --tradeoff-margin .02 \
    --tradeoff-max-distance .70 --tradeoff-full-interval 8 \
    --tradeoff-batch-size 4 --tradeoff-max-queue 8 \
    --min-cluster-size 3 --min-samples 6 \
    --arrival-interval-ms 5 --checkpoint-every 9999 \
    >"$dest/run.log" 2>&1
  echo "=== n=${n} done ==="
done
echo "ALL DONE"
