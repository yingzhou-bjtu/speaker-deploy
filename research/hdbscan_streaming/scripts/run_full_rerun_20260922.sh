#!/usr/bin/env bash
# Full experiment rerun after repair-fix (2026-09-22). Outputs to results/rerun_20260922.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/results/rerun_20260922"
mkdir -p "$OUT"
log="$OUT/_run.log"
exec > >(tee -a "$log") 2>&1

echo "=== group 1: async_validation (synthetic low-noise, repair-affected) ==="
INT="${INTERVAL_MS:-5}"
# FSDD (30)
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/fsdd/embeddings/fsdd_smoke.npz" \
  --out "$OUT/async_fsdd_adaptive" --methods adaptive_fishdbc_async \
  --warmup 12 --tradeoff-warmup 12 --tradeoff-margin 0.02 \
  --tradeoff-max-distance 0.70 --tradeoff-full-interval 4 \
  --tradeoff-batch-size 4 --arrival-interval-ms "$INT" --tradeoff-max-queue 8 \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 6
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/fsdd/embeddings/fsdd_smoke.npz" \
  --out "$OUT/async_fsdd_fishdbc" --methods fishdbc \
  --warmup 12 --tradeoff-warmup 12 --tradeoff-margin 0.02 \
  --tradeoff-max-distance 0.70 --tradeoff-full-interval 4 \
  --tradeoff-batch-size 4 --arrival-interval-ms "$INT" --tradeoff-max-queue 8 \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 6
# synthetic 320
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/synthetic_320.npz" \
  --out "$OUT/async_synthetic_adaptive" --methods adaptive_fishdbc_async \
  --warmup 20 --tradeoff-warmup 20 --tradeoff-margin 0.02 \
  --tradeoff-max-distance 0.70 --tradeoff-full-interval 8 \
  --tradeoff-batch-size 4 --arrival-interval-ms "$INT" --tradeoff-max-queue 8 \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 80
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/synthetic_320.npz" \
  --out "$OUT/async_synthetic_fishdbc" --methods fishdbc \
  --warmup 20 --arrival-interval-ms "$INT" \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 80

echo "=== group 2: synthetic main (S320 + drift D320/D640) ==="
COMMON=(--warmup 20 --tradeoff-warmup 20 --tradeoff-margin 0.02
  --tradeoff-max-distance 0.70 --tradeoff-full-interval 8
  --tradeoff-batch-size 4 --arrival-interval-ms 5 --tradeoff-max-queue 8
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 80)
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/synthetic_320.npz" \
  --out "$OUT/synthetic_S320" --methods adaptive_fishdbc_async fishdbc "${COMMON[@]}"
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/synthetic_drift_320.npz" \
  --out "$OUT/synthetic_D320" --methods adaptive_fishdbc_async fishdbc "${COMMON[@]}"
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/synthetic_drift_640.npz" \
  --out "$OUT/synthetic_D640" --methods adaptive_fishdbc_async fishdbc "${COMMON[@]}"

echo "=== group 3: multiseed 5 seeds ==="
"$PY" "$ROOT/scripts/run_multiseed_validation.py" --out "$OUT/multiseed" \
  --seeds 20260910 20260911 20260912 20260913 20260914

echo "=== group 4: rate sweep (synthetic 320) ==="
"$PY" "$ROOT/scripts/run_rate_sweep.py" --data "$ROOT/data/synthetic_320.npz" \
  --out "$OUT/rate_sweep" --intervals 1 5 10 20

echo "ALL RERUN COMPLETE"
