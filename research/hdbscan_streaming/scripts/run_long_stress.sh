#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DATA="$ROOT/data/synthetic_drift_640.npz"
META="$ROOT/data/synthetic_drift_640.json"
OUT="$ROOT/results/long_stress_640"

"$PY" "$ROOT/scripts/generate_drift_stream.py" --points 640 --out "$DATA" --meta "$META"
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$DATA" --out "$OUT" \
  --methods adaptive_fishdbc_async --warmup 20 --tradeoff-warmup 20 \
  --tradeoff-margin 0.02 --tradeoff-max-distance 0.70 --tradeoff-full-interval 8 \
  --tradeoff-batch-size 4 --tradeoff-max-queue 8 --arrival-interval-ms 2 \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 160
printf 'LONG_STRESS_PASS %s\n' "$OUT/adaptive_fishdbc_async/summary.json"
