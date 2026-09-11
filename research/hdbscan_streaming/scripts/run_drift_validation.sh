#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DATA="$ROOT/data/synthetic_drift_320.npz"
META="$ROOT/data/synthetic_drift_320.json"
OUT="$ROOT/results/drift_validation"

"$PY" "$ROOT/scripts/generate_drift_stream.py" --out "$DATA" --meta "$META"
COMMON=(--warmup 20 --tradeoff-warmup 20 --tradeoff-margin 0.02
  --tradeoff-max-distance 0.70 --tradeoff-full-interval 8
  --tradeoff-batch-size 4 --arrival-interval-ms 5
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 80)
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$DATA" --out "$OUT/adaptive" \
  --methods adaptive_fishdbc_async "${COMMON[@]}"
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$DATA" --out "$OUT/fishdbc" \
  --methods fishdbc --warmup 20 --arrival-interval-ms 5 \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 80

printf 'DRIFT_VALIDATION_PASS %s\n' "$OUT"
