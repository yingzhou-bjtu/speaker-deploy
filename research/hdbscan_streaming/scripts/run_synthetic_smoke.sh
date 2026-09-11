#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DATA="$ROOT/data/synthetic_smoke.npz"
META="$ROOT/data/synthetic_smoke.json"
OUT="$ROOT/results/smoke_latest"

test -x "$PY"
"$PY" "$ROOT/scripts/generate_stream.py" \
  --out "$DATA" \
  --meta "$META" \
  --speakers 5 \
  --points-per-speaker 24 \
  --dim 32 \
  --seed 20260910
"$PY" "$ROOT/scripts/validate_dataset.py" "$DATA"
"$PY" "$ROOT/scripts/run_benchmark.py" \
  --data "$DATA" \
  --out "$OUT" \
  --methods full_hdbscan compressed_hdbscan window_hdbscan fishdbc approx_predict fast_hdbscan \
  --cap 40 \
  --warmup 40 \
  --min-cluster-size 4 \
  --min-samples 2 \
  --checkpoint-every 10
