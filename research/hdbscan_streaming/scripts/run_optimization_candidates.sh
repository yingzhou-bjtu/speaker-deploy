#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
EMBED="$ROOT/data/fsdd/embeddings/fsdd_smoke.npz"
FULL="$ROOT/results/fsdd_full_baseline/summary.json"
OUT="$ROOT/results/fsdd_optimization_candidates"
COMPARE="$OUT/compare_to_full.json"

test -x "$PY"
test -f "$EMBED"
test -f "$FULL"

"$PY" "$ROOT/scripts/run_benchmark.py" \
  --data "$EMBED" \
  --out "$OUT" \
  --methods adaptive_tradeoff adaptive_fishdbc fishdbc ahc sc_pna diart_style \
  --cap 20 \
  --warmup 12 \
  --tradeoff-warmup 12 \
  --tradeoff-margin 0.02 \
  --tradeoff-max-distance 0.70 \
  --tradeoff-full-interval 4 \
  --tradeoff-batch-size 4 \
  --min-cluster-size 3 \
  --min-samples 2 \
  --checkpoint-every 6

"$PY" "$ROOT/scripts/compare_to_full.py" \
  --full "$FULL" \
  --candidates "$OUT/summary.json" \
  --out "$COMPARE"

printf 'OPTIMIZATION_CANDIDATES_PASS %s\n' "$OUT"
