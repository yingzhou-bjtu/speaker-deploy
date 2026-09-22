#!/usr/bin/env bash
# Full main-table rerun (L/M/H, six methods) with the repaired code.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/final_rerun"
METHODS="full_hdbscan fishdbc adaptive_fishdbc adaptive_fishdbc_async ahc sc_pna diart_style"

run() {
  local label="$1" data="$2" warmup="$3" min_samples="$4" checkpoint="$5"
  local dest="$OUT/$label"
  mkdir -p "$dest"
  echo "=== $label ==="
  "$ROOT/.venv/bin/python" "$ROOT/scripts/run_benchmark.py" \
    --data "$data" --out "$dest" --methods $METHODS \
    --warmup "$warmup" --tradeoff-warmup "$warmup" --tradeoff-margin .02 \
    --tradeoff-max-distance .70 --tradeoff-full-interval 8 \
    --tradeoff-batch-size 4 --tradeoff-max-queue 8 \
    --min-cluster-size 3 --min-samples "$min_samples" \
    --arrival-interval-ms 5 --checkpoint-every "$checkpoint" \
    >>"$dest/run.log" 2>&1
  echo "=== $label done ==="
}

run L "$ROOT/data/fsdd/embeddings/fsdd_smoke.npz" 12 2 6
run M "$ROOT/data/fsdd/medium/embeddings.npz" 20 6 80
run H "$ROOT/data/fsdd/heavy/embeddings.npz" 20 6 80
echo "ALL DONE"
