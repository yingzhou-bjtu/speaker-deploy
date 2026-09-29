#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA="${1:-${DATASET:-$ROOT/data/fsdd/embeddings/fsdd_smoke.npz}}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/full_baseline}"

if [[ ! -x "$PYTHON" ]]; then
  bash "$ROOT/scripts/setup_research_env.sh"
fi
test -f "$DATA"

"$PYTHON" "$ROOT/scripts/validate_dataset.py" "$DATA"
"$PYTHON" "$ROOT/scripts/run_benchmark.py" \
  --data "$DATA" \
  --out "$OUT_ROOT" \
  --methods full_hdbscan \
  --cap 20 \
  --warmup 12 \
  --min-cluster-size 3 \
  --min-samples 2 \
  --checkpoint-every 6

printf 'FULL_BASELINE_PASS %s\n' "$OUT_ROOT"
