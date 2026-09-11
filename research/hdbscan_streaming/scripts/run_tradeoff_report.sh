#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
PLOT_PY="${PLOT_PY:-python3}"

test -x "$PY"
test -f "$ROOT/results/fsdd_full_baseline/summary.json"

bash "$ROOT/scripts/run_optimization_candidates.sh"
"$PLOT_PY" "$ROOT/scripts/plot_tradeoff_report.py" \
  --comparison "$ROOT/results/fsdd_optimization_candidates/compare_to_full.csv" \
  --results "$ROOT/results/fsdd_optimization_candidates" \
  --out "$ROOT/results/fsdd_optimization_candidates/tradeoff_report.png"

printf 'TRADEOFF_REPORT_PASS %s\n' "$ROOT/results/fsdd_optimization_candidates"
