#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODE="${1:-smoke}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results}"

test -x "$PYTHON"

case "$MODE" in
  smoke)
    bash "$ROOT/scripts/run_synthetic_smoke.sh"
    ;;
  async)
    bash "$ROOT/scripts/run_async_validation.sh"
    ;;
  rate)
    "$PYTHON" "$ROOT/scripts/run_rate_sweep.py" \
      --data "$ROOT/data/synthetic_320.npz" \
      --out "$OUT_ROOT/rate_sweep_release"
    ;;
  multiseed)
    "$PYTHON" "$ROOT/scripts/run_multiseed_validation.py" \
      --out "$OUT_ROOT/multiseed_release" \
      --seeds 20260910 20260911 20260912 20260913 20260914
    ;;
  release)
    OUT_ROOT="$OUT_ROOT/smoke_release" \
      bash "$ROOT/scripts/run_synthetic_smoke.sh"
    OUT_ROOT="$OUT_ROOT/async_validation_release" \
      bash "$ROOT/scripts/run_async_validation.sh"
    "$PYTHON" "$ROOT/scripts/run_rate_sweep.py" \
      --data "$ROOT/data/synthetic_320.npz" \
      --out "$OUT_ROOT/rate_sweep_release"
    "$PYTHON" "$ROOT/scripts/run_multiseed_validation.py" \
      --out "$OUT_ROOT/multiseed_release" \
      --seeds 20260910 20260911 20260912 20260913 20260914
    ;;
  main)
    OUT_ROOT="$OUT_ROOT/final_rerun_release" \
      bash "$ROOT/scripts/run_final_rerun.sh"
    ;;
  *)
    printf 'usage: %s {smoke|async|rate|multiseed|release|main}\n' "$0" >&2
    exit 2
    ;;
esac
