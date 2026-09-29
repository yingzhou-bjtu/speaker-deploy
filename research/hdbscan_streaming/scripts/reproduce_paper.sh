#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODE="${1:-smoke}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results}"

ensure_env() {
  if [[ ! -x "$PYTHON" ]]; then
    bash "$ROOT/scripts/setup_research_env.sh"
  fi
  test -x "$PYTHON"
}

case "$MODE" in
  setup)
    bash "$ROOT/scripts/setup_research_env.sh"
    ;;
  data)
    bash "$ROOT/scripts/download_fsdd.sh"
    ;;
  audio)
    ensure_env
    bash "$ROOT/scripts/run_audio_smoke.sh"
    ;;
  baseline)
    ensure_env
    bash "$ROOT/scripts/run_full_baseline.sh"
    ;;
  smoke)
    ensure_env
    bash "$ROOT/scripts/run_synthetic_smoke.sh"
    ;;
  async)
    ensure_env
    bash "$ROOT/scripts/run_async_validation.sh"
    ;;
  rate)
    ensure_env
    "$PYTHON" "$ROOT/scripts/run_rate_sweep.py" \
      --data "$ROOT/data/synthetic_320.npz" \
      --out "$OUT_ROOT/rate_sweep_release"
    ;;
  multiseed)
    ensure_env
    "$PYTHON" "$ROOT/scripts/run_multiseed_validation.py" \
      --out "$OUT_ROOT/multiseed_release" \
      --seeds 20260910 20260911 20260912 20260913 20260914
    ;;
  release)
    ensure_env
    "$PYTHON" "$ROOT/scripts/check_release.py"
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
    ensure_env
    "$PYTHON" "$ROOT/scripts/check_release.py"
    OUT_ROOT="$OUT_ROOT/final_rerun_release" \
      bash "$ROOT/scripts/run_final_rerun.sh"
    ;;
  all)
    ensure_env
    "$PYTHON" "$ROOT/scripts/check_release.py"
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
  *)
    printf 'usage: %s {setup|data|audio|baseline|smoke|async|rate|multiseed|release|main|all}\n' "$0" >&2
    exit 2
    ;;
esac
