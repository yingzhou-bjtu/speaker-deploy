#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/.venv/bin/python" "$ROOT/scripts/run_multiseed_validation.py" \
  --out "$ROOT/results/multiseed_validation"
printf 'MULTISEED_VALIDATION_PASS %s\n' "$ROOT/results/multiseed_validation/multiseed_summary.json"
