#!/bin/bash
# 用 deploy 目录下的 venv 跑 run.py，避免吃到系统 NumPy 2.x。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "缺少 $ROOT/.venv ，请先执行: ./setup_venv.sh" >&2
  exit 1
fi
exec "$VENV_PY" "$ROOT/run.py" "$@"
