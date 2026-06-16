#!/bin/bash
# 板端启动声纹 HTTP API，需已建好 venv。
set -euo pipefail
cd "$(dirname "$0")"
PORT="${SV_API_PORT:-8765}"
LOG="${SV_API_LOG:-/data/local/tmp/sv_api.log}"
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少 .venv，请先: ./setup_board_deps.sh 或 pip+virtualenv" >&2
  exit 1
fi
pkill -f 'api_server.py' 2>/dev/null || true
sleep 1
nohup "$PY" api_server.py \
  --host 0.0.0.0 --port "$PORT" \
  --ort-threads "${SV_ORT_THREADS:-4}" \
  --max-hdbscan-utterances "${SV_MAX_HDBSCAN_UTTERANCES:-60}" \
  >>"$LOG" 2>&1 &
echo "started pid=$! port=$PORT log=$LOG"
sleep 8
curl -sf "http://127.0.0.1:${PORT}/v1/health" && echo
