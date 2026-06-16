#!/bin/bash
# 板端启动声纹、robotmedia PCM 和 HTTP API 一体服务。
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash 2>/dev/null || true

set -eu
PORT="${SV_API_PORT:-8765}"
LOG="${SV_ROBOT_LOG:-/data/local/tmp/sv_robot.log}"
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "缺少 .venv，请先 ./setup_board_deps.sh" >&2
  exit 1
fi

echo "=== preflight ==="
"$PY" scripts/preflight_board.py \
  --pcm-host "${SV_PCM_HOST:-127.0.0.1}" \
  --pcm-port "${SV_PCM_PORT:-5001}"

pkill -f 'robot_service.py' 2>/dev/null || true
pkill -f 'api_server.py' 2>/dev/null || true
sleep 1
# 日志和落盘 wav 由 robot_service 维护线程自动裁剪。
export SV_ROBOT_LOG="$LOG"
export SV_LOG_MAX_MB="${SV_LOG_MAX_MB:-8}"
export SV_LOG_KEEP_MB="${SV_LOG_KEEP_MB:-4}"
export SV_MAINT_INTERVAL_SEC="${SV_MAINT_INTERVAL_SEC:-120}"
export SV_WAV_MAX_FILES="${SV_WAV_MAX_FILES:-2000}"
export SV_WAV_KEEP_FILES="${SV_WAV_KEEP_FILES:-1000}"
export SV_WAV_PRUNE_MIN_AGE_SEC="${SV_WAV_PRUNE_MIN_AGE_SEC:-600}"

EXTRA_ARGS=()
if [[ -n "${SV_VERBOSE:-}" ]]; then
  EXTRA_ARGS+=(--verbose)
fi
nohup "$PY" robot_service.py \
  --api-port "$PORT" \
  --ort-threads "${SV_ORT_THREADS:-4}" \
  --max-hdbscan-utterances "${SV_MAX_HDBSCAN_UTTERANCES:-60}" \
  --hdbscan-min-cluster-size "${SV_HDBSCAN_MIN_CLUSTER_SIZE:-3}" \
  --hdbscan-min-samples "${SV_HDBSCAN_MIN_SAMPLES:-2}" \
  --hdbscan-cluster-selection-epsilon "${SV_HDBSCAN_EPS:-0.05}" \
  --max-ingest-audio-sec "${SV_MAX_INGEST_AUDIO_SEC:-15}" \
  --pcm-max-duration-sec "${SV_PCM_MAX_DURATION_SEC:-15}" \
  --pcm-host "${SV_PCM_HOST:-127.0.0.1}" \
  --pcm-port "${SV_PCM_PORT:-5001}" \
  --pcm-timeout "${SV_PCM_TIMEOUT:-0}" \
  --pcm-mode "${SV_PCM_MODE:-vad}" \
  --pcm-source "${SV_PCM_SOURCE:-wav_dir}" \
  --pcm-wav-dir "${SV_PCM_WAV_DIR:-/userdata/cloud_voice}" \
  --pcm-min-duration-sec "${SV_PCM_MIN_DURATION_SEC:-2.0}" \
  --pcm-segment-end-idle-sec "${SV_PCM_SEGMENT_END_IDLE_SEC:-1.0}" \
  --pcm-vad-end-mode "${SV_PCM_VAD_END_MODE:-silero_wav}" \
  --pcm-vad-end-audio "${SV_PCM_VAD_END_AUDIO:-auto}" \
  --pcm-vad-end-stable-sec "${SV_PCM_VAD_END_STABLE_SEC:-0.15}" \
  --pcm-tcp-buf-stale-sec "${SV_PCM_TCP_BUF_STALE_SEC:-45}" \
  "${EXTRA_ARGS[@]}" \
  >>"$LOG" 2>&1 &
echo "robot service pid=$! api=:$PORT log=$LOG"
sleep 12
curl -sf "http://127.0.0.1:${PORT}/v1/health" && echo
