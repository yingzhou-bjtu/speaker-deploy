#!/bin/bash
# 清空声纹和 PCM 日志与运行时落盘，不删 venv 和 ONNX。
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 停止服务 ==="
pkill -f 'robot_service.py' 2>/dev/null || true
pkill -f 'api_server.py' 2>/dev/null || true
sleep 1

echo "=== 清空日志 ==="
: > "${SV_ROBOT_LOG:-/data/local/tmp/sv_robot.log}" 2>/dev/null || rm -f /data/local/tmp/sv_robot.log
: > "${SV_API_LOG:-/data/local/tmp/sv_api.log}" 2>/dev/null || rm -f /data/local/tmp/sv_api.log
rm -f /data/local/tmp/sv_probe*.json 2>/dev/null || true

echo "=== 清空 deploy 运行时 ==="
rm -rf ./runs/*.json ./runs/probe_* 2>/dev/null || true
rm -f ./speaker_registry.npz ./embedding_store.npz 2>/dev/null || true

echo "=== 完成（进程内聚类缓冲已随停止释放）==="
echo "重新启动: ./start_robot.sh"
