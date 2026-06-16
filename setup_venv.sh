#!/bin/bash
# 在 deploy 目录建独立 venv，避免系统 NumPy 2.x 和 hdbscan 冲突。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "需要 python3" >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq python3-venv python3-dev build-essential libsndfile1 2>/dev/null || true
fi

echo "=== 创建 venv: $ROOT/.venv ==="
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip wheel setuptools

echo "=== 1) 固定 NumPy 1.x ==="
pip install --force-reinstall 'numpy==1.26.4'
python -c "import numpy as np; print('numpy', np.__version__, np.__file__)"
case "$(python -c 'import numpy; print(numpy.__version__)')" in
  2.*) echo "FATAL: venv 里仍是 NumPy 2.x" >&2; exit 1 ;;
esac

echo "=== 2) 编译 hdbscan（绑定当前 NumPy）==="
pip uninstall -y hdbscan 2>/dev/null || true
pip install --no-cache-dir --no-binary=hdbscan 'hdbscan>=0.8.33'

echo "=== 3) 其余依赖（不再升级 numpy）==="
pip install \
  'scipy>=1.10,<1.15' \
  'scikit-learn>=1.2,<1.6' \
  'onnxruntime>=1.16,<1.20' \
  'kaldi-native-fbank>=1.20' \
  'soundfile>=0.12'

echo "=== 验证 ==="
python -c "import numpy; import hdbscan; import onnxruntime; print('OK numpy', numpy.__version__, 'hdbscan', hdbscan.__version__)"
echo ""
echo "以后请用:"
echo "  source $ROOT/.venv/bin/activate"
echo "  python run.py audio /path/to.wav"
echo "或: $ROOT/run_venv.sh audio ..."
