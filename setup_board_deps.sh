#!/bin/bash
# pip install hdbscan 会拉 numpy 2.x，必须用 venv 和固定安装顺序。
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV_DIR:-.venv}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "=== 创建虚拟环境 $VENV ==="
if [[ ! -x "$PY" ]]; then
  if python3 -m venv "$VENV" 2>/dev/null; then
    :
  else
    echo "python3 -m venv 不可用，改用 virtualenv"
    pip3 install -q virtualenv || python3 -m pip install -q virtualenv
    python3 -m virtualenv "$VENV"
  fi
fi
"$PIP" install --upgrade pip wheel setuptools cython

echo "=== 1) 先钉死 NumPy 1.x ==="
"$PIP" install 'numpy==1.26.4'
"$PY" -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__"

echo "=== 2) scipy / sklearn（与 hdbscan 0.8.42 兼容的一组）==="
"$PIP" install 'scipy==1.11.4' 'scikit-learn==1.5.2'

echo "=== 3) 在已有 NumPy 1.x 下编译 hdbscan（禁止 build isolation 拉 numpy 2）==="
"$PIP" uninstall -y hdbscan 2>/dev/null || true
PIP_NO_BUILD_ISOLATION=1 "$PIP" install --no-cache-dir --no-binary=hdbscan --no-deps hdbscan==0.8.42

echo "=== 4) 其余依赖 ==="
"$PIP" install onnxruntime kaldi-native-fbank soundfile

echo "=== 验证 ==="
"$PY" -c "import numpy, hdbscan, scipy, sklearn; print('OK', numpy.__version__, hdbscan.__version__)"
echo ""
echo "以后请先: source $VENV/bin/activate"
echo "再运行: python run.py audio ..."
