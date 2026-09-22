#!/usr/bin/env bash
# Full-HDBSCAN full-prefix latency at final prefixes for scaling reference.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/scale_canonical/full_base"
mkdir -p "$OUT"

for n in 1280 2560; do
  data="$ROOT/data/scale/synthetic_scale_${n}.npz"
  echo "=== full_base n=${n} ==="
  "$ROOT/.venv/bin/python" - "$data" "$OUT" "$n" <<'PY'
import sys, json, time, numpy as np
data_path, out_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = np.load(data_path); X = d['X'].astype('float32'); y = d['y']
Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
import hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
t0 = time.perf_counter()
lab = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean').fit_predict(Xn)
wall_ms = (time.perf_counter() - t0) * 1000.0
out = {
    "n": n,
    "method": "full_hdbscan",
    "full_prefix_ms": round(wall_ms, 3),
    "ari": round(float(adjusted_rand_score(y, lab)), 6),
    "nmi": round(float(normalized_mutual_info_score(y, lab)), 6),
}
open(f"{out_dir}/full_{n}.json", "w").write(json.dumps(out, indent=2))
print(json.dumps(out))
PY
done
echo "FULL_BASE DONE"
