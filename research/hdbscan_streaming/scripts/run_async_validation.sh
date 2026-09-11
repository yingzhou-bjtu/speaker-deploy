#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
INTERVAL_MS="${INTERVAL_MS:-5}"
COMMON=(--warmup 12 --tradeoff-warmup 12 --tradeoff-margin 0.02
  --tradeoff-max-distance 0.70 --tradeoff-full-interval 4
  --tradeoff-batch-size 4 --arrival-interval-ms "$INTERVAL_MS"
  --tradeoff-max-queue 8
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 6)

"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/fsdd/embeddings/fsdd_smoke.npz" \
  --out "$ROOT/results/async_validation/fsdd_adaptive" \
  --methods adaptive_fishdbc_async "${COMMON[@]}"
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/fsdd/embeddings/fsdd_smoke.npz" \
  --out "$ROOT/results/async_validation/fsdd_fishdbc" \
  --methods fishdbc "${COMMON[@]}"
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/synthetic_320.npz" \
  --out "$ROOT/results/async_validation/synthetic_adaptive" \
  --methods adaptive_fishdbc_async --warmup 20 --tradeoff-warmup 20 \
  --tradeoff-margin 0.02 --tradeoff-max-distance 0.70 --tradeoff-full-interval 8 \
  --tradeoff-batch-size 4 --arrival-interval-ms "$INTERVAL_MS" \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 80
"$PY" "$ROOT/scripts/run_benchmark.py" --data "$ROOT/data/synthetic_320.npz" \
  --out "$ROOT/results/async_validation/synthetic_fishdbc" \
  --methods fishdbc --warmup 20 --arrival-interval-ms "$INTERVAL_MS" \
  --min-cluster-size 3 --min-samples 2 --checkpoint-every 80

"$PY" - "$ROOT/results/async_validation" "$INTERVAL_MS" <<'PY'
import json
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
interval = float(sys.argv[2])
rows = []
for dataset in ("fsdd", "synthetic"):
    for method in ("adaptive", "fishdbc"):
        name = f"{dataset}_{method}"
        path = root / name / ("adaptive_fishdbc_async" if method == "adaptive" else "fishdbc") / "summary.json"
        item = json.loads(path.read_text())
        final = item["final"]
        rows.append({"dataset": dataset, "method": method, "arrival_interval_ms": interval,
                     "decision_steady_mean_ms": item["steady_mean_step_ms"],
                     "decision_steady_p95_ms": item["steady_p95_step_ms"],
                     "maintenance_ms_total": item.get("maintenance_ms_total", 0.0),
                     "max_pending": item.get("max_pending", 0),
                     "max_queue_depth": item.get("max_queue_depth", 0),
                     "recompute_count": item.get("recompute_count", 0),
                     "acc": final["cluster_accuracy_noise_singletons"],
                     "ari": final["ari"], "false_merge_rate": final["false_merge_rate"]})
(root / "async_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
with (root / "async_comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(json.dumps({"status": "ok", "rows": len(rows), "output": str(root / "async_comparison.json")}, ensure_ascii=False))
PY

printf 'ASYNC_VALIDATION_PASS %s\n' "$ROOT/results/async_validation/async_comparison.json"
