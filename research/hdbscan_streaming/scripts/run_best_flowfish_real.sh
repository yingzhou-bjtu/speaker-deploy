#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/best_flowfish_real_audio"
rm -rf "$OUT"
mkdir -p "$OUT"

run_one() {
  local name="$1" data="$2" min_samples="$3"
  mkdir -p "$OUT/$name"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/run_benchmark.py" \
    --data "$data" --out "$OUT/$name" \
    --methods adaptive_fishdbc_async --warmup 20 \
    --tradeoff-warmup 20 --tradeoff-margin .02 \
    --tradeoff-max-distance .70 --tradeoff-full-interval 8 \
    --tradeoff-batch-size 4 --arrival-interval-ms 5 \
    --tradeoff-max-queue 8 --min-cluster-size 3 \
    --min-samples "$min_samples" --checkpoint-every 80 \
    >"$OUT/$name/run.log" 2>&1
}

run_one light "$ROOT/data/fsdd/embeddings/fsdd_smoke.npz" 2
run_one medium "$ROOT/data/fsdd/medium/embeddings.npz" 4
run_one heavy "$ROOT/data/fsdd/heavy/embeddings.npz" 4

printf 'scenario,ari,nmi,p95_ms,wall_s,recomputations\n' >"$OUT/summary.csv"
for scenario in light medium heavy; do
  SCENARIO="$scenario" OUTDIR="$OUT" "$ROOT/.venv/bin/python" - <<'PY' >>"$OUT/summary.csv"
import json, os
s=os.environ['SCENARIO']; d=json.load(open(f"{os.environ['OUTDIR']}/{s}/adaptive_fishdbc_async/summary.json")); q=d['final']
print(f"{s},{q['ari']:.6f},{q['nmi']:.6f},{d['steady_p95_step_ms']:.6f},{d['wall_ms']/1000:.6f},{d['recompute_count']}")
PY
done
