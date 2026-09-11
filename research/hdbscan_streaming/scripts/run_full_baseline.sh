#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RAW="$ROOT/data/fsdd/raw/free-spoken-digit-dataset-master/recordings"
SELECTED="$ROOT/data/fsdd/selected"
EMBED="$ROOT/data/fsdd/embeddings/fsdd_smoke.npz"
EMBED_META="$ROOT/data/fsdd/embeddings/fsdd_smoke.json"
OUT="$ROOT/results/fsdd_full_baseline"
MODEL="$ROOT/../deploy/pretrained/eres2net_sv.onnx"
MIN_CLUSTER_SIZE=3
MIN_SAMPLES=2

test -x "$PY"
test -d "$RAW"
test -f "$MODEL"

"$PY" "$ROOT/scripts/prepare_fsdd_audio.py" \
  --raw-root "$RAW" \
  --out-dir "$SELECTED" \
  --per-speaker 5 \
  --seed 20260910

"$PY" "$ROOT/scripts/audio_to_embeddings.py" \
  --manifest "$SELECTED/manifest.csv" \
  --raw-root "$RAW" \
  --model "$MODEL" \
  --out "$EMBED" \
  --meta "$EMBED_META" \
  --ort-threads 1

"$PY" "$ROOT/scripts/validate_dataset.py" "$EMBED"

"$PY" "$ROOT/scripts/run_benchmark.py" \
  --data "$EMBED" \
  --out "$OUT" \
  --methods full_hdbscan \
  --cap 20 \
  --warmup 12 \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --min-samples "$MIN_SAMPLES" \
  --checkpoint-every 6

"$PY" "$ROOT/scripts/verify_full_against_deploy.py" \
  --data "$EMBED" \
  --summary "$OUT/summary.json" \
  --model "$MODEL" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --min-samples "$MIN_SAMPLES" \
  --ort-threads 1 | tee "$OUT/deploy_equivalence.json"

printf 'FULL_BASELINE_PASS %s\n' "$OUT"
