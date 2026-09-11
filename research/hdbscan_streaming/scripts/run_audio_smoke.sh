#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RAW="$ROOT/data/fsdd/raw/free-spoken-digit-dataset-master/recordings"
SELECTED="$ROOT/data/fsdd/selected"
EMBED="$ROOT/data/fsdd/embeddings/fsdd_smoke.npz"
EMBED_META="$ROOT/data/fsdd/embeddings/fsdd_smoke.json"
OUT="$ROOT/results/fsdd_audio_smoke"
MODEL="$ROOT/../deploy/pretrained/eres2net_sv.onnx"
METHODS=(full_hdbscan adaptive_tradeoff adaptive_fishdbc fishdbc ahc sc_pna diart_style)
if [[ "${RUN_FAST_HDBSCAN:-0}" == "1" ]]; then
  METHODS+=(fast_hdbscan)
fi
export EXPECTED_METHODS="${METHODS[*]}"

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
  --methods "${METHODS[@]}" \
  --cap 20 \
  --warmup 12 \
  --min-cluster-size 3 \
  --min-samples 2 \
  --checkpoint-every 6

"$PY" - "$SELECTED/manifest.csv" "$EMBED" "$OUT/summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

manifest = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8")))
data = np.load(sys.argv[2])
summaries = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
assert len(manifest) == 30, len(manifest)
assert data["X"].shape == (30, 192), data["X"].shape
assert len({row["speaker"] for row in manifest}) == 6
expected = set(__import__("os").environ["EXPECTED_METHODS"].split())
assert {item["method"] for item in summaries} == expected, summaries
assert all(item["status"] == "ok" for item in summaries), summaries
print("AUDIO_SMOKE_PASS", len(manifest), data["X"].shape, len(summaries))
PY
