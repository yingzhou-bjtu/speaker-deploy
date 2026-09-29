#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
RAW_ROOT="${FSDD_RAW_ROOT:-$ROOT/data/fsdd/raw/free-spoken-digit-dataset-26eb9aaf76e81b692f806f9140c2d2777410d7a1/recordings}"
MODEL="${FSDD_MODEL:-}"
SELECTED="$ROOT/data/fsdd/selected"
SELECTED="${SELECTED_ROOT:-$SELECTED}"
EMBED="${EMBED_PATH:-$ROOT/data/fsdd/embeddings/fsdd_smoke.npz}"
META="${EMBED_META:-$ROOT/data/fsdd/embeddings/fsdd_smoke.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/fsdd_audio_smoke}"
METHODS=(full_hdbscan adaptive_tradeoff adaptive_fishdbc fishdbc ahc sc_pna diart_style)

if [[ ! -x "$PYTHON" ]]; then
  bash "$ROOT/scripts/setup_research_env.sh"
fi
test -d "$RAW_ROOT"
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  printf 'set FSDD_MODEL to a local ONNX speaker-embedding model\n' >&2
  exit 2
fi

"$PYTHON" "$ROOT/scripts/prepare_fsdd_audio.py" \
  --raw-root "$RAW_ROOT" \
  --out-dir "$SELECTED" \
  --per-speaker 5 \
  --seed 20260910

"$PYTHON" "$ROOT/scripts/audio_to_embeddings.py" \
  --manifest "$SELECTED/manifest.csv" \
  --raw-root "$RAW_ROOT" \
  --model "$MODEL" \
  --out "$EMBED" \
  --meta "$META" \
  --ort-threads "${ORT_THREADS:-1}"

"$PYTHON" "$ROOT/scripts/validate_dataset.py" "$EMBED"
"$PYTHON" "$ROOT/scripts/run_benchmark.py" \
  --data "$EMBED" \
  --out "$OUT_ROOT" \
  --methods "${METHODS[@]}" \
  --cap 20 \
  --warmup 12 \
  --min-cluster-size 3 \
  --min-samples 2 \
  --checkpoint-every 6

"$PYTHON" - "$SELECTED/manifest.csv" "$EMBED" "$OUT_ROOT/summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

manifest = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8")))
data = np.load(sys.argv[2], allow_pickle=False)
summaries = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
assert len(manifest) == 30, len(manifest)
assert data["X"].shape[0] == 30, data["X"].shape
assert len({row["speaker"] for row in manifest}) == 6
assert {item["method"] for item in summaries} == {
    "full_hdbscan",
    "adaptive_tradeoff",
    "adaptive_fishdbc",
    "fishdbc",
    "ahc",
    "sc_pna",
    "diart_style",
}
assert all(item["status"] == "ok" for item in summaries), summaries
print("AUDIO_SMOKE_PASS", len(manifest), tuple(data["X"].shape), len(summaries))
PY
