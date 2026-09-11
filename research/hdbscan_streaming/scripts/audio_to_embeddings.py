#!/usr/bin/env python3
"""Run the deploy ONNX speaker embedder over a fixed audio manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--ort-threads", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略有效缓存，重新执行全部音频推理。",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "deploy"))
    from speaker_id.onnx_backend import OnnxSpeakerEmbedder

    manifest = args.manifest.resolve()
    raw_root = args.raw_root.resolve()
    model = args.model.resolve()
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty manifest: {manifest}")
    model_sha256 = sha256_file(model)
    manifest_sha256 = sha256_file(manifest)
    if args.out.is_file() and args.meta.is_file() and not args.force:
        try:
            cached_meta = json.loads(args.meta.read_text(encoding="utf-8"))
            cache_valid = (
                cached_meta.get("manifest_sha256") == manifest_sha256
                and cached_meta.get("model_sha256") == model_sha256
                and int(cached_meta.get("ort_threads", -1)) == int(args.ort_threads)
                and int(cached_meta.get("n", -1)) == len(rows)
            )
            if cache_valid:
                cached = np.load(args.out)
                if cached["X"].shape[0] == len(rows):
                    print(json.dumps({"status": "cache_hit", "output": str(args.out.resolve())}, ensure_ascii=False))
                    return
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    embedder = OnnxSpeakerEmbedder(model, ort_threads=args.ort_threads)

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    speakers: list[str] = []
    digits: list[int] = []
    paths: list[str] = []
    elapsed_ms: list[float] = []
    for index, row in enumerate(rows):
        wav_path = raw_root / row["path"]
        start = time.perf_counter()
        vector = np.asarray(embedder.embed_wav_path(wav_path), dtype=np.float32).reshape(-1)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"invalid embedding at row {index}: {wav_path}")
        vectors.append(vector)
        labels.append(int(row["speaker_id"]))
        speakers.append(row["speaker"])
        digits.append(int(row["digit"]))
        paths.append(str(wav_path))

    dimensions = {int(vector.size) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent embedding dimensions: {dimensions}")
    X = np.stack(vectors).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest": str(manifest),
        "raw_root": str(raw_root),
        "model": str(model),
        "model_sha256": model_sha256,
        "manifest_sha256": manifest_sha256,
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "speakers": sorted(set(speakers)),
        "sample_rate_hz": 16000,
        "ort_threads": args.ort_threads,
        "embedding_ms_mean": float(np.mean(elapsed_ms)),
        "embedding_ms_p95": float(np.percentile(elapsed_ms, 95)),
        "embedding_ms_min": float(np.min(elapsed_ms)),
        "embedding_ms_max": float(np.max(elapsed_ms)),
        "output": str(args.out.resolve()),
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_name(args.out.name + ".tmp.npz")
    tmp_meta = args.meta.with_name(args.meta.name + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()
    np.savez_compressed(
        tmp_out,
        X=X,
        y=y,
        speaker_names=np.asarray(speakers),
        digits=np.asarray(digits, dtype=np.int64),
        wav_paths=np.asarray(paths),
        embedding_ms=np.asarray(elapsed_ms, dtype=np.float64),
    )
    os.replace(tmp_out, args.out)
    tmp_meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_meta, args.meta)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
