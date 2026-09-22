#!/usr/bin/env python3
"""Generate a deterministic stream with late speakers and gradual drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x), 1e-12)


def build_stream(speakers: int, points: int, dim: int, seed: int, drift: float, noise: float,
                 max_center_similarity: float = 0.45):
    rng = np.random.default_rng(seed)
    centers = []
    while len(centers) < speakers:
        candidate = unit(rng.normal(size=dim))
        if all(float(candidate @ c) < max_center_similarity for c in centers):
            centers.append(candidate)
    directions = [unit(rng.normal(size=dim)) for _ in range(speakers)]
    rows, labels = [], []
    split = points // 2
    for t in range(points):
        active = min(speakers, 4 if t < split else speakers)
        sid = t % active
        progress = t / max(1, points - 1)
        center = unit(centers[sid] + drift * progress * directions[sid])
        rows.append(unit(center + noise * rng.normal(size=dim)).astype(np.float32))
        labels.append(sid)
    meta = {"seed": seed, "speakers": speakers, "points": points, "dim": dim,
            "drift": drift, "noise": noise, "late_speakers": speakers - 4,
            "max_center_similarity": max_center_similarity,
            "order": "first four speakers, then late speakers with gradual drift"}
    return np.asarray(rows), np.asarray(labels, dtype=np.int64), meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--speakers", type=int, default=8)
    p.add_argument("--points", type=int, default=320)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--drift", type=float, default=0.20)
    p.add_argument("--noise", type=float, default=0.12)
    p.add_argument("--max-center-similarity", type=float, default=0.45)
    args = p.parse_args()
    X, y, meta = build_stream(args.speakers, args.points, args.dim, args.seed, args.drift,
                               args.noise, args.max_center_similarity)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.meta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    meta = {**meta, "n": int(len(y)), "out": str(args.out.resolve())}
    args.meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
