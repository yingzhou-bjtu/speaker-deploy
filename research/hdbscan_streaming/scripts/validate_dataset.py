#!/usr/bin/env python3
"""Validate the common X/y npz contract used by all benchmark adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    args = p.parse_args()
    d = np.load(args.data)
    if "X" not in d or "y" not in d:
        raise ValueError("npz 必须包含 X 和 y")
    X = np.asarray(d["X"])
    y = np.asarray(d["y"])
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"shape 不合法: X={X.shape}, y={y.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X 含 NaN 或 Inf")
    result = {
        "path": str(args.data.resolve()),
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "dtype": str(X.dtype),
        "speakers": int(np.unique(y).size),
        "class_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
        "row_norm_min": float(np.linalg.norm(X, axis=1).min()),
        "row_norm_max": float(np.linalg.norm(X, axis=1).max()),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
