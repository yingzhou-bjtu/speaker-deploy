#!/usr/bin/env python3
"""Convert an existing deploy embedding store into the common X/y contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="保留 true_speaker_slot<0 的行；保留后这些行不适合监督指标。",
    )
    args = p.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "E" not in data:
        raise ValueError("输入 npz 缺少 E")
    X = np.asarray(data["E"], dtype=np.float32)
    y = np.asarray(
        data["true_speaker_slot"]
        if "true_speaker_slot" in data
        else np.full(len(X), -1, dtype=np.int64),
        dtype=np.int64,
    )
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"输入 shape 不合法: E={X.shape}, y={y.shape}")
    if not args.keep_unlabeled:
        keep = y >= 0
        X, y = X[keep], y[keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"saved {args.out.resolve()} X={X.shape} speakers={np.unique(y).size}")


if __name__ == "__main__":
    main()
