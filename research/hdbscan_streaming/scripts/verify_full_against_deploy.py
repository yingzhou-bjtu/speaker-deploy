#!/usr/bin/env python3
"""Verify the benchmark Full HDBSCAN path against deploy clustering semantics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from run_benchmark import _fit_labels, l2_row_normalize, pairwise_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--min-cluster-size", type=int, required=True)
    parser.add_argument("--min-samples", type=int, required=True)
    parser.add_argument("--ort-threads", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "deploy"))
    from speaker_id.engine import SpeakerEmbedCluster

    data = np.load(args.data, allow_pickle=True)
    X_raw = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    X = l2_row_normalize(X_raw)
    bench_dense = _fit_labels(
        X,
        "hdbscan",
        args.min_cluster_size,
        args.min_samples,
    )

    engine = SpeakerEmbedCluster(
        backend="onnx",
        onnx_path=args.model,
        ort_threads=args.ort_threads,
    )
    cr = engine.cluster_speakers(
        X_raw,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    deploy_dense = np.asarray(cr.dense_labels, dtype=np.int64)
    deploy_public = np.asarray(cr.labels, dtype=np.int64)
    if not np.array_equal(bench_dense, deploy_dense):
        mismatch = np.flatnonzero(bench_dense != deploy_dense).tolist()
        raise AssertionError(
            f"benchmark Full dense labels differ from deploy at rows {mismatch[:20]}"
        )

    summaries = json.loads(args.summary.read_text(encoding="utf-8"))
    if isinstance(summaries, dict):
        summaries = [summaries]
    full_items = [item for item in summaries if item.get("method") == "full_hdbscan"]
    if len(full_items) != 1:
        raise AssertionError("summary must contain exactly one full_hdbscan item")
    final = dict(full_items[0]["final"])
    dense_metrics = pairwise_metrics(y, bench_dense)
    deploy_metrics = pairwise_metrics(y, deploy_public)
    checks = {
        "cluster_accuracy": dense_metrics["cluster_accuracy"],
        "cluster_accuracy_non_noise": dense_metrics["cluster_accuracy_non_noise"],
        "cluster_accuracy_noise_singletons": dense_metrics[
            "cluster_accuracy_noise_singletons"
        ],
        "ari": dense_metrics["ari"],
        "nmi": dense_metrics["nmi"],
        "coverage": dense_metrics["coverage"],
    }
    for key, expected in checks.items():
        observed = float(final[key])
        if abs(observed - float(expected)) > 1e-12:
            raise AssertionError(f"{key} mismatch: summary={observed}, expected={expected}")

    report = {
        "status": "ok",
        "data": str(args.data.resolve()),
        "n": int(len(y)),
        "dim": int(X_raw.shape[1]),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "benchmark_matches_deploy_dense_labels": True,
        "dense_metrics": dense_metrics,
        "deploy_public_label_metrics": deploy_metrics,
        "deploy_noise_count": int(np.sum(cr.hdbscan_was_noise)),
        "deploy_public_cluster_count": int(len(np.unique(deploy_public))),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
