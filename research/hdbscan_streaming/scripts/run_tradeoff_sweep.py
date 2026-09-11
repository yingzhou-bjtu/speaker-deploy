#!/usr/bin/env python3
"""Sweep adaptive tradeoff knobs and emit a quality/compute table."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-distances", type=float, nargs="+", default=[0.70, 0.85, 0.95])
    p.add_argument("--intervals", type=int, nargs="+", default=[4, 8, 16])
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = root / ".venv/bin/python"
    args.out.mkdir(parents=True, exist_ok=True)
    full_items = [x for x in json.loads(args.full.read_text()) if x.get("method") == "full_hdbscan"]
    if len(full_items) != 1:
        raise ValueError("full summary must contain one Full item")
    full_ms = float(full_items[0]["steady_mean_step_ms"])
    rows = []
    for max_distance in args.max_distances:
        for interval in args.intervals:
            name = f"d{max_distance:.2f}_i{interval}"
            run_dir = args.out / name
            cmd = [
                str(py), str(root / "scripts/run_benchmark.py"),
                "--data", str(args.data), "--out", str(run_dir),
                "--methods", "adaptive_tradeoff", "--warmup", "12",
                "--tradeoff-warmup", "12", "--tradeoff-margin", "0.02",
                "--tradeoff-max-distance", str(max_distance),
                "--tradeoff-full-interval", str(interval),
                "--min-cluster-size", "3", "--min-samples", "2",
                "--checkpoint-every", "6",
            ]
            subprocess.run(cmd, cwd=root, check=True, stdout=subprocess.DEVNULL)
            item = json.loads((run_dir / "adaptive_tradeoff/summary.json").read_text())
            final = item["final"]
            rows.append({
                "max_distance": max_distance, "full_interval": interval,
                "steady_mean_step_ms": item["steady_mean_step_ms"],
                "speedup_vs_full": full_ms / item["steady_mean_step_ms"],
                "cluster_accuracy_noise_singletons": final["cluster_accuracy_noise_singletons"],
                "ari": final["ari"], "nmi": final["nmi"],
                "false_merge_rate": final["false_merge_rate"], "coverage": final["coverage"],
                "full_trigger_count": item["full_trigger_count"],
                "cheap_update_count": item["cheap_update_count"],
                "distance_comparisons": item["distance_comparisons"],
                "run_dir": str(run_dir.resolve()),
            })
    out_csv = args.out / "tradeoff_sweep.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out / "tradeoff_sweep.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "csv": str(out_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
