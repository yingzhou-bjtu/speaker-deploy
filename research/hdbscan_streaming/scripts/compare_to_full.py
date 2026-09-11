#!/usr/bin/env python3
"""Compare optimization candidate summaries against a Full HDBSCAN baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


QUALITY_KEYS = (
    "cluster_accuracy",
    "cluster_accuracy_noise_singletons",
    "ari",
    "nmi",
    "pairwise_precision",
    "pairwise_recall",
    "false_merge_rate",
    "coverage",
)
METHOD_ROLES = {
    "full_hdbscan": "full_reference",
    "adaptive_tradeoff": "adaptive_tradeoff_candidate",
    "adaptive_fishdbc": "proposed_risk_gated_fishdbc",
    "adaptive_fishdbc_async": "proposed_async_risk_gated_fishdbc",
    "compressed_hdbscan": "cache_compression_candidate",
    "window_hdbscan": "bounded_window_baseline",
    "fishdbc": "incremental_mst_candidate",
    "approx_predict": "fixed_model_prediction_baseline",
    "fast_hdbscan": "static_backend_control",
    "ahc": "classic_agglomerative_baseline",
    "sc_pna": "ICASSP2025_spectral_baseline",
    "diart_style": "DIART_style_online_baseline",
}


def load_summary(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    return list(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    full_items = [
        item for item in load_summary(args.full) if item.get("method") == "full_hdbscan"
    ]
    if len(full_items) != 1:
        raise ValueError("--full must contain exactly one full_hdbscan summary")
    full = full_items[0]
    full_final = full["final"]

    rows: list[dict[str, Any]] = []
    all_items = [full]
    all_items.extend(
        item
        for item in load_summary(args.candidates)
        if item.get("method") != "full_hdbscan"
    )
    full_steady = float(full.get("steady_mean_step_ms", full["mean_step_ms"]))
    full_steady_p95 = float(full.get("steady_p95_step_ms", full["p95_step_ms"]))
    for item in all_items:
        final = item.get("final", {})
        item_mean = float(item.get("steady_mean_step_ms", item.get("mean_step_ms", 0.0)))
        row: dict[str, Any] = {
            "method": item.get("method"),
            "role": METHOD_ROLES.get(item.get("method"), "candidate"),
            "status": item.get("status"),
            "steady_mean_step_ms": item_mean,
            "full_steady_mean_step_ms": full_steady,
            "speedup_vs_full": (
                full_steady / item_mean
                if item.get("status") == "ok" and item_mean > 0
                else 0.0
            ),
            "steady_p95_step_ms": float(
                item.get("steady_p95_step_ms", item.get("p95_step_ms", 0.0))
            ),
            "full_steady_p95_step_ms": full_steady_p95,
            "distance_comparisons": int(item.get("distance_comparisons", 0)),
            "cheap_update_count": int(item.get("cheap_update_count", 0)),
            "full_trigger_count": int(item.get("full_trigger_count", 0)),
            "uncertainty_count": int(item.get("uncertainty_count", 0)),
            "local_update_count": int(item.get("local_update_count", 0)),
            "full_trigger_rate": float(item.get("full_trigger_rate", 0.0)),
            "cheap_update_rate": float(item.get("cheap_update_rate", 0.0)),
            "distance_comparisons_per_input": float(
                item.get("distance_comparisons_per_input", 0.0)
            ),
        }
        for key in QUALITY_KEYS:
            row[key] = final.get(key)
            row[f"delta_{key}"] = (
                float(final[key]) - float(full_final[key])
                if key in final and key in full_final
                else None
            )
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": "ok", "rows": len(rows), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
