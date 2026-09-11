#!/usr/bin/env python3
"""Plot async decision latency and maintenance cost versus arrival rate."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": 0.8, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linestyle": ":", "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "savefig.dpi": 300})
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    intervals = sorted({float(r["interval_ms"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = {"adaptive_fishdbc_async": "#009e73", "fishdbc": "#2868f8"}
    names = {"adaptive_fishdbc_async": "Adaptive-F async", "fishdbc": "FISHDBC"}
    for method in colors:
        subset = [r for r in rows if r["method"] == method]
        subset.sort(key=lambda r: float(r["interval_ms"]))
        axes[0].plot([float(r["interval_ms"]) for r in subset], [float(r["decision_p95_ms"]) for r in subset], marker="o", linewidth=1.8, markersize=4, color=colors[method], label=names[method])
    axes[0].set_xlabel("Arrival interval (ms)"); axes[0].set_ylabel("Decision P95 (ms)")
    axes[0].set_title("(a) Tail latency"); axes[0].legend(frameon=False, loc="best")
    adaptive = [r for r in rows if r["method"] == "adaptive_fishdbc_async"]
    adaptive.sort(key=lambda r: float(r["interval_ms"]))
    ax = axes[1]
    ax.plot([float(r["interval_ms"]) for r in adaptive], [float(r["maintenance_ms_total"]) for r in adaptive], marker="s", linewidth=1.8, markersize=4, color="#d97706", label="Maintenance")
    ax2 = ax.twinx()
    ax2.plot([float(r["interval_ms"]) for r in adaptive], [float(r.get("max_queue_depth", r["max_pending"])) for r in adaptive], marker="^", linewidth=1.5, markersize=4, color="#7c3aed", label="Max queue depth")
    ax.set_xlabel("Arrival interval (ms)"); ax.set_ylabel("Maintenance time (ms)", color="#d97706")
    ax2.set_ylabel("Max queue depth", color="#7c3aed"); ax.set_title("(b) Background cost")
    ax.tick_params(axis="y", labelcolor="#d97706"); ax2.tick_params(axis="y", labelcolor="#7c3aed")
    fig.tight_layout(pad=0.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight"); fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"RATE_SWEEP_PLOT_PASS {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
