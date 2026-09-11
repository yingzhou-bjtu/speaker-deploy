#!/usr/bin/env python3
"""Create compact quality/latency/action plots from a benchmark comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", "None") else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.usetex": False,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "savefig.dpi": 300,
    })

    rows = read_rows(args.comparison)
    names = [row["method"] for row in rows]
    labels = {"full_hdbscan": "Full", "adaptive_tradeoff": "Adaptive", "adaptive_fishdbc": "Adaptive-F", "adaptive_fishdbc_async": "Adaptive-F async", "fishdbc": "FISHDBC", "ahc": "AHC", "sc_pna": "SC-pNA", "diart_style": "DIART-style"}
    colors_by_method = {"full_hdbscan": "#263238", "adaptive_tradeoff": "#007c83", "adaptive_fishdbc": "#009e73", "adaptive_fishdbc_async": "#00a6a6", "fishdbc": "#2868f8", "ahc": "#d97706", "sc_pna": "#7c3aed", "diart_style": "#c2415b"}
    colors = [colors_by_method.get(name, "#6b7280") for name in names]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))

    ax = axes[0]
    label_offsets = {
        "full_hdbscan": (4, -15), "adaptive_tradeoff": (4, -15),
        "fishdbc": (-42, 8), "ahc": (4, -15), "sc_pna": (4, -15),
        "diart_style": (4, 8), "adaptive_fishdbc": (4, 8),
    }
    for row, color in zip(rows, colors):
        ax.scatter(f(row, "steady_mean_step_ms"), f(row, "cluster_accuracy_noise_singletons"), color=color, s=34, zorder=3)
        ax.annotate(labels.get(row["method"], row["method"]), (f(row, "steady_mean_step_ms"), f(row, "cluster_accuracy_noise_singletons")), xytext=label_offsets.get(row["method"], (4, 4)), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Steady latency (ms, log)")
    ax.set_ylabel("ACC (noise as singleton)")
    ax.set_title("(a) Speed-quality frontier")

    ax = axes[1]
    y_pos = list(range(len(rows)))
    ax.barh(y_pos, [f(row, "speedup_vs_full") for row in rows], color=colors, height=0.62)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#374151", linewidth=0.8)
    ax.set_xlabel("Speedup vs. Full (log)")
    ax.set_yticks(y_pos, [labels.get(name, name) for name in names])
    ax.invert_yaxis()
    ax.set_title("(b) Relative latency")
    ax = axes[2]
    adaptive_dir = args.results / "adaptive_tradeoff"
    trace_path = adaptive_dir / "trace.csv"
    if trace_path.exists():
        trace = read_rows(trace_path)
        actions = [row["action"] for row in trace]
        action_codes = {name: i for i, name in enumerate(sorted(set(actions)))}
        ax.step([int(row["step"]) for row in trace], [action_codes[action] for action in actions], where="post", color="#0f766e")
        action_labels = {"tradeoff_warmup": "Warm-up", "tradeoff_initial_full": "Initial Full", "tradeoff_cached_assign": "Cache assign", "tradeoff_uncertainty_full": "Risk Full"}
        ax.set_yticks(list(action_codes.values()), [action_labels.get(x, x) for x in action_codes], fontsize=7)
        ax.set_xlabel("Stream step")
        ax.set_title("(c) Adaptive path")
    else:
        ax.text(0.5, 0.5, "adaptive trace missing", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout(pad=0.5)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    pdf_path = args.out.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    report = {
        "comparison": str(args.comparison.resolve()),
        "plot": str(args.out.resolve()),
        "pdf": str(pdf_path.resolve()),
        "methods": names,
        "adaptive_trace": str(trace_path.resolve()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
