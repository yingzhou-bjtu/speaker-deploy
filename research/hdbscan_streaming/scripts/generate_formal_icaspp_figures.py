#!/usr/bin/env python3
"""Generate a consistent, reviewer-first figure set for the ICASSP draft."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    }
)

COLORS = {
    "Full": "#5B6470",
    "Fish": "#D9822B",
    "Ours": "#007F86",
    "adaptive_fishdbc_async": "#007F86",
    "fishdbc": "#D9822B",
}
LABELS = {
    "Full": "Full HDBSCAN",
    "Fish": "FISHDBC",
    "Ours": "FlowFish",
    "adaptive_fishdbc_async": "FlowFish",
    "fishdbc": "FISHDBC",
}
WORKLOADS = ["FSDD", "Synth-320", "Drift-320", "Drift-640"]
WORKLOAD_KEYS = ["FSDD", "S320", "D320", "D640"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setup_axis(ax, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, linestyle=(0, (2, 3)), linewidth=0.55,
                color="#AAB2B8", alpha=0.62)
    ax.tick_params(direction="in", length=3.2, width=0.75, top=True, right=True,
                   pad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#30343B")


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.025,
                dpi=300)
    plt.close(fig)


def plot_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 1.62))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blocks = [
        (0.02, 0.29, 0.17, 0.42, "Embedding\nstream", "#F4F6F8", "#59636E"),
        (0.27, 0.29, 0.19, 0.42, "FAST gate\n$d_1, d_2, m$", "#E7F3F2", COLORS["Ours"]),
        (0.55, 0.29, 0.19, 0.42, "Bounded FIFO\nqueue", "#FFF1E2", COLORS["Fish"]),
        (0.83, 0.29, 0.15, 0.42, "Committed\nreference", "#EEF1F5", "#59636E"),
    ]
    for x, y, w, h, text, fill, edge in blocks:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=fill, edgecolor=edge, linewidth=1.15,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color="#27313A", linespacing=1.25, fontweight="bold")
    arrows = [
        (0.19, 0.50, 0.27, 0.50, "arrival"),
        (0.46, 0.58, 0.55, 0.58, "REFRESH"),
        (0.46, 0.42, 0.83, 0.42, "FAST label"),
        (0.74, 0.50, 0.83, 0.50, "FIFO commit"),
    ]
    for x1, y1, x2, y2, label in arrows:
        color = COLORS["Ours"] if label == "FAST label" else "#59636E"
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color=color, connectionstyle="arc3,rad=0.0",
        ))
        ax.text((x1 + x2) / 2, y1 + (0.055 if y1 >= 0.5 else -0.07), label,
                ha="center", va="center", fontsize=7.2, color=color)
    ax.text(0.365, 0.12, "confident arrivals stay on the foreground path",
            ha="center", va="center", fontsize=7.4, color=COLORS["Ours"])
    ax.text(0.645, 0.88, "uncertain arrivals are ordered and refreshed",
            ha="center", va="center", fontsize=7.4, color=COLORS["Fish"])
    save(fig, out, "fig_pipeline_formal")


def plot_main(out: Path, rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.35), sharex=True,
                             gridspec_kw={"wspace": 0.32})
    x = np.arange(len(WORKLOADS))
    methods = ["Full", "Fish", "Ours"]
    offsets = [-0.22, 0.0, 0.22]
    width = 0.21
    for method, offset in zip(methods, offsets):
        values = [float(next(r for r in rows if r["workload"] == key and r["method"] == method)["p95_ms"])
                  for key in WORKLOAD_KEYS]
        axes[0].bar(x + offset, values, width=width, color=COLORS[method],
                    edgecolor="white", linewidth=0.35, label=LABELS[method])
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.25, 80)
    axes[0].set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    axes[0].set_ylabel("Foreground P95 (ms)")
    axes[0].set_title("Latency")

    for method in methods:
        values = [float(next(r for r in rows if r["workload"] == key and r["method"] == method)["ari"])
                  for key in WORKLOAD_KEYS]
        axes[1].plot(x, values, marker="o", markersize=4.2, linewidth=1.7,
                     color=COLORS[method], label=LABELS[method])
    axes[1].set_ylim(0.55, 1.03)
    axes[1].set_yticks([0.6, 0.8, 1.0])
    axes[1].set_ylabel("ARI")
    axes[1].set_title("Cluster quality")

    for method in methods:
        values = [float(next(r for r in rows if r["workload"] == key and r["method"] == method)["fmr"])
                  for key in WORKLOAD_KEYS]
        axes[2].plot(x, values, marker="o", markersize=4.2, linewidth=1.7,
                     color=COLORS[method], label=LABELS[method])
    axes[2].set_ylim(-0.001, 0.031)
    axes[2].set_yticks([0.00, 0.01, 0.02, 0.03])
    axes[2].set_ylabel("False merge rate")
    axes[2].set_title("Error control")

    for ax in axes:
        ax.set_xticks(x, WORKLOADS, rotation=25, ha="right")
        setup_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=False, handlelength=1.4, columnspacing=1.5)
    save(fig, out, "fig_main_formal")


def plot_rate(out: Path, rows: list[dict[str, str]]) -> None:
    ours = sorted([r for r in rows if r["method"] == "adaptive_fishdbc_async"],
                  key=lambda r: float(r["interval_ms"]))
    fish = sorted([r for r in rows if r["method"] == "fishdbc"],
                  key=lambda r: float(r["interval_ms"]))
    intervals = np.array([float(r["interval_ms"]) for r in ours])
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.85),
                             gridspec_kw={"wspace": 0.40})
    axes[0].plot(intervals, [float(r["decision_p95_ms"]) for r in fish],
                 marker="o", markersize=4.5, linewidth=1.8, color=COLORS["Fish"],
                 label="FISHDBC")
    axes[0].plot(intervals, [float(r["decision_p95_ms"]) for r in ours],
                 marker="s", markersize=4.5, linewidth=1.8, color=COLORS["Ours"],
                 label="FlowFish")
    axes[0].set_xlabel("Arrival interval (ms)")
    axes[0].set_ylabel("Foreground P95 (ms)")
    axes[0].set_ylim(0, 18)
    axes[0].set_title("Responsiveness under load")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.4)

    width = 0.52
    axes[1].bar(intervals, [float(r["max_queue_depth"]) for r in ours], width=width,
                color=COLORS["Ours"], alpha=0.86, label="Max queue depth")
    axes[1].axhline(8, color="#59636E", linestyle=(0, (3, 2)), linewidth=1.0,
                    label="Queue capacity")
    axes[1].set_xlabel("Arrival interval (ms)")
    axes[1].set_ylabel("Queue depth")
    axes[1].set_ylim(0, 9.5)
    axes[1].set_yticks([0, 2, 4, 6, 8])
    axes[1].set_title("Bounded background work")
    axes[1].legend(frameon=False, loc="upper right", handlelength=1.4)
    for ax in axes:
        setup_axis(ax)
        ax.set_xticks(intervals, [str(int(v)) for v in intervals])
    save(fig, out, "fig_rate_formal")


def plot_frontier(out: Path, rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    markers = {"Full": "^", "Fish": "o", "Ours": "s"}
    offsets = {"Full": (4, 4), "Fish": (4, -10), "Ours": (4, 5)}
    for method in ("Full", "Fish", "Ours"):
        for key, label in zip(WORKLOAD_KEYS, WORKLOADS):
            row = next(r for r in rows if r["workload"] == key and r["method"] == method)
            x = float(row["p95_ms"])
            y = float(row["ari"])
            ax.scatter(x, y, s=38, marker=markers[method], color=COLORS[method],
                       edgecolor="white", linewidth=0.55, zorder=3)
            if method == "Ours" or (method == "Full" and key == "D640"):
                ax.annotate(label, (x, y), xytext=offsets[method],
                            textcoords="offset points", fontsize=6.8,
                            color="#30343B")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 70)
    ax.set_ylim(0.55, 1.03)
    ax.set_xlabel("Foreground P95 (ms)")
    ax.set_ylabel("ARI")
    ax.set_title("Quality--latency frontier")
    setup_axis(ax, grid_axis="both")
    handles = [Line2D([0], [0], marker=markers[m], color="none",
                      markerfacecolor=COLORS[m], markeredgecolor="white",
                      markersize=6, label=LABELS[m]) for m in ("Full", "Fish", "Ours")]
    ax.legend(handles=handles, frameon=False, loc="lower right", handletextpad=0.35)
    save(fig, out, "fig_frontier_formal")


def plot_multiseed(out: Path, rows: list[dict[str, str]]) -> None:
    seeds = sorted({r["seed"] for r in rows})
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.85),
                             gridspec_kw={"wspace": 0.40})
    for method in ("fishdbc", "adaptive_fishdbc_async"):
        subset = {r["seed"]: r for r in rows if r["method"] == method}
        label = LABELS[method]
        axes[0].plot(x, [float(subset[s]["decision_p95_ms"]) for s in seeds],
                     marker="o", markersize=4.2, linewidth=1.6,
                     color=COLORS[method], label=label)
        axes[1].plot(x, [float(subset[s]["ari"]) for s in seeds],
                     marker="o", markersize=4.2, linewidth=1.6,
                     color=COLORS[method], label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.25, 15)
    axes[0].set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    axes[0].set_ylabel("P95 latency (ms)")
    axes[0].set_title("Paired seed latency")
    axes[1].set_ylim(0.65, 0.95)
    axes[1].set_yticks([0.7, 0.8, 0.9])
    axes[1].set_ylabel("ARI")
    axes[1].set_title("Paired seed quality")
    for ax in axes:
        ax.set_xlabel("Seed")
        ax.set_xticks(x, [s[-2:] for s in seeds])
        setup_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=2, frameon=False, handlelength=1.4)
    save(fig, out, "fig_multiseed_formal")


def plot_sensitivity(out: Path, rows: list[dict[str, str]]) -> None:
    distances = sorted({float(r["max_distance"]) for r in rows})
    intervals = sorted({int(float(r["full_interval"])) for r in rows})
    matrices = []
    for field in ("steady_mean_step_ms", "false_merge_rate"):
        matrices.append(np.array([
            [float(next(r for r in rows if float(r["max_distance"]) == distance and
                      int(float(r["full_interval"])) == interval)[field])
             for distance in distances]
            for interval in intervals
        ]))
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.85),
                             gridspec_kw={"wspace": 0.58})
    titles = ["Mean decision cost (ms)", "False merge rate"]
    cmaps = ["YlGnBu", "OrRd"]
    for ax, matrix, title, cmap in zip(axes, matrices, titles, cmaps):
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(distances)), [f"{v:.2f}" for v in distances])
        ax.set_yticks(range(len(intervals)), [str(v) for v in intervals])
        ax.set_xlabel("Distance threshold $\\delta$")
        ax.set_ylabel("Full interval")
        ax.set_title(title)
        for iy in range(matrix.shape[0]):
            for ix in range(matrix.shape[1]):
                value = matrix[iy, ix]
                text = f"{value:.2f}" if title.startswith("Mean") else f"{value:.3f}"
                ax.text(ix, iy, text, ha="center", va="center", fontsize=7,
                        color="white" if value > np.quantile(matrix, 0.62) else "#27313A")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=6.5, length=2)
        setup_axis(ax, grid_axis="none")
    save(fig, out, "fig_sensitivity_formal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(args.tables / "canonical_main.csv")
    rate_rows = read_csv(args.tables / "canonical_rate.csv")
    seed_rows = read_csv(args.tables / "canonical_multiseed.csv")
    tradeoff_rows = read_csv(args.tables / "canonical_tradeoff.csv")
    plot_pipeline(args.figures)
    plot_main(args.figures, main_rows)
    plot_rate(args.figures, rate_rows)
    plot_frontier(args.figures, main_rows)
    plot_multiseed(args.figures, seed_rows)
    plot_sensitivity(args.figures, tradeoff_rows)
    print({"status": "ok", "figures": 6, "output": str(args.figures)})


if __name__ == "__main__":
    main()
