#!/usr/bin/env python3
"""Generate canonical ICASSP tables and LoongX-aligned result figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Circle


FIG_WIDTH_IN = 4.78
FIG_HEIGHT_IN = 3.75
SCENE_WIDTH_IN = 3.30
SCENE_HEIGHT_IN = 1.95
STYLE = {
    "full": "#000000",
    "fish": "#F9413D",
    "ours": "#3071F4",
    "orange": "#FF9900",
    "purple": "#886ABE",
    "pink": "#F781BE",
    "teal": "#3DB19E",
    "cyan": "#00D2CD",
    "grid": "#C7C7C7",
    "text": "#000000",
}


def style_axis(ax) -> None:
    ax.grid(True, which="major", axis="both", linestyle="--", color=STYLE["grid"], linewidth=0.85, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(STYLE["text"])
        spine.set_linewidth(1.05)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4.0,
        width=1.0,
        top=True,
        right=True,
        color=STYLE["text"],
        labelcolor=STYLE["text"],
        pad=2.5,
    )


def rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def find_method(path: Path, method: str) -> dict:
    return next(item for item in rows(path) if item["method"] == method)


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def draw_scene_figure(path: Path) -> None:
    """Draw the compact application-to-task abstraction used on page one."""
    fig, ax = plt.subplots(figsize=(SCENE_WIDTH_IN, SCENE_HEIGHT_IN))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_edge = STYLE["text"]
    panel_fill = "#FAFBFC"
    scene_fill = "#F3F6F8"
    audio_fill = "#EAF5F4"
    cluster_fill = "#EEF2F8"
    muted = "#59636E"

    panels = [
        (0.025, 0.22, 0.275, 0.56, scene_fill),
        (0.365, 0.22, 0.275, 0.56, audio_fill),
        (0.725, 0.22, 0.25, 0.56, cluster_fill),
    ]
    for x, y, w, h, fill in panels:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                               edgecolor=panel_edge, linewidth=1.0))

    # Home scene: a minimal house and listening robot pictogram.
    ax.add_patch(Polygon([[0.065, 0.53], [0.145, 0.67], [0.225, 0.53]],
                         closed=True, facecolor="white", edgecolor=STYLE["ours"],
                         linewidth=1.15))
    ax.add_patch(Rectangle((0.085, 0.36), 0.12, 0.18, facecolor="white",
                           edgecolor=STYLE["ours"], linewidth=1.15))
    ax.add_patch(Rectangle((0.142, 0.36), 0.035, 0.10, facecolor=STYLE["ours"],
                           edgecolor=STYLE["ours"], linewidth=0.8))
    ax.add_patch(Circle((0.175, 0.47), 0.032, facecolor=STYLE["orange"],
                        edgecolor=panel_edge, linewidth=0.8))
    ax.plot([0.175, 0.175], [0.50, 0.55], color=panel_edge, linewidth=0.8)
    ax.text(0.162, 0.60, "mic", ha="center", va="center", fontsize=8.2,
            color=muted)
    ax.text(0.162, 0.26, "Home scene", ha="center", va="center",
            fontsize=9.5, fontweight="bold")

    # Live audio: waveform conveys the stream without adding a second diagram.
    wave_x = np.linspace(0.405, 0.60, 80)
    wave_y = 0.50 + 0.095 * np.sin(np.linspace(0, 8 * np.pi, 80)) * \
        (0.45 + 0.55 * np.sin(np.linspace(0, np.pi, 80)) ** 2)
    ax.plot(wave_x, wave_y, color=STYLE["teal"], linewidth=1.5)
    ax.plot([0.405, 0.60], [0.50, 0.50], color="#B8C6C8", linewidth=0.55,
            linestyle=(0, (2, 2)))
    ax.text(0.502, 0.33, "Live speech", ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    ax.text(0.502, 0.69, "audio stream", ha="center", va="center",
            fontsize=8.2, color=muted)

    # Speaker identities: two compact, separable identity markers.
    for cx, color, label in ((0.785, STYLE["ours"], "S1"),
                             (0.905, STYLE["orange"], "S2")):
        ax.add_patch(Circle((cx, 0.53), 0.036, facecolor=color,
                            edgecolor=panel_edge, linewidth=0.8))
        ax.add_patch(Rectangle((cx - 0.052, 0.37), 0.104, 0.07,
                               facecolor="white", edgecolor=color,
                               linewidth=1.0))
        ax.text(cx, 0.405, label, ha="center", va="center", fontsize=8.2,
                fontweight="bold", color=color)
    ax.text(0.85, 0.26, "Speaker identities", ha="center", va="center",
            fontsize=9.5, fontweight="bold")

    arrow_style = dict(arrowstyle="-|>", mutation_scale=9, linewidth=1.1,
                       color=panel_edge, shrinkA=3, shrinkB=3)
    ax.add_patch(FancyArrowPatch((0.31, 0.50), (0.36, 0.50), **arrow_style))
    ax.add_patch(FancyArrowPatch((0.64, 0.50), (0.72, 0.50), **arrow_style))
    ax.text(0.85, 0.91, "Who spoke?", ha="center", va="center", fontsize=11.0,
            fontweight="bold", color=STYLE["ours"])
    ax.plot([0.85, 0.85], [0.84, 0.78], color=STYLE["ours"], linewidth=0.9)

    fig.tight_layout(pad=0.25)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit.resolve()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)

    workloads = [
        ("FSDD", "fsdd_full_baseline/summary.json", "full_hdbscan"),
        ("S320", "synthetic_320/summary.json", "full_hdbscan"),
        ("D320", "drift_320/summary.json", "full_hdbscan"),
        ("D640", "drift_640/summary.json", "full_hdbscan"),
    ]
    main_records = []
    for workload, full_path, _ in workloads:
        if workload == "FSDD":
            fish = find_method(audit / "fsdd_candidates/summary.json", "fishdbc")
            ours = find_method(audit / "fsdd_audio_ours/summary.json", "adaptive_fishdbc_async")
        else:
            fish = find_method(audit / full_path, "fishdbc")
            ours = find_method(audit / full_path, "adaptive_fishdbc_async")
        full = find_method(audit / full_path, "full_hdbscan")
        for label, item in (("Full", full), ("Fish", fish), ("Ours", ours)):
            final = item["final"]
            main_records.append({
                "workload": workload,
                "method": label,
                "p95_ms": f"{item['steady_p95_step_ms']:.6f}",
                "acc": f"{final['cluster_accuracy_noise_singletons']:.6f}",
                "ari": f"{final['ari']:.6f}",
                "nmi": f"{final['nmi']:.6f}",
                "fmr": f"{final['false_merge_rate']:.6f}",
            })
    write_csv(args.tables / "canonical_main.csv", main_records)

    seed_records = []
    for record in csv.DictReader(
        (audit / "multiseed_validation_5seeds/multiseed.csv").open(encoding="utf-8")
    ):
        seed_records.append(record)
    write_csv(args.tables / "canonical_multiseed.csv", seed_records)

    rate_records = list(csv.DictReader(
        (audit / "rate_sweep/rate_sweep.csv").open(encoding="utf-8")
    ))
    write_csv(args.tables / "canonical_rate.csv", rate_records)

    tradeoff_records = list(csv.DictReader(
        (audit / "tradeoff_sweep/tradeoff_sweep.csv").open(encoding="utf-8")
    ))
    write_csv(args.tables / "canonical_tradeoff.csv", tradeoff_records)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 15.0,
        "xtick.labelsize": 14.0,
        "ytick.labelsize": 14.0,
        "legend.fontsize": 12.5,
        "axes.labelcolor": STYLE["text"],
        "axes.edgecolor": STYLE["text"],
        "axes.linewidth": 1.05,
        "xtick.color": STYLE["text"],
        "ytick.color": STYLE["text"],
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": STYLE["grid"],
        "grid.alpha": 0.90,
        "grid.linestyle": "--",
        "grid.linewidth": 0.85,
        "lines.linewidth": 3.0,
        "lines.markersize": 7.0,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    draw_scene_figure(args.figures / "fig_scene")

    colors = {"Fish": STYLE["fish"], "Ours": STYLE["ours"]}
    labels = [x[0] for x in workloads]
    x = np.arange(len(labels))
    display_methods = (("Fish", "FISHDBC"), ("Ours", "FlowFish"))

    def save_figure(fig, stem: str) -> None:
        fig.tight_layout(pad=0.6)
        fig.savefig(args.figures / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(args.figures / f"{stem}.png", bbox_inches="tight")
        plt.close(fig)

    # LoongX habit: one question per PDF, then compose related PDFs in LaTeX.
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    width = 0.34
    for offset, (method, display) in zip((-width / 2, width / 2), display_methods):
        values = [
            float(next(r for r in main_records if r["workload"] == w and r["method"] == method)["p95_ms"])
            for w in labels
        ]
        ax.bar(x + offset, values, width, label=display, color=colors[method], edgecolor="none")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylim(0.25, 30)
    ax.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    ax.set_ylabel("P95 latency (ms)")
    ax.set_xlabel("Workload")
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.4)
    style_axis(ax)
    save_figure(fig, "fig_latency")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    width = 0.34
    quality_workload = "D640"
    quality_metrics = (("acc", "ACC"), ("ari", "ARI"), ("nmi", "NMI"))
    quality_x = np.arange(len(quality_metrics))
    for offset, (method, display) in zip((-width / 2, width / 2), display_methods):
        values = [
            float(next(
                r for r in main_records
                if r["workload"] == quality_workload and r["method"] == method
            )[metric])
            for metric, _ in quality_metrics
        ]
        ax.bar(
            quality_x + offset, np.asarray(values) - 0.65, width, bottom=0.65,
            label=display, color=colors[method], edgecolor="none",
        )
    ax.set_xticks(quality_x, [label for _, label in quality_metrics])
    ax.set_ylim(0.65, 1.01)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Final quality")
    ax.set_xlabel("Metric")
    ax.legend(frameon=False, ncol=2, loc="lower left", handlelength=1.4)
    style_axis(ax)
    save_figure(fig, "fig_quality")

    multiseed_records = list(csv.DictReader(
        (audit / "multiseed_validation_5seeds/multiseed.csv").open(encoding="utf-8")
    ))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    method_style = {
        "fishdbc": (STYLE["fish"], "FISHDBC"),
        "adaptive_fishdbc_async": (STYLE["ours"], "FlowFish"),
    }
    summary_x = np.arange(len(method_style))
    seed_summary = []
    for method, (color, label) in method_style.items():
        subset = [float(r["decision_p95_ms"]) for r in multiseed_records if r["method"] == method]
        seed_summary.append((label, color, float(np.mean(subset)), float(np.std(subset, ddof=1))))
    ax.bar(
        summary_x, [item[2] for item in seed_summary],
        yerr=[item[3] for item in seed_summary], capsize=4,
        color=[item[1] for item in seed_summary], width=0.56, edgecolor="none",
        error_kw={"elinewidth": 1.3, "capthick": 1.3},
    )
    ax.set_xticks(summary_x, [item[0] for item in seed_summary])
    ax.set_xlabel("Method")
    ax.set_ylabel("P95 latency (ms)")
    ax.set_ylim(0, 11)
    style_axis(ax)
    save_figure(fig, "fig_repeatability")

    rate_ours = sorted(
        [r for r in rate_records if r["method"] == "adaptive_fishdbc_async"],
        key=lambda r: float(r["interval_ms"]),
    )
    rate_positions = np.arange(len(rate_ours))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_xlabel("Arrival interval (ms)")
    ax.set_ylabel("P95 latency (ms)")
    ax.set_xticks(rate_positions, [str(int(float(r["interval_ms"]))) for r in rate_ours])
    ax.set_ylim(0, 12)
    ax.plot(
        rate_positions, [float(r["decision_p95_ms"]) for r in rate_ours],
        marker="o", markersize=7.0, linewidth=3.0, color=STYLE["ours"],
    )
    style_axis(ax)
    save_figure(fig, "fig_arrival_rate")

    report = {
        "audit": str(audit),
        "main_csv": str((args.tables / "canonical_main.csv").resolve()),
        "multiseed_csv": str((args.tables / "canonical_multiseed.csv").resolve()),
        "rate_csv": str((args.tables / "canonical_rate.csv").resolve()),
        "tradeoff_csv": str((args.tables / "canonical_tradeoff.csv").resolve()),
        "figures": [
            str((args.figures / name).resolve())
            for name in (
                "fig_scene.pdf",
                "fig_latency.pdf",
                "fig_quality.pdf",
                "fig_repeatability.pdf",
                "fig_arrival_rate.pdf",
            )
        ],
        "figure_style": {
            "reference": "LoongX_SIGCOMM2026 visual language",
            "font_family": "sans-serif",
            "plot_types": ["scene", "bar", "line"],
            "method_display": {"baseline": "FISHDBC", "ours": "FlowFish"},
            "latex_placement": "figure[t] with compact scene abstraction on page one",
            "figure_size_in": [FIG_WIDTH_IN, FIG_HEIGHT_IN],
            "scene_size_in": [SCENE_WIDTH_IN, SCENE_HEIGHT_IN],
            "colors": STYLE,
            "export_dpi": 300,
            "active_figures": [
                "fig_scene.pdf",
                "fig_latency.pdf",
                "fig_quality.pdf",
                "fig_repeatability.pdf",
                "fig_arrival_rate.pdf",
            ],
        },
        "status": "ok",
    }
    (args.tables / "canonical_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
