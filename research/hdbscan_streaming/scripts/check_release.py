#!/usr/bin/env python3
"""Check that the public research package is self-contained enough to inspect."""
from __future__ import annotations

import compileall
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def main() -> int:
    required = (
        PROJECT_ROOT / "LICENSE",
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "scripts/run_benchmark.py",
        ROOT / "scripts/setup_research_env.sh",
        ROOT / "vendor/flexible_clustering/LICENSE",
        ROOT / "vendor/fast_hdbscan/LICENSE",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "RESULTS.md",
        ROOT / "CITATION.cff",
        ROOT / "scripts/reproduce_paper.sh",
        ROOT / "scripts/download_fsdd.sh",
        ROOT / "scripts/prepare_fsdd_audio.py",
        ROOT / "scripts/audio_to_embeddings.py",
        ROOT / "scripts/run_audio_smoke.sh",
        ROOT / "scripts/run_full_baseline.sh",
        ROOT / "figures/fig_main_results.png",
        ROOT / "figures/fig_stability_rate.png",
        ROOT / "figures/flowfish_framework.png",
        ROOT / "figures/flowfish_framework.pdf",
        ROOT / "results/smoke_release/summary.json",
        ROOT / "results/async_validation_release/async_comparison.json",
        ROOT / "results/rate_sweep_release/rate_sweep.json",
        ROOT / "results/multiseed_release/multiseed_summary.json",
        ROOT / "results/final_rerun_release/L/summary.json",
        ROOT / "results/final_rerun_release/M/summary.json",
        ROOT / "results/final_rerun_release/H/summary.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("missing release files:", *missing, sep="\n", file=sys.stderr)
        return 1
    non_executable = [
        str(path)
        for path in required
        if path.suffix == ".sh" and not path.stat().st_mode & 0o111
    ]
    if non_executable:
        print(
            "release shell scripts are not executable:",
            *non_executable,
            sep="\n",
            file=sys.stderr,
        )
        return 1

    forbidden_tokens = (
        "/home/",
        "../deploy/",
        "paper/references/",
        "hdbscan_streaming_bench",
    )
    scan_paths = [ROOT]
    violations: list[str] = []
    for base in scan_paths:
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if ".venv" in path.parts:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.is_file() and path.suffix in {
                ".cff",
                ".csv",
                ".lean",
                ".json",
                ".md",
                ".py",
                ".rst",
                ".sh",
                ".toml",
                ".txt",
            }:
                text = path.read_text(encoding="utf-8", errors="replace")
                for token in forbidden_tokens:
                    if token in text:
                        violations.append(f"{path}: {token}")
    if violations:
        print("non-portable release references:", *violations, sep="\n", file=sys.stderr)
        return 1

    data = np.load(ROOT / "data/synthetic_smoke.npz")
    if data["X"].ndim != 2 or len(data["X"]) != len(data["y"]):
        print("synthetic fixture violates X/y shape contract", file=sys.stderr)
        return 1
    for archive in (ROOT / "data").rglob("*.npz"):
        with np.load(archive, allow_pickle=False) as payload:
            if "wav_paths" not in payload:
                continue
            paths = [str(value) for value in payload["wav_paths"].tolist()]
            if any("/home/" in value or "hdbscan_streaming_bench" in value for value in paths):
                print(f"non-portable wav_paths in {archive}", file=sys.stderr)
                return 1
    for label in ("L", "M", "H"):
        summary_path = ROOT / "results/final_rerun_release" / label / "summary.json"
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(summaries) != 7 or any(item.get("status") != "ok" for item in summaries):
            print(f"paper-scale rerun is incomplete for {label}", file=sys.stderr)
            return 1
    if not compileall.compile_dir(str(ROOT / "scripts"), quiet=1):
        print("script compilation failed", file=sys.stderr)
        return 1

    print(
        "RELEASE_CHECK_PASS",
        f"fixture_shape={tuple(data['X'].shape)}",
        f"required_files={len(required)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
