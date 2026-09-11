#!/usr/bin/env python3
"""Create a deterministic, balanced FSDD audio manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_name(path: Path) -> tuple[int, str, int]:
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"unexpected FSDD filename: {path.name}")
    return int(parts[0]), parts[1], int(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-speaker", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    files = sorted(raw_root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"no WAV files under {raw_root}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    parsed: dict[Path, tuple[int, str, int]] = {}
    for path in files:
        item = parse_name(path)
        parsed[path] = item
        grouped[item[1]].append(path)
    speakers = sorted(grouped)
    if args.per_speaker <= 0:
        raise ValueError("--per-speaker must be positive")
    if any(len(grouped[name]) < args.per_speaker for name in speakers):
        raise ValueError("not enough recordings for requested per-speaker count")

    rng = np.random.default_rng(args.seed)
    selected_by_speaker: dict[str, list[Path]] = {}
    for speaker in speakers:
        candidates = np.asarray(sorted(grouped[speaker]), dtype=object)
        order = rng.permutation(len(candidates))
        selected_by_speaker[speaker] = [Path(candidates[i]) for i in order[: args.per_speaker]]

    selected: list[Path] = []
    for row in range(args.per_speaker):
        for speaker in speakers:
            selected.append(selected_by_speaker[speaker][row])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    speaker_ids = {speaker: i for i, speaker in enumerate(speakers)}
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "path",
            "speaker",
            "speaker_id",
            "digit",
            "utterance_id",
            "sample_rate_hz",
            "num_samples",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in selected:
            digit, speaker, utterance_id = parsed[path]
            info = sf.info(str(path))
            writer.writerow(
                {
                    "path": str(path.relative_to(raw_root)),
                    "speaker": speaker,
                    "speaker_id": speaker_ids[speaker],
                    "digit": digit,
                    "utterance_id": utterance_id,
                    "sample_rate_hz": int(info.samplerate),
                    "num_samples": int(info.frames),
                }
            )

    metadata = {
        "dataset": "Free Spoken Digit Dataset",
        "source_repository": "https://github.com/Jakobovski/free-spoken-digit-dataset",
        "raw_root": str(raw_root),
        "raw_wav_count": len(files),
        "selected_wav_count": len(selected),
        "seed": args.seed,
        "per_speaker": args.per_speaker,
        "speakers": speakers,
        "speaker_to_id": speaker_ids,
        "manifest": str(manifest_path.resolve()),
        "selection_order": "round-robin by speaker after seeded per-speaker sampling",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
