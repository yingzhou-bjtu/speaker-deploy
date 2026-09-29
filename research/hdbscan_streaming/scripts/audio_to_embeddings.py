#!/usr/bin/env python3
"""Convert a manifest of WAV files into the benchmark NPZ contract.

The ONNX model is intentionally supplied by the caller. The public package
only defines the audio and tensor contract: 16 kHz mono waveform, 80-bin
Kaldi FBank, and one embedding tensor per utterance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_wav_16k_mono(path: Path) -> np.ndarray:
    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.asarray(wav, dtype=np.float32).mean(axis=1)
    if int(sample_rate) != 16000:
        from scipy import signal

        mono = signal.resample_poly(mono, 16000, int(sample_rate))
    return np.asarray(mono, dtype=np.float32).reshape(-1)


def wav_to_fbank(wav: np.ndarray, n_mels: int = 80) -> np.ndarray:
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000.0
    opts.frame_opts.dither = 0.0
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.round_to_power_of_two = True
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = int(n_mels)
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = 0.0
    opts.energy_floor = 1.0

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, np.asarray(wav, dtype=np.float32).tolist())
    fbank.input_finished()
    if fbank.num_frames_ready <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    features = np.asarray(
        [fbank.get_frame(index) for index in range(fbank.num_frames_ready)],
        dtype=np.float32,
    )
    return features - features.mean(axis=0, keepdims=True)


def build_session(model: Path, ort_threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if ort_threads > 0:
        options.intra_op_num_threads = ort_threads
        options.inter_op_num_threads = ort_threads
    return ort.InferenceSession(
        str(model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def make_input(features: np.ndarray, shape: list[int | str | None], layout: str) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 80:
        raise ValueError(f"expected [frames, 80] FBank, got {features.shape}")
    if layout == "auto":
        last = shape[-1] if shape else None
        second_last = shape[-2] if len(shape) >= 2 else None
        if last in (80, "80"):
            layout = "time_mel"
        elif second_last in (80, "80"):
            layout = "mel_time"
        else:
            layout = "time_mel"
    if layout == "time_mel":
        return features[None, :, :].astype(np.float32, copy=False)
    if layout == "mel_time":
        return features.T[None, :, :].astype(np.float32, copy=False)
    raise ValueError(f"unsupported --feature-layout: {layout}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--ort-threads", type=int, default=1)
    parser.add_argument(
        "--feature-layout",
        choices=("auto", "time_mel", "mel_time"),
        default="auto",
    )
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    raw_root = args.raw_root.resolve()
    model = args.model.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not model.is_file():
        raise FileNotFoundError(model)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty manifest: {manifest}")

    manifest_sha256 = sha256_file(manifest)
    model_sha256 = sha256_file(model)
    if args.out.is_file() and args.meta.is_file() and not args.force:
        try:
            cached_meta = json.loads(args.meta.read_text(encoding="utf-8"))
            cached = np.load(args.out, allow_pickle=False)
            if (
                cached_meta.get("manifest_sha256") == manifest_sha256
                and cached_meta.get("model_sha256") == model_sha256
                and int(cached_meta.get("ort_threads", -1)) == args.ort_threads
                and cached["X"].shape[0] == len(rows)
            ):
                print(json.dumps({"status": "cache_hit", "output": str(args.out)}))
                return
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    session = build_session(model, args.ort_threads)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    speakers: list[str] = []
    digits: list[int] = []
    paths: list[str] = []
    elapsed_ms: list[float] = []
    for index, row in enumerate(rows):
        wav_path = raw_root / row["path"]
        features = wav_to_fbank(load_wav_16k_mono(wav_path))
        if features.shape[0] == 0:
            raise ValueError(f"audio is too short at row {index}: {wav_path}")
        model_input = make_input(features, list(input_meta.shape), args.feature_layout)
        start = time.perf_counter()
        outputs = session.run(None, {input_name: model_input})
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        if not outputs:
            raise ValueError(f"model returned no output at row {index}")
        vector = np.asarray(outputs[args.output_index], dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"invalid embedding at row {index}: {wav_path}")
        vectors.append(vector)
        labels.append(int(row["speaker_id"]))
        speakers.append(row["speaker"])
        digits.append(int(row["digit"]))
        paths.append(row["path"])

    dimensions = {int(vector.size) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent embedding dimensions: {dimensions}")
    X = np.stack(vectors).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_name(args.out.name + ".tmp.npz")
    tmp_meta = args.meta.with_name(args.meta.name + ".tmp")
    np.savez_compressed(
        tmp_out,
        X=X,
        y=y,
        speaker_names=np.asarray(speakers),
        digits=np.asarray(digits, dtype=np.int64),
        wav_paths=np.asarray(paths),
        embedding_ms=np.asarray(elapsed_ms, dtype=np.float64),
    )
    metadata = {
        "manifest": manifest.name,
        "raw_root": raw_root.name,
        "model": model.name,
        "model_sha256": model_sha256,
        "manifest_sha256": manifest_sha256,
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "speakers": sorted(set(speakers)),
        "sample_rate_hz": 16000,
        "feature_layout": args.feature_layout,
        "ort_threads": args.ort_threads,
        "embedding_ms_mean": float(np.mean(elapsed_ms)),
        "embedding_ms_p95": float(np.percentile(elapsed_ms, 95)),
        "embedding_ms_min": float(np.min(elapsed_ms)),
        "embedding_ms_max": float(np.max(elapsed_ms)),
        "output": args.out.name,
    }
    tmp_meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_out, args.out)
    os.replace(tmp_meta, args.meta)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
