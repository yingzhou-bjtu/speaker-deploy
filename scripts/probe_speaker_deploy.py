#!/usr/bin/env python3
"""探测声纹部署的挂载耗时、推理耗时和本进程内存，只测当前 Python 进程。"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")


def _resolve_wavs(
    audio_dir: Optional[Path],
    wav_args: List[str],
    pattern: str,
    n: int,
) -> List[Path]:
    out: List[Path] = []
    for a in wav_args:
        p = Path(a).resolve()
        if p.is_file():
            out.append(p)
    if audio_dir is not None:
        d = audio_dir.resolve()
        if not d.is_dir():
            raise SystemExit(f"不是目录: {d}")
        out.extend(sorted(d.rglob(pattern)))
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq[: max(0, int(n))]


def run_probe(
    *,
    audio_dir: Optional[Path],
    wav_args: List[str],
    pattern: str,
    n_files: int,
    onnx_path: Optional[Path],
    sv_offline_root: Optional[Path],
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    ort_threads: int,
    out_json: Optional[Path],
    skip_stream: bool,
) -> int:
    from speaker_id.engine import default_sv_onnx_path
    from speaker_id.probe_metrics import ProbeSession
    from speaker_id.streaming_pipeline import StreamingClusterPipeline

    files = _resolve_wavs(audio_dir, wav_args, pattern, n_files)
    if not files and not skip_stream:
        print("需要 wav 或 --audio-dir", file=sys.stderr)
        return 1

    probe = ProbeSession()
    probe.meta["n_files_requested"] = n_files
    probe.meta["n_files_used"] = len(files)
    probe.meta["audio_dir"] = str(audio_dir) if audio_dir else None

    onnx_p = default_sv_onnx_path(
        sv_offline_root.resolve() if sv_offline_root else None
    )
    if onnx_path is not None:
        onnx_p = onnx_path.resolve()
    if onnx_p.is_file():
        probe.meta["onnx_path"] = str(onnx_p)
        probe.meta["onnx_size_mb"] = round(onnx_p.stat().st_size / (1024 * 1024), 2)
    else:
        probe.meta["onnx_path"] = str(onnx_p)
        probe.meta["onnx_missing"] = True

    # 依赖导入。
    with probe.phase("import_deps") as bag:
        import numpy as np  # noqa: F401
        import onnxruntime as ort  # noqa: F401

        bag["numpy"] = np.__version__
        bag["onnxruntime"] = ort.__version__

    # 构造 pipeline 并触发 ONNX 加载。
    pipe: Optional[StreamingClusterPipeline] = None
    with probe.phase("mount_pipeline") as bag:
        pkw: Dict[str, Any] = {
            "device": "cpu",
            "sv_backend": "onnx",
            "min_cluster_size": int(hdbscan_min_cluster_size),
            "min_samples": int(hdbscan_min_samples),
            "ort_threads": int(ort_threads),
        }
        if onnx_path is not None:
            pkw["onnx_path"] = onnx_path.resolve()
        if sv_offline_root is not None:
            pkw["offline_root"] = sv_offline_root.resolve()
        pipe = StreamingClusterPipeline(**pkw)
        pipe._eng._ensure_onnx()
        bag["backend"] = pipe._eng.backend
        bag["onnx_path"] = str(pipe._eng.onnx_path)
        bag["buffer_cap"] = getattr(pipe, "max_hdbscan_utterances", 0)

    assert pipe is not None

    if files:
        first = files[0]

        with probe.phase("embed_only") as bag:
            emb = pipe._eng.embed_file(first)
            bag["emb_dim"] = int(emb.shape[-1])
            bag["wav"] = first.name

        with probe.phase("first_ingest") as bag:
            r = pipe.ingest_wav(first, wav_key=first.name)
            bag["v_id_for_use"] = r.v_id_for_use
            bag["confidence"] = r.confidence
            bag["backend"] = (r.cluster_meta or {}).get("clustering_backend")
            bag["n_buffer"] = pipe.n_arrived

        if len(files) > 1 and not skip_stream:
            per_file: List[Dict[str, Any]] = []
            with probe.phase("stream_ingest") as bag:
                import time

                for i, p in enumerate(files[1:], start=2):
                    t0 = time.perf_counter()
                    r = pipe.ingest_wav(p, wav_key=p.name)
                    dt = time.perf_counter() - t0
                    per_file.append(
                        {
                            "index": i,
                            "wav": p.name,
                            "wall_s": round(dt, 4),
                            "v_id_for_use": r.v_id_for_use,
                            "confidence": round(r.confidence, 4),
                            "n_buffer": pipe.n_arrived,
                            "cluster_backend": (r.cluster_meta or {}).get(
                                "clustering_backend"
                            ),
                        }
                    )
                walls = [x["wall_s"] for x in per_file]
                bag["n_ingested"] = len(per_file)
                bag["wall_total_s"] = round(sum(walls), 4)
                bag["wall_mean_s"] = round(sum(walls) / len(walls), 4) if walls else 0
                bag["wall_max_s"] = round(max(walls), 4) if walls else 0
                bag["per_file"] = per_file
                bag["n_buffer_final"] = pipe.n_arrived

            from speaker_id.cluster_panel import cluster_panel_counts

            assigned, uncertain, total = cluster_panel_counts(pipe)
            probe.meta["final_panel"] = {
                "assigned": assigned,
                "uncertain": uncertain,
                "total": total,
            }

    gc.collect()
    with probe.phase("finalize"):
        pass

    probe.print_human()
    if out_json is not None:
        probe.write_json(out_json.resolve())
        print(f"\nJSON 已写入: {out_json.resolve()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="声纹 deploy 探测：挂载/评估耗时与本进程资源",
    )
    ap.add_argument("wav", nargs="*", default=(), help="wav 文件（可与 --audio-dir 合用）")
    ap.add_argument("--audio-dir", type=Path, default=None)
    ap.add_argument("--pattern", default="*.wav")
    ap.add_argument("-n", "--num-files", type=int, default=20, help="最多探测条数")
    ap.add_argument("--onnx-path", type=Path, default=None)
    ap.add_argument("--sv-offline-root", type=Path, default=None)
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=3)
    ap.add_argument("--hdbscan-min-samples", type=int, default=2)
    ap.add_argument("--ort-threads", type=int, default=4)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="写出 JSON 报告路径（默认 deploy/runs/probe_<timestamp>.json）",
    )
    ap.add_argument(
        "--skip-stream",
        action="store_true",
        help="只做挂载+首条 embed/ingest，不跑流式多条",
    )
    args = ap.parse_args()

    out = args.out
    if out is None:
        from datetime import datetime

        runs = _ROOT / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = runs / f"probe_{ts}.json"

    return run_probe(
        audio_dir=args.audio_dir,
        wav_args=list(args.wav),
        pattern=str(args.pattern),
        n_files=int(args.num_files),
        onnx_path=args.onnx_path,
        sv_offline_root=args.sv_offline_root,
        hdbscan_min_cluster_size=int(args.hdbscan_min_cluster_size),
        hdbscan_min_samples=int(args.hdbscan_min_samples),
        ort_threads=int(args.ort_threads),
        out_json=out,
        skip_stream=bool(args.skip_stream),
    )


if __name__ == "__main__":
    raise SystemExit(main())
