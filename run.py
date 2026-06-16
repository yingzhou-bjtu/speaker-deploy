#!/usr/bin/env python3
"""板端常驻声纹，默认 serve，每次 ingest 后刷新聚类面板。"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Union

_ROOT = Path(__file__).resolve().parent


def _default_max_hdbscan_utterances() -> int:
    from speaker_id.pipeline_factory import default_max_hdbscan_utterances

    return default_max_hdbscan_utterances()


def _default_ort_threads() -> int:
    from speaker_id.pipeline_factory import default_ort_threads

    return default_ort_threads()


def _default_cluster_selection_epsilon() -> float:
    from speaker_id.pipeline_factory import default_cluster_selection_epsilon

    return default_cluster_selection_epsilon()


def _default_max_ingest_audio_sec() -> float:
    from speaker_id.pipeline_factory import default_max_ingest_audio_sec

    return default_max_ingest_audio_sec()

def _ensure_path() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _quiet_runtime() -> None:
    os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")


def _resolve_wav_paths(
    *,
    audio_dir: Path | None,
    wav_args: Sequence[str],
    pattern: str,
) -> List[Path]:
    out: List[Path] = []
    for a in wav_args:
        p = Path(a).resolve()
        if not p.is_file():
            raise SystemExit(f"不是文件: {p}")
        out.append(p)
    if audio_dir is not None:
        d = audio_dir.resolve()
        if not d.is_dir():
            raise SystemExit(f"不是目录: {d}")
        for p in sorted(d.rglob(pattern)):
            if p.is_file():
                out.append(p)
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def _make_pipeline(args: argparse.Namespace):
    from speaker_id.pipeline_factory import build_pipeline_kwargs, default_max_ingest_audio_sec
    from speaker_id.streaming_pipeline import StreamingClusterPipeline

    cap = int(getattr(args, "max_hdbscan_utterances", 0))
    max_sec = float(getattr(args, "max_ingest_audio_sec", 0) or 0)
    kw = build_pipeline_kwargs(
        onnx_path=getattr(args, "onnx_path", None),
        offline_root=getattr(args, "sv_offline_root", None),
        ort_threads=int(getattr(args, "ort_threads", _default_ort_threads())),
        max_hdbscan_utterances=cap if cap > 0 else None,
        min_cluster_size=int(getattr(args, "hdbscan_min_cluster_size", 3)),
        min_samples=int(getattr(args, "hdbscan_min_samples", 2)),
        cluster_selection_epsilon=float(
            getattr(args, "hdbscan_cluster_selection_epsilon", 0.05)
        ),
        max_ingest_audio_sec=max_sec if max_sec > 0 else None,
    )
    return StreamingClusterPipeline(**kw)


def _display_name(path: Path, wav_key: Optional[str] = None) -> str:
    if wav_key:
        return str(wav_key)
    return path.name


def _ingest_refresh_panel(
    pipe,
    path: Union[str, Path],
    wav_key: Optional[str] = None,
    *,
    clear: bool = True,
    verbose: bool = False,
) -> None:
    from speaker_id.cluster_panel import refresh_cluster_panel

    p = Path(path).resolve()
    key = _display_name(p, wav_key)
    try:
        r = pipe.ingest_wav(p, wav_key=key)
        if verbose:
            vid = r.v_id_for_use if r.v_id_for_use is not None else "?"
            print(f"{key} -> {vid}", flush=True)
        refresh_cluster_panel(pipe, clear=clear, last_key=key)
    except FileNotFoundError:
        print(f"{key} -> ERROR 文件不存在: {p}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"{key} -> ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def _stdin_loop(pipe, stop_on_empty: bool = False) -> None:
    clear = not bool(getattr(pipe, "_panel_no_clear", False))
    verbose = bool(getattr(pipe, "_panel_verbose", False))
    for line in sys.stdin:
        if not line:
            if stop_on_empty:
                break
            continue
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower() in ("quit", "exit", "shutdown", "q"):
            break
        _ingest_refresh_panel(pipe, s, clear=clear, verbose=verbose)


def _add_pipeline_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--onnx-path", type=Path, default=None)
    ap.add_argument("--sv-offline-root", type=Path, default=None)
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=3)
    ap.add_argument("--hdbscan-min-samples", type=int, default=2)
    ap.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=_default_cluster_selection_epsilon(),
    )
    ap.add_argument(
        "--max-ingest-audio-sec",
        type=float,
        default=_default_max_ingest_audio_sec(),
        help="单条 embed 最长秒数，0=不截断",
    )
    ap.add_argument(
        "--max-hdbscan-utterances",
        type=int,
        default=_default_max_hdbscan_utterances(),
        help="流式缓冲上限，0=全前缀 HDBSCAN",
    )
    ap.add_argument(
        "--ort-threads",
        type=int,
        default=_default_ort_threads(),
        help="ONNX Runtime CPU 线程数，RK3588 实测 4 较快（0=ORT 默认）",
    )


def _serve_use_stdin(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_stdin", False)):
        return False
    return bool(getattr(args, "stdin", True))


def cmd_serve(args: argparse.Namespace) -> int:
    """常驻服务，可处理启动 wav、标准输入或 Redis，每次刷新面板。"""
    _ensure_path()
    _quiet_runtime()
    from speaker_id.streaming_ipc import redis_streams_loop

    use_stdin = _serve_use_stdin(args)
    pipe = _make_pipeline(args)
    pipe._panel_no_clear = bool(getattr(args, "no_clear", False))  # type: ignore[attr-defined]
    pipe._panel_verbose = bool(getattr(args, "verbose", False))  # type: ignore[attr-defined]
    clear = not pipe._panel_no_clear  # type: ignore[attr-defined]
    verbose = bool(pipe._panel_verbose)  # type: ignore[attr-defined]

    paths = _resolve_wav_paths(
        audio_dir=args.audio_dir,
        wav_args=tuple(args.wav),
        pattern=str(args.pattern),
    )
    for wav in paths:
        _ingest_refresh_panel(pipe, wav, clear=clear, verbose=verbose)

    redis_thread: Optional[threading.Thread] = None
    if args.redis:
        redis_thread = threading.Thread(
            target=redis_streams_loop,
            kwargs={
                "pipe": pipe,
                "redis_url": str(args.redis_url),
                "list_jobs": str(args.redis_jobs),
                "list_results": str(args.redis_results),
                "timeout_sec": float(args.redis_timeout),
                "refresh_panel": True,
                "panel_clear": clear,
                "push_results": False,
            },
            name="sv-redis-ingest",
            daemon=not use_stdin,
        )
        redis_thread.start()

    if use_stdin:
        _stdin_loop(pipe, stop_on_empty=not sys.stdin.isatty())
    elif redis_thread is not None:
        redis_thread.join()
    return 0


def cmd_audio(args: argparse.Namespace) -> int:
    """一次性处理给定 wav 后退出。"""
    _ensure_path()
    _quiet_runtime()
    paths = _resolve_wav_paths(
        audio_dir=args.audio_dir,
        wav_args=tuple(args.wav),
        pattern=str(args.pattern),
    )
    if not paths:
        print("需要 wav 路径或 --audio-dir", file=sys.stderr)
        return 1
    pipe = _make_pipeline(args)
    clear = not bool(getattr(args, "no_clear", False))
    verbose = bool(getattr(args, "verbose", False))
    for wav in paths:
        _ingest_refresh_panel(pipe, wav, clear=clear, verbose=verbose)
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    """向 Redis 队列投递一条 wav。"""
    _ensure_path()
    from speaker_id.streaming_ipc import ingest_message, redis_push_job

    wav = Path(args.wav).resolve()
    if not wav.is_file():
        print(f"不是文件: {wav}", file=sys.stderr)
        return 1
    redis_push_job(
        str(args.redis_url),
        str(args.redis_jobs),
        ingest_message(wav, wav_key=args.wav_key or wav.name),
    )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """探测模型挂载和推理耗时及内存占用。"""
    _ensure_path()
    _quiet_runtime()
    from scripts.probe_speaker_deploy import run_probe

    return run_probe(
        audio_dir=getattr(args, "audio_dir", None),
        wav_args=list(getattr(args, "wav", ()) or ()),
        pattern=str(getattr(args, "pattern", "*.wav")),
        n_files=int(getattr(args, "num_files", 20)),
        onnx_path=getattr(args, "onnx_path", None),
        sv_offline_root=getattr(args, "sv_offline_root", None),
        hdbscan_min_cluster_size=int(getattr(args, "hdbscan_min_cluster_size", 3)),
        hdbscan_min_samples=int(getattr(args, "hdbscan_min_samples", 2)),
        ort_threads=int(getattr(args, "ort_threads", _default_ort_threads())),
        out_json=getattr(args, "out", None),
        skip_stream=bool(getattr(args, "skip_stream", False)),
    )


def cmd_serve_robot(args: argparse.Namespace) -> int:
    """HTTP API 加 robotmedia 语音的一体服务。"""
    _ensure_path()
    _quiet_runtime()
    from robot_service import run_service

    return int(run_service(args))


def cmd_pcm_debug(args: argparse.Namespace) -> int:
    """只测 TCP PCM 收流，不加载声纹模型。"""
    _ensure_path()
    from speaker_id.tcp_audio import TCPAudioClient

    client = TCPAudioClient(
        str(args.pcm_host),
        int(args.pcm_port),
        timeout_sec=float(args.pcm_timeout),
    )
    n = int(args.frames)
    try:
        client.connect()
        total = 0
        i = 0
        while n == 0 or i < n:
            pcm, _ = client.recv_pcm_with_time(int(args.pcm_frame_bytes))
            total += len(pcm)
            i += 1
            print(f"frame {i}: {len(pcm)} bytes", flush=True)
        print(f"OK: {i} frames, {total} bytes", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        client.close()


def cmd_api(args: argparse.Namespace) -> int:
    """启动 HTTP API 服务。"""
    _ensure_path()
    _quiet_runtime()
    from api_server import run_api_server

    from speaker_id.pipeline_worker import PipelineWorker
    from speaker_id.pipeline_factory import default_pipeline_queue_maxsize

    pipe = _make_pipeline(args)
    run_api_server(
        PipelineWorker(pipe, queue_maxsize=default_pipeline_queue_maxsize()),
        host=str(getattr(args, "api_host", "0.0.0.0")),
        port=int(getattr(args, "api_port", 8765)),
    )
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    _ensure_path()
    _quiet_runtime()
    from pathlib import Path

    from speaker_id.pipeline_factory import build_pipeline_kwargs, default_speaker_registry_path
    from speaker_id.streaming_pipeline import StreamingClusterPipeline

    wav = Path(args.wav).resolve()
    if not wav.is_file():
        print(f"不是文件: {wav}", file=sys.stderr)
        return 1

    npz_path = (
        Path(args.npz).resolve()
        if args.npz
        else default_speaker_registry_path()
    )
    kw = build_pipeline_kwargs(
        onnx_path=getattr(args, "onnx_path", None),
        offline_root=getattr(args, "sv_offline_root", None),
        speaker_registry_path=npz_path,
        persist_registry=True,
    )
    pipe = StreamingClusterPipeline(**kw)
    emb = pipe.engine.embed_file(wav)
    sid = pipe.register_exemplar(emb, str(args.name).strip())
    print(
        f"{wav.name} -> registered:{args.name} stable_v_id={sid} npz={npz_path}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="声纹 deploy：serve 实时刷新 v_id / 不确定 聚类面板",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser(
        "serve",
        help="常驻 worker（stdin/Redis/启动参数）；每次 ingest 刷新聚类面板",
    )
    ps.add_argument(
        "--verbose",
        action="store_true",
        help="除面板外再打印每条 名 -> v_id",
    )
    ps.add_argument(
        "--no-clear",
        action="store_true",
        help="不清屏，用分隔线输出各次面板快照（适合重定向日志）",
    )
    ps.add_argument("wav", nargs="*", default=(), help="启动时先处理的 wav")
    ps.add_argument("--audio-dir", type=Path, default=None)
    ps.add_argument("--pattern", default="*.wav")
    ps.add_argument(
        "--redis",
        action="store_true",
        help="同时监听 Redis 队列",
    )
    ps.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    ps.add_argument("--redis-jobs", default="speak:stream:jobs")
    ps.add_argument("--redis-results", default="speak:stream:results")
    ps.add_argument("--redis-timeout", type=float, default=5.0)
    ps.add_argument(
        "--no-stdin",
        action="store_true",
        help="不读标准输入（仅 Redis 或启动参数）",
    )
    _add_pipeline_args(ps)
    ps.set_defaults(_fn=cmd_serve, stdin=True)

    pa = sub.add_parser("audio", help="一次性处理 wav 后退出")
    pa.add_argument("wav", nargs="*", default=())
    pa.add_argument("--audio-dir", type=Path, default=None)
    pa.add_argument("--pattern", default="*.wav")
    pa.add_argument("--verbose", action="store_true")
    pa.add_argument("--no-clear", action="store_true")
    _add_pipeline_args(pa)
    pa.set_defaults(_fn=cmd_audio)

    pp = sub.add_parser("push", help="向 serve --redis 投递一条 wav")
    pp.add_argument("wav", type=Path)
    pp.add_argument("--wav-key", default=None)
    pp.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    pp.add_argument("--redis-jobs", default="speak:stream:jobs")
    pp.set_defaults(_fn=cmd_push)

    pr = sub.add_parser("register", help="注册声纹到 npz")
    pr.add_argument("name")
    pr.add_argument("wav", type=Path)
    pr.add_argument("--npz", type=Path, default=None)
    _add_pipeline_args(pr)
    pr.set_defaults(_fn=cmd_register)

    pprobe = sub.add_parser(
        "probe",
        help="探测挂载/流式评估耗时与本进程 RSS、CPU（写出 JSON）",
    )
    pprobe.add_argument("wav", nargs="*", default=())
    pprobe.add_argument("--audio-dir", type=Path, default=None)
    pprobe.add_argument("--pattern", default="*.wav")
    pprobe.add_argument("-n", "--num-files", type=int, default=20)
    pprobe.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON 报告路径（默认 deploy/runs/probe_<时间>.json）",
    )
    pprobe.add_argument(
        "--skip-stream",
        action="store_true",
        help="仅挂载+首条 embed/ingest",
    )
    _add_pipeline_args(pprobe)
    pprobe.set_defaults(_fn=cmd_probe)

    papi = sub.add_parser("api", help="HTTP API：POST /v1/ingest，GET /v1/panel")
    papi.add_argument("--api-host", default="0.0.0.0")
    papi.add_argument("--api-port", type=int, default=8765)
    _add_pipeline_args(papi)
    papi.set_defaults(_fn=cmd_api)

    probot = sub.add_parser(
        "serve-robot",
        help="推荐板端：HTTP API + robotmedia PCM → 声纹聚类",
    )
    probot.add_argument("--api-host", default="0.0.0.0")
    probot.add_argument("--api-port", type=int, default=8765)
    probot.add_argument("--no-api", action="store_true")
    probot.add_argument("--no-pcm", action="store_true")
    probot.add_argument("--pcm-host", default="127.0.0.1")
    probot.add_argument("--pcm-port", type=int, default=5001)
    probot.add_argument(
        "--pcm-mode",
        choices=("vad", "fixed"),
        default="vad",
        help="vad=按 robotmedia VAD 段(TCP会话)；fixed=固定秒切片",
    )
    probot.add_argument("--pcm-frame-bytes", type=int, default=3200)
    probot.add_argument("--pcm-segment-sec", type=float, default=3.0)
    probot.add_argument("--pcm-timeout", type=float, default=0.0)
    probot.add_argument("--pcm-reconnect-sec", type=float, default=5.0)
    probot.add_argument("--verbose", action="store_true")
    probot.add_argument("--no-panel", action="store_true")
    probot.add_argument("--skip-preflight", action="store_true")
    probot.add_argument(
        "--robotmedia-config",
        default="/usr/share/robotmedia/config/config.ini",
    )
    _add_pipeline_args(probot)
    probot.set_defaults(_fn=cmd_serve_robot)

    ppcm = sub.add_parser("pcm-debug", help="仅测 TCP PCM 收流")
    ppcm.add_argument("--pcm-host", default="127.0.0.1")
    ppcm.add_argument("--pcm-port", type=int, default=5001)
    ppcm.add_argument("--pcm-frame-bytes", type=int, default=3200)
    ppcm.add_argument("-n", "--frames", type=int, default=5)
    ppcm.add_argument("--pcm-timeout", type=float, default=10.0)
    ppcm.set_defaults(_fn=cmd_pcm_debug)

    # 兼容旧 worker 子命令。
    pw = sub.add_parser("worker", help="同 serve --redis --no-stdin")
    pw.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    pw.add_argument("--redis-jobs", default="speak:stream:jobs")
    pw.add_argument("--redis-results", default="speak:stream:results")
    pw.add_argument("--redis-timeout", type=float, default=5.0)
    _add_pipeline_args(pw)
    pw.set_defaults(
        _fn=cmd_serve,
        wav=(),
        audio_dir=None,
        pattern="*.wav",
        redis=True,
        stdin=False,
    )

    args = ap.parse_args()
    return int(args._fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
