#!/usr/bin/env python3
"""机器人一体服务，HTTP API 加 robotmedia 语音流式声纹。"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent

# 和 robotmedia 单段最长 20 秒对齐。
MAX_VAD_SEGMENT_SEC = 20.0

# both 模式下落盘 wav 和 TCP 常是同一句，去重避免算两遍。
_wav_dedup_lock = threading.Lock()
_last_wav_ingest_mono: float = 0.0
_WAV_DEDUP_SEC = 5.0


class TcpSegmentBuffer:
    """5001 口持续收 PCM，vad 结束时快照，和落盘 wav 对齐比对。"""

    def __init__(self, *, max_bytes: int = 0, stale_sec: float = 45.0) -> None:
        self._lock = threading.Lock()
        self._buf = b""
        self._max_bytes = max(0, int(max_bytes))
        self._stale_sec = max(0.0, float(stale_sec))
        self._last_mono = 0.0
        self._trunc_warn_mono = 0.0
        self._seg_start_ts: Optional[float] = None

    def append(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if not self._buf:
                self._seg_start_ts = time.time()
            self._buf += data
            self._last_mono = time.monotonic()
            if self._max_bytes > 0 and len(self._buf) > self._max_bytes:
                drop = len(self._buf) - self._max_bytes
                self._buf = self._buf[-self._max_bytes :]
                now = time.monotonic()
                if now - self._trunc_warn_mono >= 60.0:
                    self._trunc_warn_mono = now
                    print(
                        f"[pcm] TcpSegmentBuffer trim {drop}B (cap {self._max_bytes}B)",
                        flush=True,
                    )

    def snapshot_clear(self) -> bytes:
        snap, _, _ = self.snapshot_clear_span()
        return snap

    def snapshot_clear_span(self) -> tuple[bytes, Optional[float], float]:
        """返回 pcm 数据和起止时间。"""
        with self._lock:
            snap = self._buf
            start = self._seg_start_ts
            end = time.time()
            self._buf = b""
            self._seg_start_ts = None
            return snap, start, end

    def prune_stale(self) -> int:
        """太久没新数据就清空缓冲，返回丢弃字节数。"""
        with self._lock:
            if not self._buf or self._stale_sec <= 0:
                return 0
            if time.monotonic() - self._last_mono < self._stale_sec:
                return 0
            n = len(self._buf)
            self._buf = b""
            self._seg_start_ts = None
            return n


def _note_wav_ingest() -> None:
    global _last_wav_ingest_mono
    with _wav_dedup_lock:
        _last_wav_ingest_mono = time.monotonic()


def _recent_wav_ingest() -> bool:
    with _wav_dedup_lock:
        return (time.monotonic() - _last_wav_ingest_mono) < _WAV_DEDUP_SEC


def _ensure_path() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _make_pipeline(args: argparse.Namespace):
    from speaker_id.pipeline_factory import build_pipeline_kwargs
    from speaker_id.streaming_pipeline import StreamingClusterPipeline

    cap = int(args.max_hdbscan_utterances)
    kw = build_pipeline_kwargs(
        onnx_path=args.onnx_path,
        ort_threads=int(args.ort_threads),
        max_hdbscan_utterances=cap if cap > 0 else None,
        min_cluster_size=int(args.hdbscan_min_cluster_size),
        min_samples=int(args.hdbscan_min_samples),
        cluster_selection_epsilon=float(args.hdbscan_cluster_selection_epsilon),
        max_ingest_audio_sec=float(getattr(args, "max_ingest_audio_sec", 0)) or None,
    )
    return StreamingClusterPipeline(**kw)


def _run_preflight(args: argparse.Namespace) -> None:
    from scripts.preflight_board import run_checks

    onnx = (
        args.onnx_path.resolve()
        if args.onnx_path
        else (_ROOT / "pretrained" / "eres2net_sv.onnx")
    )
    errs = run_checks(
        config_ini=Path(args.robotmedia_config),
        pcm_host=str(args.pcm_host),
        pcm_port=int(args.pcm_port),
        onnx_path=onnx,
        venv_python=_ROOT / ".venv" / "bin" / "python",
        require_pcm=not args.no_pcm,
    )
    if errs:
        raise SystemExit(1)


def _min_segment_bytes(min_speech_ms: float = 300.0) -> int:
    from speaker_id.pcm_io import BYTES_PER_SAMPLE, SAMPLE_RATE_HZ

    return int(SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * float(min_speech_ms) / 1000.0)


def _max_vad_segment_bytes() -> int:
    from speaker_id.pcm_io import segment_byte_length

    return segment_byte_length(MAX_VAD_SEGMENT_SEC)


def _wav_pcm_frame_count(wav_path: Path) -> Optional[int]:
    try:
        import soundfile as sf

        return int(sf.info(str(wav_path)).frames)
    except (OSError, RuntimeError, ValueError):
        return None


def _tcp_snap_aligned_float32(pcm_snap: bytes, wav_path: Path) -> Optional[Any]:
    """TCP 缓冲和落盘 wav 帧数接近时直接返回波形，省得读文件。"""
    import numpy as np

    exp = _wav_pcm_frame_count(wav_path)
    if exp is None or exp <= 0 or not pcm_snap:
        return None
    exp_bytes = int(exp) * 2
    tol = max(640, int(0.06 * exp_bytes))
    if abs(len(pcm_snap) - exp_bytes) > tol:
        return None
    from speaker_id.pcm_io import pcm16le_bytes_to_float32

    use = pcm_snap[:exp_bytes]
    wav = pcm16le_bytes_to_float32(use)
    if int(wav.shape[0]) < int(exp):
        return None
    if int(wav.shape[0]) > int(exp):
        wav = wav[: int(exp)]
    return np.asarray(wav, dtype=np.float32)


def _format_ts(t: Optional[float] = None) -> str:
    t = time.time() if t is None else t
    ms = int((t % 1.0) * 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{ms:03d}"


def _log_audio_recv_span(key: str, recv_start_ts: float, recv_end_ts: float) -> None:
    span_ms = max(0, int((recv_end_ts - recv_start_ts) * 1000))
    print(
        f"[audio recv {_format_ts(recv_start_ts)} -> {_format_ts(recv_end_ts)} "
        f"span={span_ms}ms] {key}",
        flush=True,
    )


def _wav_recv_start_ts(wav: Path, key: str, recv_end_ts: float) -> float:
    from speaker_id.vad_wav_watch import silero_vad_samples_in_name

    ns = silero_vad_samples_in_name(key)
    if ns is not None and int(ns) > 0:
        return float(recv_end_ts) - float(ns) / 16000.0
    try:
        return float(wav.stat().st_mtime)
    except OSError:
        return float(recv_end_ts)


def _print_ingest(
    r: Any,
    *,
    verbose: bool,
    recv_ts: Optional[float] = None,
    recv_start_ts: Optional[float] = None,
) -> None:
    from speaker_id.streaming_ipc import ingest_result_to_dict

    d = ingest_result_to_dict(r)
    vid = d.get("v_id_for_use")
    label = "?" if vid is None else str(vid)
    key = d.get("wav_key")
    conf = d.get("confidence")
    cm = d.get("cluster_meta") or {}
    dense = cm.get("n_hdbscan_dense_clusters")
    is_noise = bool(d.get("hdbscan_was_noise"))
    csize = d.get("pred_cluster_size")
    emit_note = cm.get("emit_note") or ""
    hint = ""
    if emit_note:
        hint = f" [{emit_note}]"
    elif is_noise and conf is not None and float(conf) <= 0.2:
        hint = f" [HDBSCAN噪声,稠密簇={dense},sz={csize}]"
    elif is_noise and dense is not None and int(dense) == 0:
        hint = " [待≥3条同人稠密簇]"
    done_ts = time.time()
    if recv_ts is not None:
        embed_ms = int((done_ts - recv_ts) * 1000)
        if recv_start_ts is not None:
            recv_ms = max(0, int((recv_ts - recv_start_ts) * 1000))
            timing = f"(embed {embed_ms}ms, recv {recv_ms}ms)"
        else:
            timing = f"(proc {embed_ms}ms)"
        print(
            f"[audio done {_format_ts(done_ts)}] {key} -> v_id {label} conf={conf}{hint} "
            f"{timing}",
            flush=True,
        )
    else:
        print(f"{key} -> v_id {label} conf={conf}", flush=True)
    if verbose:
        print(d, flush=True)


def _on_ingest_done_factory(
    worker_pipe_holder: list,
    *,
    verbose: bool,
    refresh_panel: bool,
    last_key: str,
    recv_ts: float,
    recv_start_ts: Optional[float] = None,
) -> Any:
    def _done(r: Any) -> None:
        from speaker_id.cluster_panel import refresh_cluster_panel

        _print_ingest(
            r,
            verbose=verbose,
            recv_ts=recv_ts,
            recv_start_ts=recv_start_ts,
        )
        if refresh_panel and worker_pipe_holder:
            refresh_cluster_panel(worker_pipe_holder[0], clear=False, last_key=last_key)

    return _done


def _submit_pcm_segment(
    worker: Any,
    chunk: bytes,
    *,
    seg_idx: int,
    ts: int,
    verbose: bool,
    refresh_panel: bool,
    pipe: Any,
    dedupe_after_wav: bool = False,
) -> int:
    from speaker_id.pcm_io import pcm16le_bytes_to_float32

    if dedupe_after_wav and _recent_wav_ingest():
        if verbose:
            print("[pcm] skip vad segment (wav ingested recently)", flush=True)
        return seg_idx + 1

    key = f"vad_{seg_idx}_{ts}"
    recv_end_ts = time.time()
    _log_audio_recv_span(key, recv_end_ts, recv_end_ts)
    wav = pcm16le_bytes_to_float32(chunk)
    worker.submit_audio(
        wav,
        wav_key=key,
        client_ref={"ts_ms": ts, "source": "vad_pcm_socket", "recv_ts": recv_end_ts},
        on_done=_on_ingest_done_factory(
            [pipe],
            verbose=verbose,
            refresh_panel=refresh_panel,
            last_key=key,
            recv_ts=recv_end_ts,
            recv_start_ts=recv_end_ts,
        ),
    )
    return seg_idx + 1


def pcm_stream_loop_vad(
    worker: Any,
    pipe: Any,
    *,
    host: str,
    port: int,
    frame_bytes: int,
    timeout_sec: float,
    reconnect_sec: float,
    verbose: bool,
    refresh_panel: bool,
    dedupe_after_wav: bool = False,
    segment_end_idle_sec: float = 1.0,
    vad_end_mode: str = "silero_wav",
    vad_end_audio: str = "auto",
    tcp_seg_buf: Optional[TcpSegmentBuffer] = None,
) -> None:
    """按 robotmedia 的 vad 分段做声纹，段末和落盘 wav 对齐。"""
    from speaker_id.pcm_io import DEFAULT_FRAME_BYTES
    from speaker_id.tcp_audio import TCPAudioClient, current_time_ms

    min_bytes = _min_segment_bytes()
    max_bytes = _max_vad_segment_bytes()
    frame_bytes = int(frame_bytes) if frame_bytes > 0 else DEFAULT_FRAME_BYTES
    seg_idx = 0

    def emit_buffer(buf: bytes, ts: int) -> None:
        nonlocal seg_idx
        while len(buf) >= max_bytes:
            head, buf = buf[:max_bytes], buf[max_bytes:]
            seg_idx = _submit_pcm_segment(
                worker,
                head,
                seg_idx=seg_idx,
                ts=ts,
                verbose=verbose,
                refresh_panel=refresh_panel,
                pipe=pipe,
                dedupe_after_wav=dedupe_after_wav,
            )
        if len(buf) >= min_bytes:
            seg_idx = _submit_pcm_segment(
                worker,
                buf,
                seg_idx=seg_idx,
                ts=ts,
                verbose=verbose,
                refresh_panel=refresh_panel,
                pipe=pipe,
                dedupe_after_wav=dedupe_after_wav,
            )

    idle_poll = 0.5
    no_data_reconnect_streak = int(max(10, 30.0 / idle_poll))  # 非 seg_buf 时 ~30s 无字节重连
    stale_prune_streak = int(max(1, 60.0 / idle_poll))  # seg_buf 空闲时约 60s 检查陈旧缓冲
    end_idle = max(0.5, float(segment_end_idle_sec))
    segment_end_streak = int(max(1, end_idle / idle_poll))  # 段末静音后再提交
    use_seg_buf = tcp_seg_buf is not None and vad_end_mode in (
        "silero_wav",
        "silero_wav_or_idle",
    )
    use_idle_end = vad_end_mode in ("idle", "silero_wav_or_idle") and not use_seg_buf
    last_idle_log_mono = 0.0

    def flush_segment(reason: str) -> None:
        nonlocal pcm_buf, session_ts, empty_polls, seg_idx
        if not pcm_buf or session_ts is None:
            return
        emit_buffer(pcm_buf, session_ts)
        pcm_buf = b""
        session_ts = None
        empty_polls = 0
        print(f"[pcm] VAD segment end ({reason}), analyze", flush=True)

    while True:
        client = TCPAudioClient(host, port, timeout_sec=timeout_sec)
        pcm_buf = b""
        session_ts: Optional[int] = None
        empty_polls = 0
        try:
            client.connect(quiet=use_seg_buf)
            if not use_seg_buf:
                print(f"[pcm] connected, wait VAD PCM on :{port} ...", flush=True)
            while True:
                chunk = client.recv_chunk(4096, idle_timeout=idle_poll)
                if not chunk:
                    empty_polls += 1
                    if (
                        use_idle_end
                        and pcm_buf
                        and session_ts is not None
                        and empty_polls >= segment_end_streak
                    ):
                        flush_segment(f"idle {end_idle:.1f}s")
                    elif use_seg_buf and tcp_seg_buf is not None:
                        if empty_polls % stale_prune_streak == 0:
                            dropped = tcp_seg_buf.prune_stale()
                            if dropped > 0:
                                print(
                                    f"[pcm] TcpSegmentBuffer stale clear {dropped}B",
                                    flush=True,
                                )
                    elif (
                        not use_seg_buf
                        and session_ts is None
                        and empty_polls >= no_data_reconnect_streak
                    ):
                        now = time.monotonic()
                        if now - last_idle_log_mono >= 300.0:
                            last_idle_log_mono = now
                            print(
                                "[pcm] no PCM yet (idle); reconnect",
                                flush=True,
                            )
                        break
                    continue
                empty_polls = 0
                if use_seg_buf and tcp_seg_buf is not None:
                    tcp_seg_buf.append(chunk)
                    continue
                if session_ts is None:
                    session_ts = current_time_ms()
                    print(f"[pcm] VAD session start ts={session_ts}", flush=True)
                pcm_buf += chunk
        except (ConnectionError, OSError) as e:
            flush_segment(f"disconnect {e!r}")
            time.sleep(reconnect_sec)
        except socket.timeout as e:
            if pcm_buf and session_ts is not None:
                emit_buffer(pcm_buf, session_ts)
            print(f"[pcm] {e!r}", file=sys.stderr, flush=True)
            time.sleep(reconnect_sec)
        finally:
            client.close()


def pcm_stream_loop_fixed(
    worker: Any,
    pipe: Any,
    *,
    host: str,
    port: int,
    frame_bytes: int,
    segment_sec: float,
    timeout_sec: float,
    reconnect_sec: float,
    verbose: bool,
    refresh_panel: bool,
) -> None:
    """按固定秒数切片，仅供调试。"""
    from speaker_id.pcm_io import DEFAULT_FRAME_BYTES, segment_byte_length
    from speaker_id.tcp_audio import TCPAudioClient, current_time_ms

    seg_bytes = segment_byte_length(segment_sec)
    min_flush = _min_segment_bytes()
    frame_bytes = int(frame_bytes) if frame_bytes > 0 else DEFAULT_FRAME_BYTES
    seg_idx = 0

    def flush_tail(buf: bytes, ts: int) -> None:
        nonlocal seg_idx
        if len(buf) >= min_flush:
            seg_idx = _submit_pcm_segment(
                worker,
                buf,
                seg_idx=seg_idx,
                ts=ts,
                verbose=verbose,
                refresh_panel=refresh_panel,
                pipe=pipe,
            )

    while True:
        client = TCPAudioClient(host, port, timeout_sec=timeout_sec)
        pcm_buf = b""
        try:
            client.connect()
            while True:
                pcm_buf += client.recv_pcm(frame_bytes)
                while len(pcm_buf) >= seg_bytes:
                    chunk, pcm_buf = pcm_buf[:seg_bytes], pcm_buf[seg_bytes:]
                    seg_idx = _submit_pcm_segment(
                        worker,
                        chunk,
                        seg_idx=seg_idx,
                        ts=current_time_ms(),
                        verbose=verbose,
                        refresh_panel=refresh_panel,
                        pipe=pipe,
                    )
        except (ConnectionError, OSError) as e:
            flush_tail(pcm_buf, current_time_ms())
            pcm_buf = b""
            print(f"[pcm] {e!r}, reconnect in {reconnect_sec}s", file=sys.stderr, flush=True)
            time.sleep(reconnect_sec)
        except socket.timeout as e:
            flush_tail(pcm_buf, current_time_ms())
            print(f"[pcm] {e!r}", file=sys.stderr, flush=True)
            time.sleep(reconnect_sec)
        finally:
            client.close()


def _start_self_maintenance(args: argparse.Namespace) -> None:
    import os

    from speaker_id.runtime_maintenance import start_runtime_maintenance

    log_raw = os.environ.get("SV_ROBOT_LOG") or getattr(args, "robot_log", None)
    log_path = Path(str(log_raw)) if log_raw else None
    wav_dir = Path(str(args.pcm_wav_dir)) if not args.no_pcm else None
    interval = float(os.environ.get("SV_MAINT_INTERVAL_SEC", "120") or "120")
    log_max = float(os.environ.get("SV_LOG_MAX_MB", "8") or "8")
    log_keep = float(os.environ.get("SV_LOG_KEEP_MB", "4") or "4")
    wav_max = int(os.environ.get("SV_WAV_MAX_FILES", "2000") or "2000")
    wav_keep = int(os.environ.get("SV_WAV_KEEP_FILES", "1000") or "1000")
    wav_age = float(os.environ.get("SV_WAV_PRUNE_MIN_AGE_SEC", "600") or "600")
    if log_path is None and wav_dir is None:
        return
    start_runtime_maintenance(
        log_path=log_path,
        wav_dir=wav_dir,
        interval_sec=interval,
        log_max_mb=log_max,
        log_keep_mb=log_keep,
        wav_max_files=wav_max,
        wav_keep_files=wav_keep,
        wav_min_age_sec=wav_age,
    )
    caps = []
    if log_path is not None:
        caps.append(f"log<={log_max}MB keep {log_keep}MB")
    if wav_dir is not None and wav_max > 0:
        caps.append(f"wav<={wav_max} keep {wav_keep}")
    print(
        f"[maint] self-care on interval={interval}s ({', '.join(caps)})",
        flush=True,
    )


def run_service(args: argparse.Namespace) -> int:
    _ensure_path()
    import os

    os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")

    if not args.skip_preflight:
        _run_preflight(args)

    _start_self_maintenance(args)

    try:
        from cy_audio_node.time_utils import current_time_ms as _  # noqa: F401

        print("cy_audio_node: OK", flush=True)
    except ImportError:
        print("cy_audio_node: 未安装（时间戳用本地 clock）", flush=True)

    print("loading ONNX pipeline...", flush=True)
    pipe = _make_pipeline(args)
    from speaker_id.pipeline_worker import PipelineWorker
    from speaker_id.pipeline_factory import default_pipeline_queue_maxsize

    worker = PipelineWorker(pipe, queue_maxsize=default_pipeline_queue_maxsize())
    print("ready.", flush=True)

    threads: list[threading.Thread] = []

    if not args.no_api:
        from api_server import make_handler
        from http.server import ThreadingHTTPServer

        handler = make_handler(worker)
        server = ThreadingHTTPServer((str(args.api_host), int(args.api_port)), handler)

        def _api() -> None:
            print(
                f"API http://{args.api_host}:{args.api_port}  POST /v1/ingest  GET /v1/panel",
                flush=True,
            )
            server.serve_forever()

        t = threading.Thread(target=_api, name="sv-api", daemon=True)
        t.start()
        threads.append(t)

    if not args.no_pcm:

        def _ingest_wav_file(
            wav: Path,
            key: str,
            *,
            recv_start_ts: Optional[float] = None,
            recv_end_ts: Optional[float] = None,
            skip_recv_log: bool = False,
        ) -> None:
            from speaker_id.cluster_panel import refresh_cluster_panel

            _note_wav_ingest()
            end_ts = recv_end_ts if recv_end_ts is not None else time.time()
            start_ts = (
                recv_start_ts
                if recv_start_ts is not None
                else _wav_recv_start_ts(wav, key, end_ts)
            )
            if not skip_recv_log:
                _log_audio_recv_span(key, start_ts, end_ts)
            r = worker.ingest_wav_path(
                wav,
                wav_key=key,
                client_ref={"recv_ts": end_ts, "source": "vad_wav_watch"},
            )
            _print_ingest(
                r,
                verbose=bool(args.verbose),
                recv_ts=end_ts,
                recv_start_ts=start_ts,
            )
            if not bool(args.no_panel):
                refresh_cluster_panel(pipe, clear=False, last_key=key)

        pcm_src = str(getattr(args, "pcm_source", "wav_dir"))
        if pcm_src == "both":
            print(
                "WARN: pcm-source=both 易对同一句重复 ingest；板端建议 wav_dir",
                flush=True,
            )
        tcp_seg_buf: Optional[TcpSegmentBuffer] = None
        if pcm_src in ("tcp", "both"):
            pcm_target = pcm_stream_loop_vad if args.pcm_mode == "vad" else pcm_stream_loop_fixed
            if args.pcm_mode == "vad" and pcm_src == "tcp":
                tcp_seg_buf = TcpSegmentBuffer(
                    max_bytes=_max_vad_segment_bytes(),
                    stale_sec=float(getattr(args, "pcm_tcp_buf_stale_sec", 45.0)),
                )
            tcp_kw: dict = {
                "worker": worker,
                "pipe": pipe,
                "host": str(args.pcm_host),
                "port": int(args.pcm_port),
                "frame_bytes": int(args.pcm_frame_bytes),
                "timeout_sec": float(args.pcm_timeout),
                "reconnect_sec": float(args.pcm_reconnect_sec),
                "verbose": bool(args.verbose),
                "refresh_panel": not bool(args.no_panel),
            }
            if args.pcm_mode == "fixed":
                tcp_kw["segment_sec"] = float(args.pcm_segment_sec)
            else:
                tcp_kw["segment_end_idle_sec"] = float(args.pcm_segment_end_idle_sec)
                tcp_kw["vad_end_mode"] = str(args.pcm_vad_end_mode)
                tcp_kw["vad_end_audio"] = str(args.pcm_vad_end_audio)
                if tcp_seg_buf is not None:
                    tcp_kw["tcp_seg_buf"] = tcp_seg_buf
                if pcm_src == "both":
                    tcp_kw["dedupe_after_wav"] = True
            t = threading.Thread(
                target=pcm_target,
                kwargs=tcp_kw,
                name="sv-pcm-tcp",
                daemon=True,
            )
            t.start()
            threads.append(t)
            end_hint = ""
            if args.pcm_mode == "vad" and pcm_src == "tcp":
                end_hint = (
                    f" vad_end={args.pcm_vad_end_mode}"
                    f" audio={args.pcm_vad_end_audio}"
                )
            print(
                f"PCM tcp://{args.pcm_host}:{args.pcm_port} mode={args.pcm_mode}{end_hint}",
                flush=True,
            )
        if pcm_src == "tcp" and args.pcm_mode == "vad" and tcp_seg_buf is not None:
            from speaker_id.vad_wav_watch import watch_vad_wav_loop

            wdir = Path(str(args.pcm_wav_dir))
            end_audio = str(args.pcm_vad_end_audio)

            def _on_vad_end(p: Path, name: str) -> None:
                pcm_snap, seg_start_ts, recv_end_ts = tcp_seg_buf.snapshot_clear_span()
                recv_start_ts = seg_start_ts if seg_start_ts is not None else recv_end_ts
                if end_audio == "wav":
                    _log_audio_recv_span(name, recv_start_ts, recv_end_ts)
                    _ingest_wav_file(
                        p,
                        name,
                        recv_start_ts=recv_start_ts,
                        recv_end_ts=recv_end_ts,
                        skip_recv_log=True,
                    )
                    print(f"[pcm] vad_end {name} -> wav ingest", flush=True)
                    return
                if end_audio == "tcp":
                    if len(pcm_snap) < _min_segment_bytes():
                        print(f"[pcm] vad_end {name} tcp empty, skip", flush=True)
                        return
                    from speaker_id.pcm_io import pcm16le_bytes_to_float32

                    _log_audio_recv_span(name, recv_start_ts, recv_end_ts)
                    worker.submit_audio(
                        pcm16le_bytes_to_float32(pcm_snap),
                        wav_key=name,
                        client_ref={
                            "recv_ts": recv_end_ts,
                            "source": "tcp_pcm_raw",
                            "wav_path": str(p),
                        },
                        on_done=_on_ingest_done_factory(
                            [pipe],
                            verbose=bool(args.verbose),
                            refresh_panel=not bool(args.no_panel),
                            last_key=name,
                            recv_ts=recv_end_ts,
                            recv_start_ts=recv_start_ts,
                        ),
                    )
                    print(f"[pcm] vad_end {name} -> tcp memory (raw)", flush=True)
                    return
                # auto 模式对齐用 TCP 内存，否则读 wav。
                aligned = _tcp_snap_aligned_float32(pcm_snap, p)
                if aligned is not None:
                    _log_audio_recv_span(name, recv_start_ts, recv_end_ts)
                    worker.submit_audio(
                        aligned,
                        wav_key=name,
                        client_ref={
                            "recv_ts": recv_end_ts,
                            "source": "tcp_pcm_aligned",
                            "wav_path": str(p),
                        },
                        on_done=_on_ingest_done_factory(
                            [pipe],
                            verbose=bool(args.verbose),
                            refresh_panel=not bool(args.no_panel),
                            last_key=name,
                            recv_ts=recv_end_ts,
                            recv_start_ts=recv_start_ts,
                        ),
                    )
                    print(
                        f"[pcm] vad_end {name} -> tcp memory aligned "
                        f"({len(pcm_snap)}B)",
                        flush=True,
                    )
                else:
                    _log_audio_recv_span(name, recv_start_ts, recv_end_ts)
                    _ingest_wav_file(
                        p,
                        name,
                        recv_start_ts=recv_start_ts,
                        recv_end_ts=recv_end_ts,
                        skip_recv_log=True,
                    )
                    print(
                        f"[pcm] vad_end {name} -> fallback wav "
                        f"(tcp {len(pcm_snap)}B misaligned)",
                        flush=True,
                    )

            t = threading.Thread(
                target=watch_vad_wav_loop,
                kwargs={
                    "on_vad_end": _on_vad_end,
                    "watch_dir": wdir,
                    "skip_existing": True,
                    "stable_sec": float(getattr(args, "pcm_vad_end_stable_sec", 0.15)),
                    "min_duration_sec": float(getattr(args, "pcm_min_duration_sec", 2.0)),
                    "max_duration_sec": float(getattr(args, "pcm_max_duration_sec", 15.0)),
                },
                name="sv-vad-end-signal",
                daemon=True,
            )
            t.start()
            threads.append(t)
        if pcm_src in ("wav_dir", "both"):
            from speaker_id.vad_wav_watch import watch_vad_wav_loop

            wdir = Path(str(args.pcm_wav_dir))
            t = threading.Thread(
                target=watch_vad_wav_loop,
                kwargs={
                    "ingest_wav": _ingest_wav_file,
                    "watch_dir": wdir,
                    "skip_existing": True,
                    "min_duration_sec": float(getattr(args, "pcm_min_duration_sec", 2.0)),
                    "max_duration_sec": float(getattr(args, "pcm_max_duration_sec", 15.0)),
                },
                name="sv-pcm-wav",
                daemon=True,
            )
            t.start()
            threads.append(t)
            print(f"PCM wav-watch {wdir} (robotmedia silero_vad_*.wav)", flush=True)

    if not threads:
        print("需要至少开启 API 或 PCM（勿同时 --no-api --no-pcm）", file=sys.stderr)
        return 2

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("shutdown.", flush=True)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    from speaker_id.pipeline_factory import (
        default_cluster_selection_epsilon,
        default_max_hdbscan_utterances,
        default_max_ingest_audio_sec,
        default_ort_threads,
    )

    ap = argparse.ArgumentParser(description="机器人声纹一体服务（API + PCM）")
    ap.add_argument("--api-host", default="0.0.0.0")
    ap.add_argument("--api-port", type=int, default=8765)
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--no-pcm", action="store_true")
    ap.add_argument("--pcm-host", default="127.0.0.1")
    ap.add_argument(
        "--pcm-port",
        type=int,
        default=5001,
        help="robotmedia [vad_pcm_socket]（无需改 robotmedia VAD 代码）",
    )
    ap.add_argument(
        "--pcm-source",
        choices=("tcp", "wav_dir", "both"),
        default="wav_dir",
        help="wav_dir=监听 silero_vad wav(板端推荐)；tcp=5001；both=并行且易重复",
    )
    ap.add_argument(
        "--pcm-wav-dir",
        default="/userdata/cloud_voice",
        help="robotmedia silero_vad_*.wav 目录",
    )
    ap.add_argument(
        "--pcm-min-duration-sec",
        type=float,
        default=2.0,
        help="短于该时长的 VAD wav 忽略；默认 2.0s，过滤碎段",
    )
    ap.add_argument(
        "--pcm-mode",
        choices=("vad", "fixed"),
        default="vad",
        help="vad=按 TCP 会话(VAD段) ingest；fixed=按秒切片(调试)",
    )
    ap.add_argument("--pcm-frame-bytes", type=int, default=3200)
    ap.add_argument("--pcm-segment-sec", type=float, default=3.0, help="仅 pcm-mode=fixed")
    ap.add_argument(
        "--pcm-segment-end-idle-sec",
        type=float,
        default=1.0,
        help="pcm-mode=vad：idle 模式或 silero_wav_or_idle 的静音兜底秒数",
    )
    ap.add_argument(
        "--pcm-vad-end-mode",
        choices=("silero_wav", "idle", "silero_wav_or_idle"),
        default="silero_wav",
        help="TCP 段末：silero_wav=跟 robotmedia VAD_END 落盘信号；idle=仅静音",
    )
    ap.add_argument(
        "--pcm-vad-end-audio",
        choices=("auto", "wav", "tcp"),
        default="auto",
        help="auto=TCP 对齐则内存(快)否则读 wav(准)；wav/tcp=固定",
    )
    ap.add_argument(
        "--pcm-vad-end-stable-sec",
        type=float,
        default=0.15,
        help="TCP 模式：vad_end 信号 wav 稳定等待(秒)，略小于 wav_dir 降延迟",
    )
    ap.add_argument(
        "--pcm-tcp-buf-stale-sec",
        type=float,
        default=45.0,
        help="TCP 段缓冲无新字节超过该秒数则丢弃，防 vad_end 丢失导致内存涨",
    )
    ap.add_argument("--pcm-timeout", type=float, default=0.0)
    ap.add_argument("--pcm-reconnect-sec", type=float, default=5.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument(
        "--robotmedia-config",
        default="/usr/share/robotmedia/config/config.ini",
    )
    ap.add_argument("--onnx-path", type=Path, default=None)
    ap.add_argument("--ort-threads", type=int, default=default_ort_threads())
    ap.add_argument(
        "--max-hdbscan-utterances",
        type=int,
        default=default_max_hdbscan_utterances(),
    )
    ap.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=3,
        help="声纹默认 3：至少 3 条才成说话人簇",
    )
    ap.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=2,
        help="声纹默认 2",
    )
    ap.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=default_cluster_selection_epsilon(),
        help="略>0 合并相近簇，减轻流式全噪声",
    )
    ap.add_argument(
        "--max-ingest-audio-sec",
        type=float,
        default=default_max_ingest_audio_sec(),
        help="单条 embed 最长秒数，超长 VAD 只取前 N 秒",
    )
    ap.add_argument(
        "--pcm-max-duration-sec",
        type=float,
        default=15.0,
        help="忽略文件名采样数超过该时长的 silero_vad wav",
    )
    args = ap.parse_args(argv)
    return run_service(args)


if __name__ == "__main__":
    raise SystemExit(main())
