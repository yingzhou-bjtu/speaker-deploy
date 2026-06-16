#!/usr/bin/env python3
"""声纹 HTTP API，单进程常驻 pipeline，业务 POST wav 路径做 ingest。"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(raw)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    n = int(handler.headers.get("Content-Length", "0") or 0)
    if n <= 0:
        return {}
    raw = handler.rfile.read(n)
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def make_handler(backend: Any) -> type[BaseHTTPRequestHandler]:
    """backend 是 PipelineWorker 或带锁的 pipe 元组。"""
    if isinstance(backend, tuple):
        pipe, lock = backend

        def ingest_wav(wav: Path, wav_key: str) -> Any:
            with lock:
                return pipe.ingest_wav(wav, wav_key=wav_key)

        def get_pipe() -> Any:
            return pipe
    else:
        worker = backend

        def ingest_wav(wav: Path, wav_key: str) -> Any:
            return worker.ingest_wav_path(wav, wav_key=wav_key)

        def get_pipe() -> Any:
            return worker._pipe

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f"[sv-api] {self.address_string()} - {fmt % args}\n")
            sys.stderr.flush()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/", "/health", "/v1/health"):
                _json_response(self, 200, {"ok": True, "service": "speaker-voiceprint"})
                return
            if path == "/v1/panel":
                from speaker_id.cluster_panel import cluster_panel_counts

                pipe = get_pipe()
                assigned, uncertain, total = cluster_panel_counts(pipe)
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "assigned": assigned,
                        "uncertain": uncertain,
                        "total": total,
                    },
                )
                return
            if path == "/v1/buffer":
                pipe = get_pipe()
                keys = pipe.buffer_wav_keys()
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "keys": keys,
                        "n": int(pipe.n_arrived),
                    },
                )
                return
            _json_response(self, 404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path not in ("/v1/ingest", "/ingest"):
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            try:
                body = _read_json_body(self)
            except (json.JSONDecodeError, ValueError) as e:
                _json_response(self, 400, {"ok": False, "error": str(e)})
                return
            wav_raw = body.get("wav_path") or body.get("path")
            if not wav_raw:
                _json_response(self, 400, {"ok": False, "error": "wav_path required"})
                return
            wav = Path(str(wav_raw)).resolve()
            if not wav.is_file():
                _json_response(self, 404, {"ok": False, "error": f"not a file: {wav}"})
                return
            wav_key = body.get("wav_key") or wav.name
            from speaker_id.streaming_ipc import ingest_result_to_dict

            try:
                r = ingest_wav(wav, str(wav_key))
                _json_response(
                    self,
                    200,
                    {"ok": True, "result": ingest_result_to_dict(r)},
                )
            except Exception as e:
                _json_response(self, 500, {"ok": False, "error": repr(e)})

    return Handler


def run_api_server(
    backend: Any,
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    handler = make_handler(backend)
    server = ThreadingHTTPServer((host, int(port)), handler)
    print(f"speaker API http://{host}:{port}  POST /v1/ingest  GET /v1/panel", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from speaker_id.pipeline_factory import (
        build_pipeline_kwargs,
        default_max_hdbscan_utterances,
        default_max_ingest_audio_sec,
        default_ort_threads,
        default_cluster_selection_epsilon,
        default_pipeline_queue_maxsize,
    )

    ap = argparse.ArgumentParser(description="声纹 HTTP API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--onnx-path", type=Path, default=None)
    ap.add_argument("--ort-threads", type=int, default=default_ort_threads())
    ap.add_argument(
        "--max-hdbscan-utterances",
        type=int,
        default=default_max_hdbscan_utterances(),
    )
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=3)
    ap.add_argument("--hdbscan-min-samples", type=int, default=2)
    ap.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=default_cluster_selection_epsilon(),
    )
    ap.add_argument(
        "--max-ingest-audio-sec",
        type=float,
        default=default_max_ingest_audio_sec(),
    )
    args = ap.parse_args(argv)

    from speaker_id.pipeline_worker import PipelineWorker
    from speaker_id.streaming_pipeline import StreamingClusterPipeline

    cap = int(args.max_hdbscan_utterances)
    max_sec = float(args.max_ingest_audio_sec or 0)
    kw = build_pipeline_kwargs(
        onnx_path=args.onnx_path,
        ort_threads=int(args.ort_threads),
        max_hdbscan_utterances=cap if cap > 0 else None,
        min_cluster_size=int(args.hdbscan_min_cluster_size),
        min_samples=int(args.hdbscan_min_samples),
        cluster_selection_epsilon=float(args.hdbscan_cluster_selection_epsilon),
        max_ingest_audio_sec=max_sec if max_sec > 0 else None,
    )

    print("loading ONNX pipeline...", flush=True)
    pipe = StreamingClusterPipeline(**kw)
    worker = PipelineWorker(pipe, queue_maxsize=default_pipeline_queue_maxsize())
    print("ready.", flush=True)
    run_api_server(worker, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
