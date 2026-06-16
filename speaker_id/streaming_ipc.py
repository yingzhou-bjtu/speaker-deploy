"""进程间通信，用 dict 走队列或 Redis，子进程独占模型。"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .streaming_pipeline import StreamIngestResult, StreamingClusterPipeline

SCHEMA_V1 = 1


def shutdown_message() -> Dict[str, Any]:
    return {"schema": SCHEMA_V1, "op": "shutdown"}


def ingest_message(
    wav_path: Union[str, Path],
    *,
    wav_key: Optional[str] = None,
    client_ref: Any = None,
    true_speaker_slot: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_V1,
        "op": "ingest",
        "wav_path": str(Path(wav_path).resolve()),
        "wav_key": wav_key,
        "client_ref": client_ref,
        "true_speaker_slot": true_speaker_slot,
    }


def is_shutdown_message(msg: Any) -> bool:
    return isinstance(msg, dict) and int(msg.get("schema", 0)) == SCHEMA_V1 and msg.get("op") == "shutdown"


def ingest_result_to_dict(r: StreamIngestResult) -> Dict[str, Any]:
    return {
        "order_index": r.order_index,
        "wav_key": r.wav_key,
        "client_ref": r.client_ref,
        "v_id": r.v_id,
        "hdbscan_was_noise": r.hdbscan_was_noise,
        "cosine_to_pred_cluster_centroid": r.cosine_to_pred_cluster_centroid,
        "pred_cluster_size": r.pred_cluster_size,
        "confidence": r.confidence,
        "confidence_band": r.confidence_band,
        "emit_id_min_effective": r.emit_id_min_effective,
        "id_emitted": r.id_emitted,
        "v_id_for_use": r.v_id_for_use,
        "cluster_meta": r.cluster_meta,
        "n_distinct_pred_clusters": r.n_distinct_pred_clusters,
        "prefix_hdbscan_noise_true_count": r.prefix_hdbscan_noise_true_count,
        "stable_v_id": r.stable_v_id,
        "stable_v_id_for_use": r.stable_v_id_for_use,
        "eval": r.eval,
    }


def _pipeline_kwargs_normalize(kw: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kw)
    if "offline_root" in out and out["offline_root"] is not None:
        out["offline_root"] = Path(out["offline_root"])
    return out


def _worker_loop(
    jobs_q: Any,
    results_q: Any,
    pipeline_kwargs: Dict[str, Any],
) -> None:
    """子进程工作循环，须在顶层定义以便 spawn。"""
    kw = _pipeline_kwargs_normalize(pipeline_kwargs)
    pipe = StreamingClusterPipeline(**kw)
    while True:
        try:
            msg = jobs_q.get()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        if is_shutdown_message(msg):
            break
        if not isinstance(msg, dict) or msg.get("op") != "ingest":
            results_q.put(
                {
                    "schema": SCHEMA_V1,
                    "ok": False,
                    "error": f"unknown message: {type(msg)}",
                    "traceback": "",
                    "job": msg,
                }
            )
            continue
        p = Path(msg["wav_path"])
        try:
            r = pipe.ingest_wav(
                p,
                wav_key=msg.get("wav_key"),
                client_ref=msg.get("client_ref"),
                true_speaker_slot=msg.get("true_speaker_slot"),
            )
            results_q.put(
                {"schema": SCHEMA_V1, "ok": True, "result": ingest_result_to_dict(r)}
            )
        except Exception as e:
            results_q.put(
                {
                    "schema": SCHEMA_V1,
                    "ok": False,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "job": msg,
                }
            )


def spawn_streaming_worker(
    *,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    jobs_maxsize: int = 256,
    results_maxsize: int = 256,
    context: Optional[str] = None,
) -> Tuple[mp.Process, Any, Any]:
    """启动子进程 worker，返回进程和两个队列。"""
    ctx = mp.get_context(context or "spawn")
    jq = ctx.Queue(maxsize=int(jobs_maxsize))
    rq = ctx.Queue(maxsize=int(results_maxsize))
    kw = dict(pipeline_kwargs or {})
    proc = ctx.Process(
        target=_worker_loop,
        args=(jq, rq, kw),
        name="streaming-cluster-ipc-worker",
        daemon=False,
    )
    proc.start()
    return proc, jq, rq


def _print_v_id_line(path: Path, wav_key: Optional[str], v_id: int) -> None:
    name = str(wav_key) if wav_key else path.name
    print(f"{name} -> {v_id}", flush=True)


def redis_streams_loop(
    *,
    redis_url: str,
    list_jobs: str = "speak:stream:jobs",
    list_results: str = "speak:stream:results",
    pipeline: Optional[StreamingClusterPipeline] = None,
    pipe: Optional[StreamingClusterPipeline] = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 5.0,
    decode_responses: bool = True,
    print_line_only: bool = False,
    refresh_panel: bool = False,
    panel_clear: bool = True,
    push_results: bool = True,
) -> None:
    """从 Redis 取任务做 ingest，可选每次刷新面板。"""
    try:
        import redis  # type: ignore
    except ImportError as e:
        raise ImportError("redis_streams_loop 需要 pip install redis") from e

    r = redis.from_url(redis_url, decode_responses=decode_responses)
    shared = pipe if pipe is not None else pipeline
    if shared is None:
        kw = _pipeline_kwargs_normalize(dict(pipeline_kwargs or {}))
        shared = StreamingClusterPipeline(**kw)
    brpop_timeout = int(max(1, min(30, round(timeout_sec))))
    panel_mode = bool(refresh_panel) or bool(print_line_only)
    do_push = bool(push_results) and not panel_mode

    while True:
        item = r.brpop(list_jobs, timeout=brpop_timeout)
        if item is None:
            continue
        _, raw = item
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            err = {
                "schema": SCHEMA_V1,
                "ok": False,
                "error": f"json decode: {e}",
                "traceback": "",
                "job": raw,
            }
            if do_push:
                r.lpush(list_results, json.dumps(err, ensure_ascii=False))
            continue

        if is_shutdown_message(msg):
            if do_push:
                ack = {"schema": SCHEMA_V1, "ok": True, "result": {"note": "shutdown"}}
                r.lpush(list_results, json.dumps(ack, ensure_ascii=False))
            break

        if not isinstance(msg, dict) or msg.get("op") != "ingest":
            err = {
                "schema": SCHEMA_V1,
                "ok": False,
                "error": "expected op=ingest",
                "traceback": "",
                "job": msg,
            }
            if do_push:
                r.lpush(list_results, json.dumps(err, ensure_ascii=False))
            continue

        p = Path(msg["wav_path"])
        try:
            out = shared.ingest_wav(
                p,
                wav_key=msg.get("wav_key"),
                client_ref=msg.get("client_ref"),
                true_speaker_slot=msg.get("true_speaker_slot"),
            )
            if refresh_panel:
                from .cluster_panel import refresh_cluster_panel

                name = str(msg.get("wav_key") or p.name)
                refresh_cluster_panel(
                    shared, clear=bool(panel_clear), last_key=name
                )
                continue
            if print_line_only:
                _print_v_id_line(p, msg.get("wav_key"), int(out.v_id))
                continue
            payload = {
                "schema": SCHEMA_V1,
                "ok": True,
                "result": ingest_result_to_dict(out),
            }
        except Exception as e:
            if refresh_panel or print_line_only:
                name = str(msg.get("wav_key") or p.name)
                print(f"{name} -> ERROR {e}", file=sys.stderr, flush=True)
                continue
            payload = {
                "schema": SCHEMA_V1,
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "job": msg,
            }
        if do_push:
            r.lpush(list_results, json.dumps(payload, ensure_ascii=False))


def redis_push_job(
    redis_url: str,
    list_jobs: str,
    msg: Dict[str, Any],
    *,
    decode_responses: bool = True,
) -> None:
    import redis  # type: ignore

    r = redis.from_url(redis_url, decode_responses=decode_responses)
    r.lpush(list_jobs, json.dumps(msg, ensure_ascii=False))


def redis_pop_result(
    redis_url: str,
    list_results: str,
    *,
    timeout_sec: float = 300.0,
    decode_responses: bool = True,
) -> Dict[str, Any]:
    import redis  # type: ignore

    r = redis.from_url(redis_url, decode_responses=decode_responses)
    t = int(max(1, round(timeout_sec)))
    item = r.brpop(list_results, timeout=t)
    if item is None:
        raise TimeoutError("brpop results timeout")
    _, raw = item
    return json.loads(raw)
