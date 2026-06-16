"""构造 StreamingClusterPipeline 的公共入口，命令行和 API 共用默认参数。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

_DEPLOY_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REGISTRY = _DEPLOY_ROOT / "speaker_registry.npz"


def default_speaker_registry_path() -> Path:
    raw = os.environ.get("SV_SPEAKER_REGISTRY", "").strip()
    return Path(raw).resolve() if raw else _DEFAULT_REGISTRY


def default_max_hdbscan_utterances() -> int:
    raw = os.environ.get("SV_MAX_HDBSCAN_UTTERANCES", "60").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 60


def default_ort_threads() -> int:
    raw = os.environ.get("SV_ORT_THREADS", "4").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 4


def default_cluster_selection_epsilon() -> float:
    raw = os.environ.get("SV_HDBSCAN_EPS", "0.05").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.05


def default_max_cluster_promoted_registry() -> int:
    raw = os.environ.get("SV_MAX_CLUSTER_PROMOTED_REGISTRY", "30").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 30


def default_max_ingest_audio_sec() -> float:
    raw = os.environ.get("SV_MAX_INGEST_AUDIO_SEC", "15").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 15.0


def default_pipeline_queue_maxsize() -> int:
    raw = os.environ.get("SV_PIPELINE_QUEUE_MAX", "128").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 128


def build_pipeline_kwargs(
    *,
    onnx_path: Optional[Path] = None,
    offline_root: Optional[Path] = None,
    ort_threads: Optional[int] = None,
    max_hdbscan_utterances: Optional[int] = None,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    cluster_selection_epsilon: Optional[float] = None,
    speaker_registry_path: Optional[Path] = None,
    persist_registry: bool = True,
    max_ingest_audio_sec: Optional[float] = None,
    **extra: Any,
) -> dict[str, Any]:
    cap = (
        default_max_hdbscan_utterances()
        if max_hdbscan_utterances is None
        else int(max_hdbscan_utterances)
    )
    kw: dict[str, Any] = {
        "device": "cpu",
        "sv_backend": "onnx",
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "cluster_selection_epsilon": float(
            default_cluster_selection_epsilon()
            if cluster_selection_epsilon is None
            else cluster_selection_epsilon
        ),
        "ort_threads": int(default_ort_threads() if ort_threads is None else ort_threads),
        "speaker_registry_path": (
            default_speaker_registry_path()
            if speaker_registry_path is None
            else Path(speaker_registry_path).resolve()
        ),
        "persist_registry": bool(persist_registry),
        "enable_registry_match": True,
        "enable_stable_cluster_promote": True,
        "max_cluster_promoted_registry": default_max_cluster_promoted_registry(),
    }
    if cap > 0:
        kw["max_hdbscan_utterances"] = cap
    if onnx_path is not None:
        kw["onnx_path"] = Path(onnx_path).resolve()
    if offline_root is not None:
        kw["offline_root"] = Path(offline_root).resolve()
    if max_ingest_audio_sec is not None and float(max_ingest_audio_sec) > 0:
        kw["max_ingest_audio_sec"] = float(max_ingest_audio_sec)
    kw.update(extra)
    return kw


def make_streaming_pipeline(**kwargs: Any):
    from .streaming_pipeline import StreamingClusterPipeline

    return StreamingClusterPipeline(**build_pipeline_kwargs(**kwargs))
