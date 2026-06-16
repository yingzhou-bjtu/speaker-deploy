"""板端声纹包，用 ONNX，不依赖 PyTorch。"""

from .embedding_store import (
    load_embedding_npz,
    match_utterance,
    register_speaker_embedding,
    save_embedding_npz,
)
from .engine import ClusterBatchResult, SpeakerEmbedCluster, default_sv_onnx_path
from .streaming_ipc import ingest_message, ingest_result_to_dict, spawn_streaming_worker
from .pipeline_factory import build_pipeline_kwargs, make_streaming_pipeline
from .streaming_pipeline import StreamIngestResult, StreamingClusterPipeline

__all__ = [
    "ClusterBatchResult",
    "SpeakerEmbedCluster",
    "StreamingClusterPipeline",
    "StreamIngestResult",
    "build_pipeline_kwargs",
    "make_streaming_pipeline",
    "default_sv_onnx_path",
    "ingest_message",
    "ingest_result_to_dict",
    "load_embedding_npz",
    "match_utterance",
    "register_speaker_embedding",
    "save_embedding_npz",
    "spawn_streaming_worker",
]
