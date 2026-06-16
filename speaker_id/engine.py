"""声纹嵌入和 HDBSCAN 聚类，支持 PyTorch 或 ONNX。"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .cluster_hdbscan import cluster_speakers as _cluster_speakers_impl

_DEFAULT_SV_MODEL = "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
_DEFAULT_CKPT_NAME = "pretrained_eres2netv2w24s4ep4.ckpt"
_DEFAULT_ONNX_NAME = "pretrained_eres2netv2w24s4ep4.onnx"
BackendName = Literal["auto", "torch", "onnx"]


def _deploy_root() -> Path:
    """deploy 目录根路径。"""
    return Path(__file__).resolve().parents[1]


def _speak_root() -> Path:
    env = os.environ.get("SPEAK_REPO_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return _deploy_root().parent


def _landing_root() -> Path:
    return _deploy_root()


def _d3_root() -> Path:
    env = os.environ.get("D3_SPEAKER_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return _speak_root() / "opensource" / "3D-Speaker"


def _model_dir_name() -> str:
    return _DEFAULT_SV_MODEL.split("/")[-1]


def default_sv_offline_root() -> Path:
    return _landing_root() / "pretrained"


def default_sv_onnx_path(offline_root: Path | None = None) -> Path:
    env = os.environ.get("SV_ONNX_PATH", "").strip()
    if env:
        return Path(env).resolve()
    root = Path(offline_root) if offline_root is not None else default_sv_offline_root()
    candidates = (
        root / "eres2net_sv.onnx",
        root / "iic" / _model_dir_name() / _DEFAULT_ONNX_NAME,
    )
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return candidates[0].resolve()


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_backend(name: str, *, offline_root: Path | None = None) -> BackendName:
    n = (name or "onnx").strip().lower()
    if n == "auto":
        n = os.environ.get("SV_BACKEND", "onnx").strip().lower() or "onnx"
    if n not in ("torch", "onnx"):
        raise ValueError(f"板端 deploy 仅支持 onnx|torch，收到: {name!r}")
    if n == "torch" and not _torch_available():
        raise RuntimeError("未安装 torch，请使用 --sv-backend onnx 或安装 PyTorch")
    onnx_p = default_sv_onnx_path(offline_root)
    if n == "onnx" and not onnx_p.is_file():
        raise FileNotFoundError(f"缺少 ONNX 模型: {onnx_p}")
    return n  # type: ignore[return-value]


def load_childmandarin_eval_module() -> Any:
    d3 = _d3_root()
    path = d3 / "scripts" / "childmandarin_eval_100.py"
    if str(d3) not in sys.path:
        sys.path.insert(0, str(d3))
    spec = importlib.util.spec_from_file_location("childmandarin_eval_100", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"找不到 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _l2_row_normalize(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    row_n = np.linalg.norm(x, axis=1, keepdims=True)
    row_n = np.maximum(row_n, 1e-12)
    return x / row_n


@dataclass
class ClusterBatchResult:
    labels: np.ndarray
    meta: Dict[str, Any]
    hdbscan_was_noise: np.ndarray
    #: HDBSCAN 原始簇号，噪声为负一。labels 是重标后的编号。
    dense_labels: np.ndarray


class SpeakerEmbedCluster:
    """嵌入和聚类，onnx 后端板端不需要 torch。"""

    def __init__(
        self,
        offline_root: Path | None = None,
        sv_model_id: str = _DEFAULT_SV_MODEL,
        device: str = "auto",
        backend: str = "auto",
        onnx_path: Path | None = None,
        ort_threads: int = 0,
    ) -> None:
        self.offline_root = (
            Path(offline_root) if offline_root is not None else default_sv_offline_root()
        )
        self.sv_model_id = sv_model_id
        self.device_str = str(device)
        self.backend: BackendName = resolve_backend(backend, offline_root=self.offline_root)
        self.onnx_path = (
            Path(onnx_path).resolve()
            if onnx_path is not None
            else default_sv_onnx_path(self.offline_root)
        )
        self.ort_threads = int(ort_threads)
        self._cev: Any = None
        self._sv: Any = None
        self._fbank: Any = None
        self._onnx: Any = None
        self._torch_device: Any = None

    def _ensure_torch(self) -> None:
        if self._sv is not None:
            return
        import torch

        def _pick_device(name: str) -> "torch.device":
            if name in ("auto", "cuda:0", "") and torch.cuda.is_available():
                return torch.device("cuda:0")
            if name.startswith("cuda") and torch.cuda.is_available():
                return torch.device(name)
            return torch.device("cpu")

        self._torch_device = _pick_device(self.device_str)
        self._cev = load_childmandarin_eval_module()
        try:
            import modelscope.pipelines.util as _ms_util  # type: ignore

            _ms_util.is_official_hub_path = lambda p, *a, **kw: bool(
                isinstance(p, str) and (p.startswith("iic/") or p.startswith("damo/"))
            )
            if hasattr(self._cev, "is_official_hub_path"):
                self._cev.is_official_hub_path = _ms_util.is_official_hub_path
        except ImportError:
            pass
        self._sv, self._fbank = self._cev._load_sv_by_model_id(
            self._torch_device, self.offline_root.resolve(), self.sv_model_id
        )
        if hasattr(self._sv, "eval"):
            self._sv.eval()
        if self._torch_device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    def _ensure_onnx(self) -> None:
        if self._onnx is not None:
            return
        from .onnx_backend import OnnxSpeakerEmbedder

        self._onnx = OnnxSpeakerEmbedder(
            self.onnx_path, ort_threads=self.ort_threads
        )

    def _ensure(self) -> None:
        if self.backend == "onnx":
            self._ensure_onnx()
        else:
            self._ensure_torch()

    @property
    def cev(self) -> Any:
        if self.backend == "onnx":
            raise RuntimeError("onnx 后端无 childmandarin 模块；聚类已内置")
        self._ensure_torch()
        return self._cev

    def embed_array(self, wav_16k_mono: np.ndarray) -> np.ndarray:
        """16kHz 单声道波形转向量。"""
        self._ensure()
        wav = np.asarray(wav_16k_mono, dtype=np.float32).reshape(-1)
        if self.backend == "onnx":
            return self._onnx.embed_wav_numpy(wav)
        from speakerlab.utils.fileio import load_audio

        import torch

        w = torch.from_numpy(wav).unsqueeze(0)
        with torch.inference_mode():
            out = self._cev._embedding(self._sv, self._fbank, w)
        return np.asarray(out, dtype=np.float32)

    def embed_file(self, wav_path: str | Path) -> np.ndarray:
        self._ensure()
        if self.backend == "onnx":
            return self._onnx.embed_wav_path(wav_path)
        from speakerlab.utils.fileio import load_audio

        w = load_audio(str(wav_path), obj_fs=16000)
        import torch

        with torch.inference_mode():
            out = self._cev._embedding(self._sv, self._fbank, w)
        return np.asarray(out, dtype=np.float32)

    def embed_files(self, paths: Sequence[str | Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, 0), dtype=np.float32)
        rows = [self.embed_file(p) for p in paths]
        return np.stack(rows, axis=0)

    def cluster_speakers(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.0,
    ) -> ClusterBatchResult:
        self._ensure()
        embs = np.asarray(embeddings, dtype=np.float64)
        n = int(embs.shape[0])
        if n == 0:
            return ClusterBatchResult(
                np.zeros(0, dtype=np.int32),
                {"clustering_backend": "none", "skipped": True, "reason": "empty"},
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.int32),
            )
        if n == 1:
            # 只有一条时没法聚类，标成噪声，避免首条就输出编号。
            return ClusterBatchResult(
                np.zeros(1, dtype=np.int32),
                {"clustering_backend": "trivial", "n_distinct_pred_clusters": 1},
                np.ones(1, dtype=bool),
                np.array([-1], dtype=np.int32),
            )
        normed = _l2_row_normalize(embs)
        if self.backend == "onnx":
            lab, meta, noise, lab_dense = _cluster_speakers_impl(
                normed,
                int(min_cluster_size),
                int(min_samples),
                float(cluster_selection_epsilon),
            )
        else:
            out = self._cev._cluster_speakers(
                normed,
                int(min_cluster_size),
                int(min_samples),
                float(cluster_selection_epsilon),
            )
            if len(out) >= 4:
                lab, meta, noise, lab_dense = out[0], out[1], out[2], out[3]
            else:
                lab, meta, noise = out[0], out[1], out[2]
                lab_dense = np.asarray(lab, dtype=np.intp).copy()
        md = dict(meta) if isinstance(meta, dict) else {}
        md["embedding_preprocess"] = "l2_row_normalize_for_hdbscan"
        md["hdbscan_effective_geometry"] = "euclidean_on_unit_rows (cosine-related)"
        md["sv_inference_backend"] = self.backend
        return ClusterBatchResult(
            np.asarray(lab, dtype=np.int32, order="C"),
            md,
            np.asarray(noise, dtype=bool, order="C"),
            np.asarray(lab_dense, dtype=np.int32, order="C"),
        )

    def assign_files(
        self,
        wav_paths: Sequence[str | Path],
        *,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.0,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], ClusterBatchResult]:
        paths = [Path(p) for p in wav_paths]
        E = self.embed_files(paths)
        cr = self.cluster_speakers(
            E,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        recs: List[Dict[str, Any]] = []
        for i, p in enumerate(paths):
            emb = E[i]
            recs.append(
                {
                    "path": str(p.resolve()),
                    "pred_speaker_cluster": int(cr.labels[i]),
                    "hdbscan_was_noise": bool(cr.hdbscan_was_noise[i]),
                }
            )
        return E, recs, cr
