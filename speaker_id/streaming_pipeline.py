"""流式声纹入口是 ingest_audio 和 ingest_wav。每条先算嵌入，再查注册表，够像就跳过聚类。否则进缓冲做全量聚类，簇很稳时把代表写进注册表。缓冲满了会压缩融合，稳定说话人编号一旦分配就不再改。"""
from __future__ import annotations

import collections
import contextlib
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from typing import Literal as _Literal

from .embedding_store import (
    ConfidenceCache,
    EmbeddingStore,
    SCHEMA_VERSION,
    compact_embedding_store,
    compute_confidence_cache,
    is_fused_key,
    save_embedding_npz,
)
from .engine import ClusterBatchResult, SpeakerEmbedCluster, default_sv_offline_root

_REGISTRY_AUTO_LABEL = re.compile(r"^speaker_\d+$")


def _guess_registry_pinned_from_label(label: str) -> bool:
    """旧 npz 无 pinned 元数据：speaker_N 视为聚类晋升，其余视为手动注册。"""
    return not bool(_REGISTRY_AUTO_LABEL.match(str(label).strip()))


def hungarian_matched_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, Dict[int, int]]:
    """匈牙利对齐后的样本准确率。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    n_true = int(y_true.max()) + 1
    n_pred = int(y_pred.max()) + 1
    W = np.zeros((n_pred, n_true), dtype=np.int64)
    for i in range(n):
        W[y_pred[i], y_true[i]] += 1
    if n_pred == 0 or n_true == 0:
        return 0.0, {}
    cost = W.max() - W
    row_ind, col_ind = linear_sum_assignment(cost)
    pred_to_true: Dict[int, int] = {
        int(row_ind[k]): int(col_ind[k]) for k in range(len(row_ind))
    }
    correct = 0
    for i in range(n):
        t_expect = pred_to_true.get(int(y_pred[i]))
        if t_expect is not None and t_expect == int(y_true[i]):
            correct += 1
    return float(correct) / float(max(1, n)), pred_to_true


def _cluster_speakers_quiet(eng: SpeakerEmbedCluster, E: np.ndarray, **kw: Any) -> ClusterBatchResult:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return eng.cluster_speakers(E, **kw)


@dataclass
class StreamIngestResult:
    order_index: int
    wav_key: str
    client_ref: Any
    v_id: int
    hdbscan_was_noise: bool
    cosine_to_pred_cluster_centroid: float
    pred_cluster_size: int
    confidence: float
    confidence_band: str
    emit_id_min_effective: float
    id_emitted: bool
    v_id_for_use: Optional[int]
    cluster_meta: Dict[str, Any]
    n_distinct_pred_clusters: int
    prefix_hdbscan_noise_true_count: int
    #: 注册表里的稳定说话人编号，和聚类簇号无关，一行一旦分配就不再改变。关闭 enable_stable_v_id 时为 None。
    stable_v_id: Optional[int] = None
    #: 置信度够才对外暴露的稳定编号，建文件夹用这个。
    stable_v_id_for_use: Optional[int] = None
    eval: Optional[Dict[str, Any]] = None


@dataclass
class StreamingClusterPipeline:
    """每步嵌入后对当前缓冲做全量聚类。缓冲超限时按策略压缩再聚类。落盘条数受 max_store_utterances 限制。"""

    offline_root: Optional[Path] = None
    sv_model_id: str = ""
    device: str = "auto"
    #: auto 时无 torch 就用 onnx，也可设环境变量 SV_BACKEND。
    sv_backend: str = "onnx"
    onnx_path: Optional[Path] = None
    ort_threads: int = 0
    #: 至少几条语音才成一个说话人簇，默认 3。
    min_cluster_size: int = 3
    #: HDBSCAN 采样数，默认 2。
    min_samples: int = 2
    #: 单条音频最长用多少秒做嵌入，更长就截断。None 或不大于零表示不截断。
    max_ingest_audio_sec: Optional[float] = None
    #: 略大于零可合并相近簇，板端常用 0.05。
    cluster_selection_epsilon: float = 0.0
    #: 缓冲最多保留多少条，多了就压缩，压缩后仍做全量聚类不是滑窗。通常和 max_store_utterances 设一样。None 或不大于零表示不压缩。
    max_hdbscan_utterances: Optional[int] = None
    #: 压缩策略，默认只融合不删。要先删低置信再融合用 drop_then_fuse。
    buffer_compact_policy: _Literal["fuse_only", "drop_then_fuse"] = "fuse_only"
    #: 同簇两条融合前余弦至少要这么高，找不到合格对就暂停压缩。默认 0.5。
    fuse_min_intra_cluster_cosine: float = 0.5
    max_store_utterances: int = 10000
    store_high_conf_floor: float = 0.68
    emit_id_min_confidence: float = 0.45
    large_v_id_from: int = 0
    min_confidence_if_large_v_id: float = 0.68
    #: 是否给每行分配稳定编号，分配后聚类和压缩都不改。
    enable_stable_v_id: bool = True
    #: 和已有注册表质心余弦够高就复用编号，否则等新簇晋升。默认 0.78。
    stable_match_threshold: float = 0.78
    #: 注册表里两条太像就合并，要比匹配阈值更严。默认 0.9，设 1 关闭。
    stable_merge_threshold: float = 0.90
    #: 每次 ingest 最多合并几对，默认 1 防止雪崩。
    stable_merge_max_per_ingest: int = 1
    #: 簇内每条到质心余弦的最小值要够高才分配稳定编号，挡住乱炖大簇。默认 0.65，不大于零关闭。
    stable_cluster_min_intra_cos: float = 0.65
    #: 注册表文件路径，启动加载，晋升可落盘。None 表示不读写磁盘。
    speaker_registry_path: Optional[Path] = None
    #: 晋升或手动注册后是否写回磁盘。
    persist_registry: bool = True
    #: 嵌入后和注册表够像就直接返回，跳过聚类。
    enable_registry_match: bool = True
    registry_match_threshold: float = 0.85
    #: 簇特别稳时自动把代表写进注册表。
    enable_stable_cluster_promote: bool = True
    #: 晋升至少要几条同一簇。
    stable_promote_min_cluster_size: int = 3
    #: 晋升要求簇内更紧，默认 0.75。
    stable_promote_min_intra_cos: float = 0.75
    #: 聚类晋升写入注册表的无监督声纹上限；手动 register 不计入。0 表示不限制。
    max_cluster_promoted_registry: int = 30

    _eng: SpeakerEmbedCluster = field(init=False, repr=False)
    _E_buf: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _emb_n: int = field(default=0, init=False, repr=False)
    _wav_keys: List[str] = field(default_factory=list, init=False)
    _client_refs: List[Any] = field(default_factory=list, init=False)
    _y_true: List[int] = field(default_factory=list, init=False)
    _last_cr: Optional[ClusterBatchResult] = field(default=None, init=False)
    _n_total_ingested: int = field(default=0, init=False, repr=False)
    _last_compact_meta: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    #: 稳定编号注册表，下标就是编号，只增不删。
    _stable_centroids: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _stable_n_evidence: List[int] = field(default_factory=list, init=False, repr=False)
    _stable_last_step: List[int] = field(default_factory=list, init=False, repr=False)
    #: 合并后旧编号指向新编号，对外只返回当前有效编号。
    _stable_redirect: List[int] = field(default_factory=list, init=False, repr=False)
    #: 业务名，比如 alice 或 speaker_0。
    _registry_labels: List[str] = field(default_factory=list, init=False, repr=False)
    #: 每行缓冲对应的稳定编号，必须和嵌入矩阵一起维护。
    _stable_id_per_row: List[Optional[int]] = field(default_factory=list, init=False, repr=False)
    #: True=手动/磁盘注册，不参与 FIFO 淘汰也不自动合并。
    _registry_pinned: List[bool] = field(default_factory=list, init=False, repr=False)
    #: 入库顺序，越小越老；仅对 unpinned 做 FIFO。
    _registry_mint_order: List[int] = field(default_factory=list, init=False, repr=False)
    _registry_evicted: List[bool] = field(default_factory=list, init=False, repr=False)
    _registry_mint_seq: int = field(default=0, init=False, repr=False)
    #: 融合行可以追溯到哪些原始 wav。
    _fusion_lineage: Dict[str, List[str]] = field(default_factory=dict, init=False, repr=False)
    #: 置信度和簇统计的缓存，避免重复计算。
    _conf_cache: Optional[ConfidenceCache] = field(default=None, init=False, repr=False)
    #: 是否全部有标注的缓存，压缩后重算。
    _y_true_all_nonneg: bool = field(default=True, init=False, repr=False)
    def __post_init__(self) -> None:
        root = self.offline_root if self.offline_root is not None else default_sv_offline_root()
        kw: Dict[str, Any] = {
            "offline_root": root,
            "device": str(self.device),
            "backend": str(self.sv_backend),
            "ort_threads": int(self.ort_threads),
        }
        if str(self.sv_model_id).strip():
            kw["sv_model_id"] = str(self.sv_model_id).strip()
        if self.onnx_path is not None:
            kw["onnx_path"] = Path(self.onnx_path)
        self._eng = SpeakerEmbedCluster(**kw)
        self._load_speaker_registry_from_disk()

    @property
    def engine(self) -> SpeakerEmbedCluster:
        return self._eng

    @property
    def n_arrived(self) -> int:
        """缓冲当前行数，启用压缩后会下降。"""
        return int(self._emb_n)

    @property
    def n_total_ingested(self) -> int:
        """累计 ingest 次数，不受压缩影响。"""
        return int(self._n_total_ingested)

    def _append_emb_row(self, row: np.ndarray) -> None:
        x = np.asarray(row, dtype=np.float32).ravel()
        d = int(x.shape[0])
        if d == 0:
            raise ValueError("嵌入维数为 0")
        if self._E_buf is None:
            cap = max(64, 1)
            self._E_buf = np.zeros((cap, d), dtype=np.float32)
        elif int(self._E_buf.shape[1]) != d:
            raise ValueError(f"嵌入维数不一致: 已有 {self._E_buf.shape[1]}，新 {d}")
        n = int(self._emb_n)
        if n >= int(self._E_buf.shape[0]):
            new_cap = max(self._E_buf.shape[0] * 2, n + 1)
            nb = np.zeros((new_cap, d), dtype=np.float32)
            nb[:n] = self._E_buf[:n]
            self._E_buf = nb
        self._E_buf[n] = x
        self._emb_n = n + 1

    def _E_prefix(self) -> np.ndarray:
        if self._emb_n == 0 or self._E_buf is None:
            return np.zeros((0, 0), dtype=np.float32)
        return self._E_buf[: self._emb_n]

    def _rebuild_conf_cache(self, E: np.ndarray) -> ConfidenceCache:
        """按当前聚类结果重建置信度缓存。"""
        cr = self._last_cr
        if cr is None or E.shape[0] == 0:
            cache = compute_confidence_cache(
                np.zeros((0, 0), dtype=np.float64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=bool),
            )
        else:
            cache = compute_confidence_cache(
                E,
                np.asarray(cr.labels, dtype=np.int64),
                np.asarray(cr.hdbscan_was_noise, dtype=bool),
                dense_labels=np.asarray(cr.dense_labels, dtype=np.int64),
            )
        self._conf_cache = cache
        return cache

    def _y_true_recompute_all_nonneg(self) -> None:
        """压缩或外部改标注后，重算是否全部有真值标签。"""
        self._y_true_all_nonneg = all(t >= 0 for t in self._y_true)

    def _enforce_buffer_policy(self) -> None:
        """可选环境变量 SV_HARD_MAX_BUFFER 硬顶，默认不限制。"""
        hard = int(os.environ.get("SV_HARD_MAX_BUFFER", "0") or "0")
        if hard > 0 and int(self._emb_n) > hard:
            raise RuntimeError(
                f"缓冲行数 {self._emb_n} 超过 SV_HARD_MAX_BUFFER={hard}；请 pipe.reset() 或设 max_hdbscan_utterances"
            )

    def reset(self, *, keep_registry: bool = True) -> None:
        saved: Optional[
            Tuple[
                List[np.ndarray],
                List[str],
                List[int],
                List[int],
                List[int],
                List[bool],
                List[int],
                List[bool],
                int,
            ]
        ] = None
        if keep_registry and self._stable_centroids:
            saved = (
                [np.ascontiguousarray(c, dtype=np.float64) for c in self._stable_centroids],
                list(self._registry_labels),
                list(self._stable_n_evidence),
                list(self._stable_last_step),
                list(self._stable_redirect),
                list(self._registry_pinned),
                list(self._registry_mint_order),
                list(self._registry_evicted),
                int(self._registry_mint_seq),
            )
        self._E_buf = None
        self._emb_n = 0
        self._wav_keys.clear()
        self._client_refs.clear()
        self._y_true.clear()
        self._last_cr = None
        self._n_total_ingested = 0
        self._last_compact_meta = None
        self._conf_cache = None
        self._y_true_all_nonneg = True
        self._fusion_lineage = {}
        self._stable_centroids = []
        self._registry_labels = []
        self._stable_n_evidence = []
        self._stable_last_step = []
        self._stable_redirect = []
        self._stable_id_per_row = []
        self._registry_pinned = []
        self._registry_mint_order = []
        self._registry_evicted = []
        self._registry_mint_seq = 0
        if saved is not None:
            (
                centroids,
                labels,
                evidence,
                last_step,
                redirect,
                pinned,
                mint_order,
                evicted,
                mint_seq,
            ) = saved
            self._stable_centroids = centroids
            self._registry_labels = labels
            self._stable_n_evidence = evidence
            self._stable_last_step = last_step
            self._stable_redirect = redirect
            self._registry_pinned = pinned
            self._registry_mint_order = mint_order
            self._registry_evicted = evicted
            self._registry_mint_seq = mint_seq

    def registry_label(self, sid: int) -> str:
        """注册表业务名，编号无效时返回空串。"""
        sid = int(self.resolve_stable_id(int(sid)))
        if not self._registry_is_active(sid):
            return ""
        if 0 <= sid < len(self._registry_labels):
            return str(self._registry_labels[sid])
        return ""

    def _load_speaker_registry_from_disk(self) -> int:
        """从磁盘加载预注册声纹，返回导入条数。"""
        path = self.speaker_registry_path
        if path is None:
            return 0
        p = Path(path).resolve()
        if not p.is_file():
            return 0
        from .embedding_store import load_embedding_npz

        store = load_embedding_npz(p)
        n_import = 0
        existing = {str(x) for x in self._registry_labels}
        pinned_meta = store.meta.get("registry_pinned")
        order_meta = store.meta.get("registry_mint_order")
        for i in range(int(store.n_utterances)):
            label = str(store.registry_id[i]).strip()
            if not label or label in existing:
                continue
            unit = self._unit(np.asarray(store.E[i], dtype=np.float64))
            if isinstance(pinned_meta, (list, tuple)) and i < len(pinned_meta):
                pinned = bool(pinned_meta[i])
            else:
                pinned = _guess_registry_pinned_from_label(label)
            mint_ord: Optional[int] = None
            if isinstance(order_meta, (list, tuple)) and i < len(order_meta):
                mint_ord = int(order_meta[i])
            self._registry_mint_exemplar(
                unit,
                label,
                persist=False,
                pinned=pinned,
                mint_order=mint_ord,
            )
            existing.add(label)
            n_import += 1
        return n_import

    def _persist_speaker_registry_to_disk(self) -> None:
        if not bool(self.persist_registry):
            return
        path = self.speaker_registry_path
        if path is None:
            return
        p = Path(path).resolve()
        live = [
            k
            for k in range(len(self._stable_redirect))
            if self._registry_is_active(k)
        ]
        if not live:
            if p.is_file():
                p.unlink(missing_ok=True)
            return
        from .embedding_store import EmbeddingStore, SCHEMA_VERSION, save_embedding_store

        E = np.stack(
            [np.asarray(self._stable_centroids[k], dtype=np.float32).ravel() for k in live],
            axis=0,
        )
        labels = [str(self._registry_labels[k]) for k in live]
        store = EmbeddingStore(
            E=np.ascontiguousarray(E, dtype=np.float32),
            wav_rel=np.asarray(labels, dtype=object),
            true_speaker_slot=np.full(len(live), -1, dtype=np.int32),
            process_order=np.arange(len(live), dtype=np.int32),
            v_id=np.full(len(live), -1, dtype=np.int32),
            hdbscan_was_noise=np.zeros(len(live), dtype=bool),
            registry_id=np.asarray(labels, dtype=object),
            meta={
                "schema_version": SCHEMA_VERSION,
                "source": "speaker_registry",
                "n_utterances": int(len(live)),
                "registered_rows": True,
                "registry_pinned": [bool(self._registry_pinned[k]) for k in live],
                "registry_mint_order": [int(self._registry_mint_order[k]) for k in live],
            },
        )
        save_embedding_store(p, store)

    def register_exemplar(
        self,
        emb: np.ndarray,
        registry_id: str,
        *,
        persist: Optional[bool] = None,
    ) -> int:
        """写入注册表，可选落盘。"""
        label = str(registry_id).strip()
        if not label:
            raise ValueError("registry_id 不能为空")
        unit = self._unit(np.asarray(emb, dtype=np.float64))
        for sid, lab in enumerate(self._registry_labels):
            if not self._registry_is_active(sid):
                continue
            if str(lab) == label:
                self._stable_centroids[sid] = np.ascontiguousarray(unit, dtype=np.float64)
                self._stable_n_evidence[sid] = max(1, int(self._stable_n_evidence[sid]))
                if not self._registry_pinned[sid]:
                    self._registry_pinned[sid] = True
                if persist if persist is not None else self.persist_registry:
                    self._persist_speaker_registry_to_disk()
                return int(sid)
        sid = self._registry_mint_exemplar(
            unit,
            label,
            persist=persist if persist is not None else self.persist_registry,
            pinned=True,
        )
        return int(sid)

    def _registry_is_active(self, sid: int) -> bool:
        sid = int(sid)
        if sid < 0 or sid >= len(self._stable_redirect):
            return False
        if sid >= len(self._registry_evicted) or bool(self._registry_evicted[sid]):
            return False
        return int(self._stable_redirect[sid]) == sid

    def _live_cluster_promoted_sids(self) -> List[int]:
        return [
            k
            for k in range(len(self._stable_redirect))
            if self._registry_is_active(k) and not bool(self._registry_pinned[k])
        ]

    def _evict_oldest_cluster_registry(self) -> Optional[int]:
        """FIFO 淘汰最老的聚类晋升声纹（不碰 pinned）。"""
        cands = self._live_cluster_promoted_sids()
        if not cands:
            return None
        victim = min(cands, key=lambda k: (int(self._registry_mint_order[k]), int(k)))
        self._registry_evicted[victim] = True
        for j in range(len(self._stable_id_per_row)):
            sid = self._stable_id_per_row[j]
            if sid is not None and int(self.resolve_stable_id(int(sid))) == int(victim):
                self._stable_id_per_row[j] = None
        return int(victim)

    def _enforce_cluster_registry_cap(self) -> List[int]:
        cap = int(self.max_cluster_promoted_registry)
        if cap <= 0:
            return []
        evicted: List[int] = []
        while len(self._live_cluster_promoted_sids()) >= cap:
            v = self._evict_oldest_cluster_registry()
            if v is None:
                break
            evicted.append(int(v))
        return evicted

    def _registry_mint_exemplar(
        self,
        exemplar_unit: np.ndarray,
        registry_label: str,
        *,
        persist: bool = True,
        pinned: bool = False,
        mint_order: Optional[int] = None,
    ) -> int:
        if not pinned:
            self._enforce_cluster_registry_cap()
        label = str(registry_label).strip() or f"speaker_{len(self._stable_centroids)}"
        sid = len(self._stable_centroids)
        self._stable_centroids.append(np.ascontiguousarray(exemplar_unit, dtype=np.float64))
        self._registry_labels.append(label)
        self._stable_n_evidence.append(1)
        self._stable_last_step.append(int(self._n_total_ingested))
        self._stable_redirect.append(sid)
        self._registry_pinned.append(bool(pinned))
        if mint_order is None:
            mint_order = int(self._registry_mint_seq)
            self._registry_mint_seq += 1
        else:
            self._registry_mint_seq = max(int(self._registry_mint_seq), int(mint_order) + 1)
        self._registry_mint_order.append(int(mint_order))
        self._registry_evicted.append(False)
        if persist:
            self._persist_speaker_registry_to_disk()
        return int(sid)

    @property
    def fusion_lineage(self) -> Dict[str, List[str]]:
        """融合行到原始 wav 的映射，只读。"""
        return {k: list(v) for k, v in self._fusion_lineage.items()}

    @property
    def stable_registry_size(self) -> int:
        """注册表里仍有效的稳定编号个数。"""
        return sum(1 for k in range(len(self._stable_redirect)) if self._registry_is_active(k))

    @property
    def cluster_promoted_registry_size(self) -> int:
        """聚类晋升、可 FIFO 淘汰的声纹数（不含手动注册）。"""
        return len(self._live_cluster_promoted_sids())

    @property
    def stable_registry_total_minted(self) -> int:
        """历史上分配过的稳定编号总数，含已合并的。"""
        return len(self._stable_centroids)

    def resolve_stable_id(self, sid: int) -> int:
        """把编号解析到当前有效编号，越界时原样返回。"""
        sid = int(sid)
        if sid < 0 or sid >= len(self._stable_redirect):
            return sid
        while self._stable_redirect[sid] != sid:
            nxt = int(self._stable_redirect[sid])
            self._stable_redirect[sid] = int(self._stable_redirect[nxt])
            sid = nxt
        return int(sid)

    def stable_id_redirect_map(self) -> Dict[int, int]:
        """已发生过合并的旧编号到新编号的映射。"""
        out: Dict[int, int] = {}
        for k in range(len(self._stable_redirect)):
            canon = self.resolve_stable_id(k)
            if canon != k:
                out[k] = canon
        return out

    def _hdb_kw(self) -> Dict[str, Any]:
        return dict(
            min_cluster_size=int(self.min_cluster_size),
            min_samples=int(self.min_samples),
            cluster_selection_epsilon=float(self.cluster_selection_epsilon),
        )

    def _best_dense_attach_cos(
        self,
        idx: int,
        cr: ClusterBatchResult,
        En: np.ndarray,
    ) -> float:
        """噪声行到最近稠密簇质心的余弦，仅供诊断。"""
        dl = np.asarray(cr.dense_labels, dtype=np.int64)
        dense_ok = dl >= 0
        if not np.any(dense_ok):
            return -1.0
        row = np.asarray(En[idx], dtype=np.float64).ravel()
        dEn = np.asarray(En[dense_ok], dtype=np.float64)
        dlab = dl[dense_ok]
        uniq_d, inv_d = np.unique(dlab, return_inverse=True)
        kd = int(uniq_d.shape[0])
        dd = int(dEn.shape[1])
        sums_d = np.zeros((kd, dd), dtype=np.float64)
        np.add.at(sums_d, inv_d, dEn)
        cn_d = np.linalg.norm(sums_d, axis=1, keepdims=True)
        safe_d = np.where(cn_d > 1e-12, cn_d, 1.0)
        cent_d = sums_d / safe_d
        return float(np.max(np.dot(cent_d, row)))

    def _provisional_emit_v_id(
        self,
        idx: int,
        cr: ClusterBatchResult,
        En: np.ndarray,
        *,
        cache: Optional[Any] = None,
    ) -> Optional[int]:
        """噪声行若和某稠密簇够近，返回该簇的 v_id。"""
        from .embedding_store import PROVISIONAL_ATTACH_MIN_COS

        if not bool(cr.hdbscan_was_noise[idx]):
            return int(cr.labels[idx])
        dl = np.asarray(cr.dense_labels, dtype=np.int64)
        if int(dl[idx]) >= 0:
            return int(cr.labels[idx])
        row = np.asarray(En[idx], dtype=np.float64).ravel()
        dense_ok = dl >= 0
        if np.any(dense_ok):
            dEn = np.asarray(En[dense_ok], dtype=np.float64)
            dlab = dl[dense_ok]
            uniq_d, inv_d = np.unique(dlab, return_inverse=True)
            kd = int(uniq_d.shape[0])
            dd = int(dEn.shape[1])
            sums_d = np.zeros((kd, dd), dtype=np.float64)
            np.add.at(sums_d, inv_d, dEn)
            cn_d = np.linalg.norm(sums_d, axis=1, keepdims=True)
            safe_d = np.where(cn_d > 1e-12, cn_d, 1.0)
            cent_d = sums_d / safe_d
            cos_row = np.dot(cent_d, row)
            bi = int(np.argmax(cos_row))
            if float(cos_row[bi]) >= PROVISIONAL_ATTACH_MIN_COS:
                dense_cid = int(uniq_d[bi])
                for j in range(int(dl.shape[0])):
                    if not bool(cr.hdbscan_was_noise[j]) and int(dl[j]) == dense_cid:
                        return int(cr.labels[j])
        if cache is not None:
            conf_arr = np.asarray(cache.conf, dtype=np.float64)
            peer = conf_arr >= float(self.emit_id_min_confidence)
            if np.any(peer):
                cos_p = np.dot(np.asarray(En[peer], dtype=np.float64), row)
                pj = int(np.argmax(cos_p))
                if float(cos_p[pj]) >= PROVISIONAL_ATTACH_MIN_COS:
                    peers = np.flatnonzero(peer)
                    return int(cr.labels[int(peers[pj])])
        return None

    def _stable_match_best(
        self, query_unit: np.ndarray, *, tau: float
    ) -> Tuple[Optional[int], float]:
        best_sid, best_cos = None, -1.0
        if not self._stable_centroids:
            return None, -1.0
        for sid, c in enumerate(self._stable_centroids):
            if not self._registry_is_active(sid):
                continue
            cos = float(np.dot(query_unit, c))
            if cos >= tau and cos > best_cos:
                best_sid, best_cos = int(sid), cos
        return best_sid, best_cos

    def _record_registry_hit(self, sid: int) -> int:
        """命中注册表时只加证据计数，不改代表向量。"""
        sid = int(self.resolve_stable_id(int(sid)))
        if not self._registry_is_active(sid):
            return sid
        self._stable_n_evidence[sid] = int(self._stable_n_evidence[sid]) + 1
        self._stable_last_step[sid] = int(self._n_total_ingested)
        return sid

    def _try_registry_match(self, emb: np.ndarray) -> Optional[Tuple[int, float]]:
        """嵌入后查注册表，够像就返回编号和余弦。"""
        if not bool(self.enable_stable_v_id) or not bool(self.enable_registry_match):
            return None
        if not self._stable_centroids or int(self.stable_registry_size) <= 0:
            return None
        tau = float(self.registry_match_threshold)
        if tau <= 0.0:
            return None
        unit = self._unit(np.asarray(emb, dtype=np.float64))
        sid, cos = self._stable_match_best(unit, tau=tau)
        if sid is None:
            return None
        return int(sid), float(cos)

    def _ingest_registry_hit(
        self,
        emb: np.ndarray,
        *,
        wav_key: str,
        client_ref: Any,
        true_speaker_slot: Optional[int],
        sid: int,
        match_cos: float,
    ) -> StreamIngestResult:
        """注册表命中，不写缓冲，不跑聚类。"""
        canon = self._record_registry_hit(int(sid))
        self._n_total_ingested += 1
        order = int(self._n_total_ingested) - 1
        mapped_v = self._v_id_for_stable_sid(canon)
        out_id = int(mapped_v) if mapped_v is not None else int(canon)
        label = self.registry_label(canon)
        n_ev = (
            int(self._stable_n_evidence[canon])
            if 0 <= canon < len(self._stable_n_evidence)
            else 1
        )
        cm: Dict[str, Any] = {
            "cluster_skipped": True,
            "registry_match": True,
            "registry_match_cos": round(float(match_cos), 6),
            "registry_match_threshold": float(self.registry_match_threshold),
            "registry_id": label,
            "stable_fast_match": True,
            "stable_fast_match_cos": round(float(match_cos), 6),
            "emit_note": f"registry_match cos={match_cos:.3f} id={label!r}",
        }
        return StreamIngestResult(
            order_index=order,
            wav_key=str(wav_key),
            client_ref=client_ref,
            v_id=out_id,
            hdbscan_was_noise=False,
            cosine_to_pred_cluster_centroid=round(float(match_cos), 6),
            pred_cluster_size=int(n_ev),
            confidence=round(float(match_cos), 6),
            confidence_band="high" if match_cos >= 0.75 else "medium",
            emit_id_min_effective=round(float(self.registry_match_threshold), 6),
            id_emitted=True,
            v_id_for_use=out_id,
            cluster_meta=cm,
            n_distinct_pred_clusters=int(self.stable_registry_size),
            prefix_hdbscan_noise_true_count=0,
            stable_v_id=int(canon),
            stable_v_id_for_use=int(canon),
            eval=(
                {"true_speaker_slot": int(true_speaker_slot)}
                if true_speaker_slot is not None
                else None
            ),
        )

    def _v_id_for_stable_sid(self, sid: int) -> Optional[int]:
        cr = self._last_cr
        if cr is None:
            return None
        for j, s in enumerate(self._stable_id_per_row):
            if s is not None and int(self.resolve_stable_id(int(s))) == int(
                self.resolve_stable_id(int(sid))
            ):
                return int(cr.labels[j])
        return None

    def _resolve_v_id_for_emit(
        self,
        j: int,
        cr: ClusterBatchResult,
        cache: Any,
    ) -> Tuple[Optional[int], float, str]:
        """统一决定对外是否输出 v_id。"""
        from .embedding_store import (
            PROVISIONAL_ATTACH_DISC,
            PROVISIONAL_ATTACH_MIN_COS,
            STREAMING_STABLE_EMIT_COS,
        )

        v_id = np.asarray(cr.labels, dtype=np.int64)
        conf_arr = cache.conf
        cval = float(conf_arr[j])
        v_emit = int(v_id[j])
        note = ""
        if bool(cr.hdbscan_was_noise[j]):
            prov = self._provisional_emit_v_id(j, cr, cache.En, cache=cache)
            if prov is not None:
                v_emit = int(prov)
                note = "dense_attach"
            else:
                nn = self._best_dense_attach_cos(j, cr, cache.En)
                sid, scos = self._stable_match_best(
                    self._unit(cache.En[j]), tau=STREAMING_STABLE_EMIT_COS
                )
                if sid is not None:
                    mapped = self._v_id_for_stable_sid(int(sid))
                    if mapped is not None:
                        v_emit = int(mapped)
                        cval = max(cval, float(scos) * PROVISIONAL_ATTACH_DISC)
                        note = f"stable cos={scos:.3f}"
                if not note and nn >= 0:
                    note = f"nn_dense={nn:.3f}<{PROVISIONAL_ATTACH_MIN_COS}"
        emit_floor = float(self.emit_id_min_confidence)
        lv_gate = int(self.large_v_id_from)
        emit_large_min = float(self.min_confidence_if_large_v_id)
        eff_min = emit_floor
        if lv_gate > 0 and int(v_emit) >= lv_gate:
            eff_min = max(eff_min, emit_large_min)
        if cval < eff_min:
            return None, cval, note
        return int(v_emit), cval, note

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).ravel()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    def _consolidate_registry(self) -> List[Tuple[int, int, float]]:
        """注册表里太像的两条合并到证据更多的那条，保留强者代表向量。"""
        events: List[Tuple[int, int, float]] = []
        if not bool(self.enable_stable_v_id):
            return events
        tau = float(self.stable_merge_threshold)
        if tau >= 1.0 or not self._stable_centroids:
            return events
        max_per_ingest = max(0, int(self.stable_merge_max_per_ingest))
        if max_per_ingest == 0:
            return events
        while len(events) < max_per_ingest:
            live = [
                k for k in range(len(self._stable_centroids))
                if self._registry_is_active(k)
                and int(self._stable_n_evidence[k]) >= 2
                and not bool(self._registry_pinned[k])
            ]
            if len(live) < 2:
                break
            best: Optional[Tuple[int, int, float]] = None
            for ai in range(len(live)):
                a = live[ai]
                ca = self._stable_centroids[a]
                for bi in range(ai + 1, len(live)):
                    b = live[bi]
                    cb = self._stable_centroids[b]
                    cos = float(np.dot(ca, cb))
                    if cos >= tau and (best is None or cos > best[2]):
                        best = (a, b, cos)
            if best is None:
                break
            a, b, cos = best
            na = int(self._stable_n_evidence[a])
            nb = int(self._stable_n_evidence[b])
            if (nb, -b) > (na, -a):
                winner, loser = b, a
            else:
                winner, loser = a, b
            nw = int(self._stable_n_evidence[winner])
            nl = int(self._stable_n_evidence[loser])
            self._stable_n_evidence[winner] = nw + nl
            self._stable_last_step[winner] = max(
                int(self._stable_last_step[winner]), int(self._stable_last_step[loser])
            )
            self._stable_redirect[loser] = winner
            for k in range(len(self._stable_id_per_row)):
                if self._stable_id_per_row[k] == loser:
                    self._stable_id_per_row[k] = winner
            events.append((int(loser), int(winner), float(cos)))
        if events:
            self._persist_speaker_registry_to_disk()
        return events

    def _register_stable_exemplar(
        self,
        exemplar_unit: np.ndarray,
        member_indices: np.ndarray,
        *,
        registry_label: Optional[str] = None,
    ) -> int:
        """特别稳的簇把代表写进注册表，这是无监督新增编号的唯一路径。"""
        canonical_sids: List[int] = []
        for j in member_indices:
            sid = self._stable_id_per_row[int(j)]
            if sid is not None:
                canonical_sids.append(int(self.resolve_stable_id(int(sid))))

        if canonical_sids:
            sid = max(
                set(canonical_sids),
                key=lambda s: (
                    int(self._stable_n_evidence[s])
                    if 0 <= s < len(self._stable_n_evidence)
                    else 0,
                    -s,
                ),
            )
        else:
            label = registry_label or f"speaker_{len(self._stable_centroids)}"
            sid = self._registry_mint_exemplar(exemplar_unit, label, persist=False, pinned=False)

        self._stable_centroids[sid] = np.ascontiguousarray(exemplar_unit, dtype=np.float64)
        n_members = int(member_indices.size)
        self._stable_n_evidence[sid] = max(int(self._stable_n_evidence[sid]), n_members)
        self._stable_last_step[sid] = int(self._n_total_ingested)

        for j in member_indices:
            self._stable_id_per_row[int(j)] = int(sid)
        self._persist_speaker_registry_to_disk()
        return int(sid)

    def _promote_stable_clusters_to_registry(
        self,
        cache: ConfidenceCache,
        v_id_arr: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """簇特别稳时，取离质心最近的一条作为代表写进注册表。"""
        events: List[Dict[str, Any]] = []
        if not bool(self.enable_stable_v_id) or not bool(self.enable_stable_cluster_promote):
            return events
        n = int(cache.En.shape[0])
        if n == 0:
            return events

        while len(self._stable_id_per_row) < n:
            self._stable_id_per_row.append(None)

        min_sz = max(2, int(self.stable_promote_min_cluster_size))
        homog_tau = float(self.stable_promote_min_intra_cos)
        emit_floor = float(self.emit_id_min_confidence)
        inv = cache.inv
        conf_arr = cache.conf
        v_arr = np.asarray(v_id_arr, dtype=np.int64)
        K = int(cache.n_distinct_clusters)

        for ci in range(K):
            sz = int(cache.cluster_sizes[ci])
            if sz < min_sz:
                continue
            min_cos = float(cache.cluster_min_cos[ci])
            if min_cos < 0.0:
                continue
            if homog_tau > 0.0 and min_cos < homog_tau:
                continue

            members = np.flatnonzero(inv == ci)
            if members.size == 0:
                continue
            if float(np.max(conf_arr[members])) < emit_floor:
                continue

            ex_j = int(members[int(np.argmax(cache.cos_c[members]))])
            sid = self._register_stable_exemplar(cache.En[ex_j], members)
            ex_key = (
                str(self._wav_keys[ex_j])
                if ex_j < len(self._wav_keys)
                else f"row_{ex_j}"
            )
            events.append(
                {
                    "stable_cluster_promote": True,
                    "registry_id": self.registry_label(int(sid)),
                    "v_id": int(v_arr[ex_j]),
                    "stable_v_id": int(sid),
                    "cluster_size": sz,
                    "min_intra_cos": round(min_cos, 6),
                    "exemplar_index": ex_j,
                    "exemplar_wav_key": ex_key,
                    "exemplar_cos_to_centroid": round(float(cache.cos_c[ex_j]), 6),
                }
            )
        return events

    def _assign_stable_ids_for_emergent_rows(
        self,
        cache: ConfidenceCache,
        v_id_arr: np.ndarray,
    ) -> None:
        """只把缓冲行挂到已有注册表编号，不新建编号。"""
        if not bool(self.enable_stable_v_id):
            return
        n = int(cache.En.shape[0])
        if n == 0:
            return
        emit_floor = float(self.emit_id_min_confidence)
        lv_gate = int(self.large_v_id_from)
        emit_large_min = float(self.min_confidence_if_large_v_id)
        homog_tau = float(self.stable_cluster_min_intra_cos)

        v_arr = np.asarray(v_id_arr, dtype=np.int64)
        conf_arr = cache.conf
        inv = cache.inv
        cluster_sizes = cache.cluster_sizes
        cluster_centroids = cache.cluster_centroids
        cluster_min_cos = cache.cluster_min_cos

        while len(self._stable_id_per_row) < n:
            self._stable_id_per_row.append(None)

        for j in range(n):
            if self._stable_id_per_row[j] is not None:
                continue
            cval = float(conf_arr[j])
            eff_min = emit_floor
            if lv_gate > 0 and int(v_arr[j]) >= lv_gate:
                eff_min = max(eff_min, emit_large_min)
            if cval < eff_min:
                continue
            ci = int(inv[j])
            # 单点簇不能当证据，等再多几条。
            if int(cluster_sizes[ci]) < 2:
                continue
            # 质心退化也跳过。
            min_cos = float(cluster_min_cos[ci])
            if min_cos < 0.0:
                continue
            # 簇太散不能分配稳定编号。
            if homog_tau > 0.0 and min_cos < homog_tau:
                continue
            sid, _ = self._stable_match_best(
                cluster_centroids[ci], tau=float(self.stable_match_threshold)
            )
            if sid is not None:
                self._stable_id_per_row[j] = int(sid)

    def _remap_stable_ids_after_compact(
        self,
        old_wav_keys: List[str],
        old_stable_id_per_row: List[Optional[int]],
        new_wav_keys: List[str],
        new_lineage: Dict[str, List[str]],
    ) -> List[Optional[int]]:
        """压缩后按新 wav 键重建每行的稳定编号，融合行用叶子多数票。"""
        old_key_to_sid: Dict[str, Optional[int]] = {
            str(k): sid for k, sid in zip(old_wav_keys, old_stable_id_per_row)
        }
        out: List[Optional[int]] = []
        for k in new_wav_keys:
            if not is_fused_key(k):
                out.append(old_key_to_sid.get(str(k)))
                continue
            leaves = new_lineage.get(str(k), [str(k)])
            counter: "collections.Counter[int]" = collections.Counter()
            for leaf in leaves:
                sid = old_key_to_sid.get(str(leaf))
                if sid is not None:
                    counter[int(sid)] += 1
            if not counter:
                out.append(None)
                continue
            ranked = sorted(
                counter.items(),
                key=lambda kv: (
                    -kv[1],
                    -int(self._stable_n_evidence[kv[0]])
                    if kv[0] < len(self._stable_n_evidence)
                    else 0,
                    kv[0],
                ),
            )
            out.append(int(ranked[0][0]))
        return out

    def _maybe_compact_buffer(self, cr: ClusterBatchResult) -> Optional[ClusterBatchResult]:
        """缓冲超限时压缩，返回新聚类结果，否则返回 None。"""
        cap_o = self.max_hdbscan_utterances
        cap = int(cap_o) if cap_o is not None else 0
        if cap <= 0:
            return None
        n = int(self._emb_n)
        if n <= cap:
            return None
        assert self._E_buf is not None

        E_view = self._E_buf[:n]
        store = EmbeddingStore(
            E=np.ascontiguousarray(E_view).copy(),
            wav_rel=np.asarray(list(self._wav_keys), dtype=object),
            true_speaker_slot=np.asarray(self._y_true, dtype=np.int32),
            process_order=np.arange(n, dtype=np.int32),
            v_id=np.asarray(cr.labels, dtype=np.int32),
            hdbscan_was_noise=np.asarray(cr.hdbscan_was_noise, dtype=bool),
            registry_id=np.asarray([""] * n, dtype=object),
            meta={
                "schema_version": SCHEMA_VERSION,
                "fusion_lineage": {
                    k: list(v) for k, v in self._fusion_lineage.items()
                },
            },
            utterance_confidence=None,
        )
        compacted = compact_embedding_store(
            store,
            cap,
            high_conf_floor=float(self.store_high_conf_floor),
            policy=self.buffer_compact_policy,
            min_intra_cluster_cosine=float(self.fuse_min_intra_cluster_cosine),
        )

        new_n = int(compacted.E.shape[0])
        if new_n == n:
            cmeta_skip = dict(compacted.meta.get("storage_compact", {}))
            cmeta_skip["max_hdbscan_utterances"] = int(cap)
            self._last_compact_meta = cmeta_skip
            return None

        new_keys = [str(x) for x in np.asarray(compacted.wav_rel, dtype=object).tolist()]
        new_y = [int(x) for x in np.asarray(compacted.true_speaker_slot, dtype=np.int32).tolist()]

        old_key_to_ref: Dict[str, Any] = {}
        for k, ref in zip(self._wav_keys, self._client_refs):
            old_key_to_ref[str(k)] = ref
        new_refs: List[Any] = []
        for k in new_keys:
            if is_fused_key(k):
                new_refs.append(None)
            else:
                new_refs.append(old_key_to_ref.get(k))

        new_lineage = {
            str(k): [str(x) for x in v]
            for k, v in dict(compacted.meta.get("fusion_lineage", {})).items()
        }

        old_wav_keys = list(self._wav_keys)
        old_stable = list(self._stable_id_per_row)

        self._fusion_lineage = new_lineage

        d = int(self._E_buf.shape[1])
        cap_alloc = max(64, new_n)
        nb = np.zeros((cap_alloc, d), dtype=np.float32)
        nb[:new_n] = np.asarray(compacted.E, dtype=np.float32)
        self._E_buf = nb
        self._emb_n = new_n
        self._wav_keys = new_keys
        self._client_refs = new_refs
        self._y_true = new_y
        self._stable_id_per_row = self._remap_stable_ids_after_compact(
            old_wav_keys, old_stable, new_keys, new_lineage
        )

        cmeta = dict(compacted.meta.get("storage_compact", {}))
        cmeta["max_hdbscan_utterances"] = int(cap)
        self._last_compact_meta = cmeta

        self._y_true_recompute_all_nonneg()
        E_re = np.ascontiguousarray(self._E_prefix())
        cr_re = _cluster_speakers_quiet(self._eng, E_re, **self._hdb_kw())
        meta_re = dict(cr_re.meta) if isinstance(cr_re.meta, dict) else {}
        meta_re["buffer_compact"] = cmeta
        cr_out = ClusterBatchResult(
            np.asarray(cr_re.labels, dtype=np.int32, order="C"),
            meta_re,
            np.asarray(cr_re.hdbscan_was_noise, dtype=bool, order="C"),
            np.asarray(cr_re.dense_labels, dtype=np.int32, order="C"),
        )
        self._last_cr = cr_out
        self._rebuild_conf_cache(E_re)
        return cr_out

    def ingest_audio(
        self,
        wav_16k_mono: np.ndarray,
        *,
        wav_key: Optional[str] = None,
        client_ref: Any = None,
        true_speaker_slot: Optional[int] = None,
    ) -> StreamIngestResult:
        """16kHz 单声道波形做嵌入和聚类。"""
        key = wav_key if wav_key is not None else f"seg_{self._n_total_ingested}"

        cap_sec = self.max_ingest_audio_sec
        if cap_sec is not None and float(cap_sec) > 0:
            cap_n = int(float(cap_sec) * 16000)
            wav_16k_mono = np.asarray(wav_16k_mono, dtype=np.float32).reshape(-1)
            if wav_16k_mono.shape[0] > cap_n:
                wav_16k_mono = wav_16k_mono[:cap_n]

        emb = np.asarray(self._eng.embed_array(wav_16k_mono), dtype=np.float32).reshape(-1)

        fast = self._try_registry_match(emb)
        if fast is not None:
            sid, mcos = fast
            return self._ingest_registry_hit(
                emb,
                wav_key=key,
                client_ref=client_ref,
                true_speaker_slot=true_speaker_slot,
                sid=int(sid),
                match_cos=float(mcos),
            )

        self._append_emb_row(emb)
        self._wav_keys.append(key)
        self._client_refs.append(client_ref)
        slot_val = int(true_speaker_slot) if true_speaker_slot is not None else -1
        self._y_true.append(slot_val)
        if slot_val < 0:
            self._y_true_all_nonneg = False
        self._n_total_ingested += 1

        E = np.ascontiguousarray(self._E_prefix())
        n = int(E.shape[0])
        idx_new = n - 1

        cr = _cluster_speakers_quiet(self._eng, E, **self._hdb_kw())
        self._last_cr = cr
        v_id = np.asarray(cr.labels, dtype=np.int64)
        cache = self._rebuild_conf_cache(E)
        cos_c = cache.cos_c
        cl_sizes = cache.sizes
        conf_arr = cache.conf
        conf_bands = cache.bands

        promote_events: List[Dict[str, Any]] = []
        if bool(self.enable_stable_v_id):
            self._stable_id_per_row.append(None)
            promote_events = self._promote_stable_clusters_to_registry(cache, v_id)
            self._assign_stable_ids_for_emergent_rows(cache, v_id)
            new_sid = (
                self._stable_id_per_row[idx_new]
                if idx_new < len(self._stable_id_per_row)
                else None
            )
        else:
            self._stable_id_per_row.append(None)
            new_sid = None

        v_use, cval, emit_note = self._resolve_v_id_for_emit(idx_new, cr, cache)
        eff_min = float(self.emit_id_min_confidence)
        lv_gate = int(self.large_v_id_from)
        if lv_gate > 0 and v_use is not None and int(v_use) >= lv_gate:
            eff_min = max(eff_min, float(self.min_confidence_if_large_v_id))
        id_emitted = v_use is not None
        stable_v_use: Optional[int] = (
            int(new_sid) if (new_sid is not None and id_emitted) else None
        )

        eval_blob: Optional[Dict[str, Any]] = None
        all_gt = self._y_true_all_nonneg
        if all_gt:
            yt = np.asarray(self._y_true, dtype=np.int64)
            ma, pred_to_true = hungarian_matched_accuracy(yt, v_id)
            ptrue = pred_to_true.get(int(v_id[idx_new]))
            ok = bool(ptrue is not None and ptrue == int(yt[idx_new]))
            eval_blob = {
                "true_speaker_slot": (
                    int(true_speaker_slot) if true_speaker_slot is not None else int(yt[idx_new])
                ),
                "hungarian_match_ok": ok,
                "pred_cluster_to_true_slot": {str(k): v for k, v in sorted(pred_to_true.items())},
                "matched_accuracy_prefix": round(float(ma), 6),
                "adjusted_rand_index": round(float(adjusted_rand_score(yt, v_id)), 6),
                "normalized_mutual_info": round(float(normalized_mutual_info_score(yt, v_id)), 6),
            }
        elif true_speaker_slot is not None:
            eval_blob = {
                "note": (
                    "前缀中存在未标注 true_speaker_slot 的样本，跳过匈牙利与 ARI/NMI。"
                ),
                "true_speaker_slot": int(true_speaker_slot),
            }

        cm = dict(cr.meta) if isinstance(cr.meta, dict) else {}
        if emit_note:
            cm["emit_note"] = emit_note
        if promote_events:
            cm["stable_cluster_promote_events"] = promote_events
            for ev in promote_events:
                if int(v_id[idx_new]) == int(ev["v_id"]):
                    cm["stable_cluster_promote"] = ev
                    break

        result = StreamIngestResult(
            order_index=idx_new,
            wav_key=key,
            client_ref=client_ref,
            v_id=int(v_id[idx_new]),
            hdbscan_was_noise=bool(cr.hdbscan_was_noise[idx_new]),
            cosine_to_pred_cluster_centroid=round(float(cos_c[idx_new]), 6),
            pred_cluster_size=int(cl_sizes[idx_new]),
            confidence=round(cval, 6),
            confidence_band=conf_bands[idx_new],
            emit_id_min_effective=round(float(eff_min), 6),
            id_emitted=id_emitted,
            v_id_for_use=v_use,
            cluster_meta=cm,
            n_distinct_pred_clusters=int(cache.n_distinct_clusters),
            prefix_hdbscan_noise_true_count=int(cache.n_noise_true),
            stable_v_id=int(new_sid) if new_sid is not None else None,
            stable_v_id_for_use=stable_v_use,
            eval=eval_blob,
        )

        self._maybe_compact_buffer(cr)
        self._enforce_buffer_policy()

        merge_events = self._consolidate_registry()
        if merge_events:
            new_sid_resolved = self.resolve_stable_id(int(new_sid)) if new_sid is not None else None
            if new_sid_resolved is not None and new_sid_resolved != new_sid:
                result.stable_v_id = int(new_sid_resolved)
                if result.stable_v_id_for_use is not None:
                    result.stable_v_id_for_use = int(new_sid_resolved)
            cm_with_merge = dict(result.cluster_meta) if isinstance(result.cluster_meta, dict) else {}
            cm_with_merge["stable_id_merge_events"] = [
                {"loser": l, "winner": w, "cos": round(c, 6)} for l, w, c in merge_events
            ]
            result.cluster_meta = cm_with_merge

        return result

    def ingest_wav(
        self,
        wav_path: Union[str, Path],
        *,
        wav_key: Optional[str] = None,
        client_ref: Any = None,
        true_speaker_slot: Optional[int] = None,
    ) -> StreamIngestResult:
        path = Path(wav_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        from .audio_io import load_wav_16k_mono

        key = wav_key if wav_key is not None else str(path.resolve())
        wav = load_wav_16k_mono(path)
        return self.ingest_audio(
            wav,
            wav_key=key,
            client_ref=client_ref,
            true_speaker_slot=true_speaker_slot,
        )

    def prefix_v_id_for_use_all(self) -> List[Optional[int]]:
        """当前缓冲每条对外可用的 v_id，规则和 ingest 一样。"""
        cr = self._last_cr
        cache = self._conf_cache
        if cr is None or self._emb_n == 0 or cache is None:
            return []
        n = int(cache.cos_c.shape[0])
        out: List[Optional[int]] = []
        for j in range(n):
            vu, _, _ = self._resolve_v_id_for_emit(j, cr, cache)
            out.append(vu)
        return out

    def prefix_stable_v_id_for_use_all(self) -> List[Optional[int]]:
        """同上，但返回稳定编号。关闭稳定编号时退回 v_id。"""
        if not bool(self.enable_stable_v_id):
            return self.prefix_v_id_for_use_all()
        cr = self._last_cr
        cache = self._conf_cache
        if cr is None or self._emb_n == 0 or cache is None:
            return []
        n = int(cache.cos_c.shape[0])
        v_id = np.asarray(cr.labels, dtype=np.int64)
        conf_arr = cache.conf
        emit_floor = float(self.emit_id_min_confidence)
        lv_gate = int(self.large_v_id_from)
        emit_large_min = float(self.min_confidence_if_large_v_id)
        out: List[Optional[int]] = []
        for j in range(n):
            sid = (
                self._stable_id_per_row[j]
                if j < len(self._stable_id_per_row)
                else None
            )
            cval = float(conf_arr[j])
            eff_min = emit_floor
            if lv_gate > 0 and int(v_id[j]) >= lv_gate:
                eff_min = max(eff_min, emit_large_min)
            if sid is not None and bool(cval >= eff_min):
                sid = int(self.resolve_stable_id(int(sid)))
                out.append(int(sid) if self._registry_is_active(sid) else None)
            else:
                out.append(None)
        return out

    def prefix_stable_v_id_for_use_by_key(self) -> Dict[str, Optional[int]]:
        """按 wav 键索引的稳定编号字典，含融合行。"""
        out: Dict[str, Optional[int]] = {}
        vals = self.prefix_stable_v_id_for_use_all()
        for j, k in enumerate(self._wav_keys[: self._emb_n]):
            out[str(k)] = vals[j] if j < len(vals) else None
        return out

    def prefix_v_id_for_use_by_key(self) -> Dict[str, Optional[int]]:
        """按 wav 键索引的 v_id 字典。"""
        out: Dict[str, Optional[int]] = {}
        vals = self.prefix_v_id_for_use_all()
        for j, k in enumerate(self._wav_keys[: self._emb_n]):
            out[str(k)] = vals[j] if j < len(vals) else None
        return out

    def buffer_wav_keys(self) -> List[str]:
        """当前缓冲所有 wav 键的拷贝，含融合行。"""
        return [str(k) for k in self._wav_keys[: self._emb_n]]

    def prefix_ingest_results(self) -> List[StreamIngestResult]:
        """按 ingest 同样规则重算当前缓冲每条的结果，供清单等用途。"""
        cr = self._last_cr
        cache = self._conf_cache
        if cr is None or self._emb_n == 0 or cache is None:
            return []
        n = int(cache.cos_c.shape[0])
        v_id = np.asarray(cr.labels, dtype=np.int64)
        noise_b = np.asarray(cr.hdbscan_was_noise, dtype=bool)
        cos_c = cache.cos_c
        cl_sizes = cache.sizes
        conf_arr = cache.conf
        conf_bands = cache.bands
        emit_floor = float(self.emit_id_min_confidence)
        lv_gate = int(self.large_v_id_from)
        emit_large_min = float(self.min_confidence_if_large_v_id)
        cm = dict(cr.meta) if isinstance(cr.meta, dict) else {}
        nd = int(cache.n_distinct_clusters)
        pn = int(cache.n_noise_true)

        all_gt = self._y_true_all_nonneg
        ma: Optional[float] = None
        pred_to_true: Optional[Dict[int, int]] = None
        ari: Optional[float] = None
        nmi: Optional[float] = None
        yt: Optional[np.ndarray] = None
        if all_gt and n > 0:
            yt = np.asarray(self._y_true, dtype=np.int64)
            ma_f, pred_to_true = hungarian_matched_accuracy(yt, v_id)
            ma = float(ma_f)
            ari = round(float(adjusted_rand_score(yt, v_id)), 6)
            nmi = round(float(normalized_mutual_info_score(yt, v_id)), 6)

        rows: List[StreamIngestResult] = []
        for j in range(n):
            cval = float(conf_arr[j])
            eff_min = emit_floor
            if lv_gate > 0 and int(v_id[j]) >= lv_gate:
                eff_min = max(eff_min, emit_large_min)
            id_emitted = bool(cval >= eff_min)
            v_use = int(v_id[j]) if id_emitted else None

            eval_blob: Optional[Dict[str, Any]] = None
            if all_gt and yt is not None and pred_to_true is not None and ma is not None:
                ptrue = pred_to_true.get(int(v_id[j]))
                ok = bool(ptrue is not None and ptrue == int(yt[j]))
                eval_blob = {
                    "true_speaker_slot": int(yt[j]),
                    "hungarian_match_ok": ok,
                    "pred_cluster_to_true_slot": {
                        str(k): v for k, v in sorted(pred_to_true.items())
                    },
                    "matched_accuracy_prefix": round(float(ma), 6),
                    "adjusted_rand_index": ari,
                    "normalized_mutual_info": nmi,
                }

            sid = (
                self._stable_id_per_row[j]
                if j < len(self._stable_id_per_row)
                else None
            )
            stable_v_use = int(sid) if (sid is not None and id_emitted) else None

            rows.append(
                StreamIngestResult(
                    order_index=j,
                    wav_key=self._wav_keys[j],
                    client_ref=self._client_refs[j],
                    v_id=int(v_id[j]),
                    hdbscan_was_noise=bool(noise_b[j]),
                    cosine_to_pred_cluster_centroid=round(float(cos_c[j]), 6),
                    pred_cluster_size=int(cl_sizes[j]),
                    confidence=round(cval, 6),
                    confidence_band=conf_bands[j],
                    emit_id_min_effective=round(float(eff_min), 6),
                    id_emitted=id_emitted,
                    v_id_for_use=v_use,
                    cluster_meta=cm,
                    n_distinct_pred_clusters=nd,
                    prefix_hdbscan_noise_true_count=pn,
                    stable_v_id=int(sid) if sid is not None else None,
                    stable_v_id_for_use=stable_v_use,
                    eval=eval_blob,
                )
            )
        return rows

    def embeddings_matrix(self) -> np.ndarray:
        return np.ascontiguousarray(self._E_prefix())

    def last_cluster_result(self) -> Optional[ClusterBatchResult]:
        return self._last_cr

    def persist_embedding_store(self, npz_path: Union[str, Path]) -> Path:
        cr = self._last_cr
        if cr is None or self._emb_n == 0:
            raise RuntimeError("尚无语音到达，无法持久化")

        E = np.ascontiguousarray(self._E_prefix()).copy()
        n = int(E.shape[0])
        v_id = np.asarray(cr.labels, dtype=np.int64)
        y_true = np.asarray(self._y_true, dtype=np.int32)
        process_order = np.arange(n, dtype=np.int32)

        store_meta: Dict[str, Any] = {
            "eval_mode": "streaming_pipeline",
            "embed_schedule": (
                "sequential: one embed per step then prefix HDBSCAN "
                "(optional max_hdbscan_utterances tail window)"
            ),
            "sv_model_id": self._eng.sv_model_id,
            "sv_offline_root": str(self._eng.offline_root.resolve()),
            "embedding_dim": int(E.shape[1]),
            "n_utterances": n,
            "hdbscan_params": self._hdb_kw(),
        }
        if self.max_hdbscan_utterances is not None and int(self.max_hdbscan_utterances) > 0:
            store_meta["max_hdbscan_utterances"] = int(self.max_hdbscan_utterances)

        cap = int(self.max_store_utterances)
        save_kw: Dict[str, Any] = {
            "E": E,
            "wav_rel": list(self._wav_keys),
            "true_speaker_slot": y_true,
            "process_order": process_order,
            "v_id": v_id,
            "hdbscan_was_noise": np.asarray(cr.hdbscan_was_noise, dtype=bool),
            "registry_id": [""] * n,
            "meta": store_meta,
        }
        if cap > 0:
            save_kw["max_utterances"] = cap
            save_kw["high_conf_floor"] = float(self.store_high_conf_floor)

        return save_embedding_npz(Path(npz_path), **save_kw)
