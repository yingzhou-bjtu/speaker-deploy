"""声纹 npz 存嵌入、路径、簇号和噪声标记，可选业务名和置信度。相似度直接用嵌入余弦。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA_VERSION = 2

# 融合行的 wav 键前缀，旧竖线拼接格式已废弃。
FUSED_KEY_PREFIX = "__fused__:"


def is_fused_key(k: str) -> bool:
    """是否是融合行键。"""
    return isinstance(k, str) and k.startswith(FUSED_KEY_PREFIX)


def make_fused_key(leaves: Sequence[str]) -> str:
    """根据原始 wav 列表生成定长融合键。"""
    leaves_s = sorted(set(str(x) for x in leaves))
    h = hashlib.sha256("|".join(leaves_s).encode("utf-8")).hexdigest()[:12]
    return f"{FUSED_KEY_PREFIX}{len(leaves_s)}:{h}"


def resolve_fusion_leaves(
    key: str, lineage: Dict[str, List[str]]
) -> List[str]:
    """有血缘就展开成叶子列表，否则当作单条返回。"""
    if key in lineage:
        return list(lineage[key])
    return [str(key)]

NPZ_KEYS = (
    "E",
    "wav_rel",
    "true_speaker_slot",
    "process_order",
    "v_id",
    "hdbscan_was_noise",
)
# registry_id 是每行的业务名，空串表示纯聚类行。


@dataclass
class ConfidenceCache:
    """置信度缓存，含每行余弦、簇规模和同质性等派生量。"""

    cos_c: np.ndarray
    sizes: np.ndarray
    conf: np.ndarray
    bands: List[str]
    En: np.ndarray
    inv: np.ndarray
    uniq_cluster_ids: np.ndarray
    cluster_sizes: np.ndarray
    cluster_centroids: np.ndarray
    cluster_min_cos: np.ndarray
    n_distinct_clusters: int
    n_noise_true: int


_EMPTY_CACHE = ConfidenceCache(
    cos_c=np.zeros(0, dtype=np.float64),
    sizes=np.zeros(0, dtype=np.int32),
    conf=np.zeros(0, dtype=np.float64),
    bands=[],
    En=np.zeros((0, 0), dtype=np.float64),
    inv=np.zeros(0, dtype=np.int64),
    uniq_cluster_ids=np.zeros(0, dtype=np.int64),
    cluster_sizes=np.zeros(0, dtype=np.int64),
    cluster_centroids=np.zeros((0, 0), dtype=np.float64),
    cluster_min_cos=np.zeros(0, dtype=np.float64),
    n_distinct_clusters=0,
    n_noise_true=0,
)

# 噪声点若和稠密簇够像，可按该簇规模计置信度。
PROVISIONAL_ATTACH_MIN_COS = 0.50
PROVISIONAL_ATTACH_DISC = 0.88
# 噪声行和注册表够像时也可对外输出编号。
STREAMING_STABLE_EMIT_COS = 0.68


def compute_confidence_cache(
    E: np.ndarray,
    v_id: np.ndarray,
    hdbscan_noise: np.ndarray,
    dense_labels: Optional[np.ndarray] = None,
) -> ConfidenceCache:
    """算每行到簇质心的余弦，再按噪声和簇规模打折得到置信度。"""
    E64 = np.asarray(E, dtype=np.float64)
    n = int(E64.shape[0])
    if n == 0:
        return _EMPTY_CACHE

    row_n = np.linalg.norm(E64, axis=1, keepdims=True)
    row_n = np.maximum(row_n, 1e-12)
    En = E64 / row_n

    v = np.asarray(v_id, dtype=np.int64)
    uniq, inv = np.unique(v, return_inverse=True)
    K = int(uniq.shape[0])
    d = int(En.shape[1])

    cluster_sizes = np.bincount(inv, minlength=K).astype(np.int64)
    sizes = cluster_sizes[inv]

    sums = np.zeros((K, d), dtype=np.float64)
    np.add.at(sums, inv, En)
    centroid_norms = np.linalg.norm(sums, axis=1, keepdims=True)
    safe = np.where(centroid_norms > 1e-12, centroid_norms, 1.0)
    cluster_centroids = sums / safe
    bad_cluster = (centroid_norms.flatten() <= 1e-12)

    cos_c = np.einsum("ij,ij->i", En, cluster_centroids[inv])
    cos_c = np.where(bad_cluster[inv], 0.0, cos_c)
    cos_c = np.clip(cos_c, 0.0, 1.0)

    cluster_min_cos = np.full(K, np.inf, dtype=np.float64)
    np.minimum.at(cluster_min_cos, inv, cos_c)
    cluster_min_cos[bad_cluster] = -1.0
    cluster_min_cos = np.where(np.isinf(cluster_min_cos), -1.0, cluster_min_cos)

    noise_mask = np.asarray(hdbscan_noise, dtype=bool)
    sz_f = sizes.astype(np.float64)
    size_disc = np.where(sz_f >= 4, 1.0, np.where(sz_f >= 2, 0.88, 0.52))
    noise_disc = np.where(noise_mask, 0.35, 1.0)
    conf = np.clip(cos_c * noise_disc * size_disc, 0.0, 1.0)

    # 噪声点挂靠稠密簇，避免单点伪簇置信度过低。
    if dense_labels is not None:
        dl = np.asarray(dense_labels, dtype=np.int64)
        if dl.shape[0] == n:
            dense_ok = dl >= 0
            if np.any(dense_ok) and np.any(noise_mask):
                dEn = En[dense_ok]
                dlab = dl[dense_ok]
                uniq_d, inv_d = np.unique(dlab, return_inverse=True)
                kd = int(uniq_d.shape[0])
                dd = int(En.shape[1])
                sums_d = np.zeros((kd, dd), dtype=np.float64)
                np.add.at(sums_d, inv_d, dEn)
                cn_d = np.linalg.norm(sums_d, axis=1, keepdims=True)
                safe_d = np.where(cn_d > 1e-12, cn_d, 1.0)
                cent_d = sums_d / safe_d
                sz_d = np.bincount(inv_d, minlength=kd).astype(np.int64)
                sz_disc_d = np.where(
                    sz_d.astype(np.float64) >= 4,
                    1.0,
                    np.where(sz_d.astype(np.float64) >= 2, 0.88, 0.52),
                )
                for i in np.flatnonzero(noise_mask):
                    cos_row = np.dot(cent_d, En[i])
                    bi = int(np.argmax(cos_row))
                    cos_best = float(np.clip(cos_row[bi], 0.0, 1.0))
                    if cos_best >= PROVISIONAL_ATTACH_MIN_COS:
                        conf[i] = float(
                            np.clip(
                                cos_best * sz_disc_d[bi] * PROVISIONAL_ATTACH_DISC,
                                0.0,
                                1.0,
                            )
                        )

    bands_arr = np.where(conf >= 0.68, "high", np.where(conf >= 0.42, "medium", "low"))

    return ConfidenceCache(
        cos_c=cos_c,
        sizes=sizes.astype(np.int32),
        conf=conf,
        bands=bands_arr.tolist(),
        En=En,
        inv=inv.astype(np.int64),
        uniq_cluster_ids=uniq.astype(np.int64),
        cluster_sizes=cluster_sizes,
        cluster_centroids=cluster_centroids,
        cluster_min_cos=cluster_min_cos,
        n_distinct_clusters=K,
        n_noise_true=int(noise_mask.sum()),
    )


def compute_utterance_confidence(
    E: np.ndarray,
    v_id: np.ndarray,
    hdbscan_noise: np.ndarray,
) -> np.ndarray:
    """返回每行置信度数组。"""
    return compute_confidence_cache(E, v_id, hdbscan_noise).conf


def utterance_confidence_fields(
    E: np.ndarray,
    v_id: np.ndarray,
    hdbscan_noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """返回余弦、规模、置信度和档位。"""
    c = compute_confidence_cache(E, v_id, hdbscan_noise)
    return c.cos_c, c.sizes, c.conf, c.bands


@dataclass
class EmbeddingStore:

    E: np.ndarray
    wav_rel: np.ndarray
    true_speaker_slot: np.ndarray
    process_order: np.ndarray
    v_id: np.ndarray
    hdbscan_was_noise: np.ndarray
    registry_id: np.ndarray
    meta: Dict[str, Any]
    utterance_confidence: Optional[np.ndarray] = None

    @property
    def embedding_dim(self) -> int:
        return int(self.E.shape[1])

    @property
    def n_utterances(self) -> int:
        return int(self.E.shape[0])


def _fuse_unit_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两条单位向量先归一化再相加再归一化，得到融合方向。"""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na < 1e-12 and nb < 1e-12:
        return np.asarray(a, dtype=np.float32)
    if na < 1e-12:
        return (b64 / nb).astype(np.float32)
    if nb < 1e-12:
        return (a64 / na).astype(np.float32)
    s = a64 / na + b64 / nb
    ns = float(np.linalg.norm(s))
    if ns < 1e-12:
        # 两向量反向相消时退回其中一条。
        return (a64 / na).astype(np.float32)
    return (s / ns).astype(np.float32)


def _pick_merge_pair(
    conf: np.ndarray,
    v_id: np.ndarray,
    *,
    E: Optional[np.ndarray] = None,
    min_intra_cluster_cosine: float = 0.0,
    require_same_cluster: bool = False,
) -> Optional[Tuple[int, int]]:
    """挑要融合的一对，同簇优先，可要求簇内余弦够高。"""
    n = int(conf.shape[0])
    if n < 2:
        raise ValueError("至少需要 2 条才能融合")

    use_cos_filter = float(min_intra_cluster_cosine) > 0.0
    if use_cos_filter and E is None:
        raise ValueError("min_intra_cluster_cosine > 0 时必须提供 E")
    En: Optional[np.ndarray] = None
    if use_cos_filter:
        En64 = np.asarray(E, dtype=np.float64)
        rn = np.linalg.norm(En64, axis=1, keepdims=True)
        En = En64 / np.maximum(rn, 1e-12)

    best: Optional[Tuple[int, int]] = None
    best_key: Tuple[float, float, float] = (-1.0, -1.0, -1.0)

    for v in np.unique(v_id):
        idx = np.flatnonzero(v_id == v)
        if idx.size < 2:
            continue
        if use_cos_filter:
            assert En is not None
            for a in range(idx.size):
                for b in range(a + 1, idx.size):
                    i, j = int(idx[a]), int(idx[b])
                    cij = float(np.dot(En[i], En[j]))
                    if cij < float(min_intra_cluster_cosine):
                        continue
                    key = (
                        cij,
                        float(min(conf[i], conf[j])),
                        float(conf[i] + conf[j]),
                    )
                    if key > best_key:
                        best_key = key
                        best = (i, j)
        else:
            c2 = conf[idx]
            top2 = idx[np.argsort(-c2)[:2]]
            i, j = int(top2[0]), int(top2[1])
            key = (
                1.0,
                float(min(conf[i], conf[j])),
                float(conf[i] + conf[j]),
            )
            if key > best_key:
                best_key = key
                best = (i, j)

    if best is not None:
        return best
    if require_same_cluster or use_cos_filter:
        return None
    order = np.argsort(-conf)
    return int(order[0]), int(order[1])


def _normalize_registry_rows(n: int, registry_id: Optional[Sequence[str]]) -> List[str]:
    if registry_id is None:
        return [""] * int(n)
    reg = [str(x) for x in registry_id]
    if len(reg) != int(n):
        raise ValueError("registry_id 长度须与嵌入行数一致")
    return reg


def compact_embedding_store(
    store: EmbeddingStore,
    cap: int,
    *,
    high_conf_floor: float = 0.68,
    policy: Literal["drop_then_fuse", "fuse_only"] = "drop_then_fuse",
    min_intra_cluster_cosine: float = 0.0,
) -> EmbeddingStore:
    """把条数压到上限。可先删低置信再融合，或只融合不删。"""
    if cap <= 0:
        raise ValueError("cap 须为正整数")
    n0 = int(store.E.shape[0])
    if n0 <= cap:
        return store

    E = np.asarray(store.E, dtype=np.float32).copy()
    wav_rel = [str(x) for x in np.asarray(store.wav_rel, dtype=object).tolist()]
    true_speaker_slot = np.asarray(store.true_speaker_slot, dtype=np.int32).copy()
    process_order = np.asarray(store.process_order, dtype=np.int32).copy()
    v_id = np.asarray(store.v_id, dtype=np.int32).copy()
    hdbscan_was_noise = np.asarray(store.hdbscan_was_noise, dtype=bool).copy()
    reg_ids = [
        str(x).strip()
        for x in np.asarray(store.registry_id, dtype=object).tolist()
    ]

    actions: List[Dict[str, Any]] = []
    lineage: Dict[str, List[str]] = {
        str(k): [str(x) for x in v]
        for k, v in dict(store.meta.get("fusion_lineage", {})).items()
    }

    def renumber_process_order() -> None:
        m = int(process_order.min()) if process_order.size else 0
        process_order[:] = np.arange(process_order.size, dtype=np.int32) + m

    fuse_only = policy == "fuse_only"
    skipped_compact: Optional[Dict[str, Any]] = None

    while int(E.shape[0]) > cap:
        conf = compute_utterance_confidence(E, v_id, hdbscan_was_noise)

        if not fuse_only:
            noise_idx = np.flatnonzero(hdbscan_was_noise)
            if noise_idx.size > 0:
                ni = int(noise_idx[np.argmin(conf[noise_idx])])
                keep = np.ones(E.shape[0], dtype=bool)
                keep[ni] = False
                E = E[keep]
                wav_rel = [wav_rel[i] for i in range(len(wav_rel)) if keep[i]]
                reg_ids = [reg_ids[i] for i in range(len(reg_ids)) if keep[i]]
                true_speaker_slot = true_speaker_slot[keep]
                process_order = process_order[keep]
                v_id = v_id[keep]
                hdbscan_was_noise = hdbscan_was_noise[keep]
                actions.append({"op": "drop_noise", "removed_index": ni})
                renumber_process_order()
                continue

            if float(np.min(conf)) < float(high_conf_floor):
                li = int(np.argmin(conf))
                keep = np.ones(E.shape[0], dtype=bool)
                keep[li] = False
                E = E[keep]
                wav_rel = [wav_rel[i] for i in range(len(wav_rel)) if keep[i]]
                reg_ids = [reg_ids[i] for i in range(len(reg_ids)) if keep[i]]
                true_speaker_slot = true_speaker_slot[keep]
                process_order = process_order[keep]
                v_id = v_id[keep]
                hdbscan_was_noise = hdbscan_was_noise[keep]
                actions.append({"op": "drop_lowest_confidence", "removed_index": li, "confidence": float(conf[li])})
                renumber_process_order()
                continue

        pair = _pick_merge_pair(
            conf,
            v_id,
            E=E if (fuse_only or float(min_intra_cluster_cosine) > 0.0) else None,
            min_intra_cluster_cosine=float(min_intra_cluster_cosine),
            require_same_cluster=fuse_only,
        )
        if pair is None:
            skipped_compact = {
                "op": "skip_compact_no_qualified_pair",
                "policy": policy,
                "min_intra_cluster_cosine": float(min_intra_cluster_cosine),
                "n_at_skip": int(E.shape[0]),
            }
            actions.append(skipped_compact)
            break
        i, j = pair
        if i > j:
            i, j = j, i
        fused_e = _fuse_unit_sum(E[i], E[j])
        leaves_i = resolve_fusion_leaves(wav_rel[i], lineage)
        leaves_j = resolve_fusion_leaves(wav_rel[j], lineage)
        new_leaves = sorted(set(leaves_i) | set(leaves_j))
        wr = make_fused_key(new_leaves)
        if wav_rel[i] in lineage:
            lineage.pop(wav_rel[i], None)
        if wav_rel[j] in lineage:
            lineage.pop(wav_rel[j], None)
        lineage[wr] = new_leaves
        ri, rj = reg_ids[i], reg_ids[j]
        if ri and rj:
            r_fused = ri if ri == rj else f"{ri}|{rj}"
        elif ri:
            r_fused = ri
        elif rj:
            r_fused = rj
        else:
            r_fused = ""
        if int(true_speaker_slot[i]) == int(true_speaker_slot[j]):
            ts = int(true_speaker_slot[i])
        else:
            ts = -1
        vid = int(v_id[i]) if int(v_id[i]) == int(v_id[j]) else min(int(v_id[i]), int(v_id[j]))
        keep = np.ones(E.shape[0], dtype=bool)
        keep[i] = False
        keep[j] = False
        E_rest = E[keep]
        E = np.vstack([E_rest, fused_e.reshape(1, -1)])
        new_list = [wav_rel[k] for k in range(len(wav_rel)) if keep[k]]
        new_list.append(wr)
        wav_rel = new_list
        new_reg = [reg_ids[k] for k in range(len(reg_ids)) if keep[k]]
        new_reg.append(r_fused)
        reg_ids = new_reg
        true_speaker_slot = np.concatenate([true_speaker_slot[keep], np.array([ts], dtype=np.int32)])
        process_order = np.concatenate([process_order[keep], np.array([int(process_order.max() + 1)], dtype=np.int32)])
        v_id = np.concatenate([v_id[keep], np.array([vid], dtype=np.int32)])
        hdbscan_was_noise = np.concatenate([hdbscan_was_noise[keep], np.array([False], dtype=bool)])
        actions.append({"op": "fuse_high_confidence", "merged_indices": [i, j], "v_id": vid})
        renumber_process_order()

    meta = dict(store.meta)
    meta["schema_version"] = SCHEMA_VERSION
    na = int(E.shape[0])
    meta["n_utterances"] = na
    meta["storage_compact"] = {
        "target_cap": cap,
        "n_before": n0,
        "n_after": na,
        "steps": len(actions),
        "policy": policy,
        "high_conf_floor": float(high_conf_floor) if not fuse_only else None,
        "min_intra_cluster_cosine": float(min_intra_cluster_cosine),
        "skipped_compact": skipped_compact,
        "actions_tail": actions[-20:],
    }
    meta["fusion_lineage"] = lineage
    uc = compute_utterance_confidence(E, v_id, hdbscan_was_noise)
    return EmbeddingStore(
        E=E,
        wav_rel=np.asarray(wav_rel, dtype=object),
        true_speaker_slot=true_speaker_slot,
        process_order=process_order,
        v_id=v_id,
        hdbscan_was_noise=hdbscan_was_noise,
        registry_id=np.asarray(reg_ids, dtype=object),
        meta=meta,
        utterance_confidence=np.asarray(uc, dtype=np.float32),
    )


def save_embedding_npz(
    path: Path,
    *,
    E: np.ndarray,
    wav_rel: List[str],
    true_speaker_slot: np.ndarray,
    process_order: np.ndarray,
    v_id: np.ndarray,
    hdbscan_was_noise: np.ndarray,
    registry_id: Optional[Sequence[str]] = None,
    meta: Dict[str, Any],
    max_utterances: Optional[int] = None,
    high_conf_floor: float = 0.68,
) -> Path:
    """写 npz，超上限先压缩。"""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    E_use = np.asarray(E, dtype=np.float32)
    n_rows = int(E_use.shape[0])
    wr = list(wav_rel)
    reg = _normalize_registry_rows(n_rows, registry_id)
    ts = np.asarray(true_speaker_slot, dtype=np.int32)
    po = np.asarray(process_order, dtype=np.int32)
    vid = np.asarray(v_id, dtype=np.int32)
    noise = np.asarray(hdbscan_was_noise, dtype=bool)
    meta_use = dict(meta)
    uc_out: Optional[np.ndarray] = None

    mx = max_utterances
    if mx is not None and int(E_use.shape[0]) > int(mx):
        st = EmbeddingStore(
            E=E_use,
            wav_rel=np.asarray(wr, dtype=object),
            true_speaker_slot=ts,
            process_order=po,
            v_id=vid,
            hdbscan_was_noise=noise,
            registry_id=np.asarray(reg, dtype=object),
            meta=meta_use,
        )
        st = compact_embedding_store(st, int(mx), high_conf_floor=high_conf_floor)
        E_use = st.E
        wr = [str(x) for x in np.asarray(st.wav_rel, dtype=object).tolist()]
        reg = [str(x) for x in np.asarray(st.registry_id, dtype=object).tolist()]
        ts = st.true_speaker_slot
        po = st.process_order
        vid = st.v_id
        noise = st.hdbscan_was_noise
        meta_use = st.meta
        uc_out = st.utterance_confidence

    blob = dict(meta_use)
    blob["schema_version"] = SCHEMA_VERSION
    meta_json = json.dumps(blob, ensure_ascii=False)
    payload = dict(
        E=E_use,
        wav_rel=np.asarray(wr, dtype=object),
        true_speaker_slot=ts,
        process_order=po,
        v_id=vid,
        hdbscan_was_noise=noise,
        registry_id=np.asarray(reg, dtype=object),
        meta_json=np.array(meta_json, dtype=object),
    )
    if uc_out is not None:
        payload["utterance_confidence"] = np.asarray(uc_out, dtype=np.float32)
    np.savez_compressed(str(path), **payload)
    return path


def load_embedding_npz(path: str | Path) -> EmbeddingStore:
    p = Path(path).resolve()
    z = np.load(str(p), allow_pickle=True)
    mj = z["meta_json"]
    if isinstance(mj, np.ndarray):
        meta_raw = mj.item() if mj.ndim == 0 else str(mj[0])
    else:
        meta_raw = str(mj)
    meta = json.loads(meta_raw)
    uc = None
    if "utterance_confidence" in z.files:
        uc = np.asarray(z["utterance_confidence"], dtype=np.float32)
    ne = int(np.asarray(z["E"]).shape[0])
    if "registry_id" in z.files:
        rid_flat = np.asarray(z["registry_id"], dtype=object).reshape(-1)
        if int(rid_flat.size) < ne:
            pad = ne - int(rid_flat.size)
            rid = np.array(list(rid_flat.tolist()) + [""] * pad, dtype=object)
        elif int(rid_flat.size) > ne:
            rid = np.asarray(rid_flat[:ne], dtype=object).reshape(ne)
        else:
            rid = np.asarray(rid_flat, dtype=object).reshape(ne)
    else:
        rid = np.array([""] * ne, dtype=object)
    return EmbeddingStore(
        E=np.asarray(z["E"], dtype=np.float32),
        wav_rel=np.asarray(z["wav_rel"], dtype=object),
        true_speaker_slot=np.asarray(z["true_speaker_slot"], dtype=np.int32),
        process_order=np.asarray(z["process_order"], dtype=np.int32),
        v_id=np.asarray(z["v_id"], dtype=np.int32),
        hdbscan_was_noise=np.asarray(z["hdbscan_was_noise"], dtype=bool),
        registry_id=rid,
        meta=dict(meta),
        utterance_confidence=uc,
    )


def save_embedding_store(
    path: Union[str, Path],
    store: EmbeddingStore,
    *,
    max_utterances: Optional[int] = None,
    high_conf_floor: float = 0.68,
) -> Path:
    wr = [str(x) for x in np.asarray(store.wav_rel, dtype=object).tolist()]
    rid = [str(x) for x in np.asarray(store.registry_id, dtype=object).tolist()]
    return save_embedding_npz(
        Path(path),
        E=store.E,
        wav_rel=wr,
        true_speaker_slot=store.true_speaker_slot,
        process_order=store.process_order,
        v_id=store.v_id,
        hdbscan_was_noise=store.hdbscan_was_noise,
        registry_id=rid,
        meta=dict(store.meta),
        max_utterances=max_utterances,
        high_conf_floor=high_conf_floor,
    )


def register_speaker_embedding(
    npz_path: Union[str, Path],
    *,
    embedding: np.ndarray,
    registry_id: str,
    wav_rel: str,
    replace_existing: bool = False,
) -> EmbeddingStore:
    """往 npz 写一条注册声纹，不经过聚类。"""
    key = str(registry_id).strip()
    if not key:
        raise ValueError("registry_id 不能为空")
    emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    path = Path(npz_path).resolve()

    if path.is_file():
        store = load_embedding_npz(path)
        regs = [str(x).strip() for x in np.asarray(store.registry_id, dtype=object).tolist()]
        if key in regs:
            if not replace_existing:
                raise ValueError(f"registry_id {key!r} 已存在，传 replace_existing=True 可覆盖声纹")
            i = regs.index(key)
            E = np.asarray(store.E, dtype=np.float32).copy()
            E[i] = emb.reshape(-1)
            wr_list = [str(x) for x in np.asarray(store.wav_rel, dtype=object).tolist()]
            wr_list[i] = str(wav_rel)
            new_store = EmbeddingStore(
                E=E,
                wav_rel=np.asarray(wr_list, dtype=object),
                true_speaker_slot=store.true_speaker_slot.copy(),
                process_order=store.process_order.copy(),
                v_id=store.v_id.copy(),
                hdbscan_was_noise=store.hdbscan_was_noise.copy(),
                registry_id=store.registry_id.copy(),
                meta=dict(store.meta),
                utterance_confidence=(
                    None
                    if store.utterance_confidence is None
                    else store.utterance_confidence.copy()
                ),
            )
            new_store.meta["n_utterances"] = int(E.shape[0])
            new_store.meta.setdefault("registered_rows", True)
            save_embedding_store(path, new_store)
            return new_store

        n = store.n_utterances
        E_new = np.vstack([np.asarray(store.E, dtype=np.float32), emb])
        wr_list = [
            str(x) for x in np.asarray(store.wav_rel, dtype=object).tolist()
        ] + [str(wav_rel)]
        ts = np.concatenate(
            [store.true_speaker_slot, np.array([-1], dtype=np.int32)]
        )
        po = np.concatenate([store.process_order, np.array([n], dtype=np.int32)])
        vid = np.concatenate([store.v_id, np.array([-1], dtype=np.int32)])
        noise = np.concatenate(
            [store.hdbscan_was_noise, np.array([False], dtype=bool)]
        )
        rid_list = regs + [key]
        meta = dict(store.meta)
        meta["n_utterances"] = int(E_new.shape[0])
        meta.setdefault("registered_rows", True)
        new_store = EmbeddingStore(
            E=E_new.astype(np.float32),
            wav_rel=np.asarray(wr_list, dtype=object),
            true_speaker_slot=ts,
            process_order=po,
            v_id=vid,
            hdbscan_was_noise=noise,
            registry_id=np.asarray(rid_list, dtype=object),
            meta=meta,
            utterance_confidence=(
                None
                if store.utterance_confidence is None
                else np.concatenate(
                    [
                        store.utterance_confidence,
                        np.zeros(1, dtype=np.float32),
                    ]
                )
            ),
        )
        save_embedding_store(path, new_store)
        return new_store

    new_store = EmbeddingStore(
        E=emb.astype(np.float32),
        wav_rel=np.array([str(wav_rel)], dtype=object),
        true_speaker_slot=np.array([-1], dtype=np.int32),
        process_order=np.array([0], dtype=np.int32),
        v_id=np.array([-1], dtype=np.int32),
        hdbscan_was_noise=np.array([False], dtype=bool),
        registry_id=np.array([key], dtype=object),
        meta={
            "schema_version": SCHEMA_VERSION,
            "source": "speaker_registry_only",
            "n_utterances": 1,
        },
        utterance_confidence=None,
    )
    save_embedding_store(path, new_store)
    return new_store


def cosine_topk(
    query: np.ndarray,
    E: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """归一化后按余弦取前 k 条。"""
    q = np.asarray(query, dtype=np.float64).ravel()
    eq = np.linalg.norm(q)
    if eq < 1e-12:
        raise ValueError("query 范数过小")
    q = q / eq
    X = np.asarray(E, dtype=np.float64)
    en = np.linalg.norm(X, axis=1, keepdims=True)
    en = np.maximum(en, 1e-12)
    Xn = X / en
    sims = Xn @ q
    k = int(min(k, len(sims)))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def match_utterance(
    store: EmbeddingStore,
    query_emb: np.ndarray,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """单条查询的前 k 个最近邻。"""
    idx, scores = cosine_topk(query_emb, store.E, k=k)
    out: List[Dict[str, Any]] = []
    for rank, (j, s) in enumerate(zip(idx.tolist(), scores.tolist()), start=1):
        rk = str(store.registry_id[j]).strip()
        hit: Dict[str, Any] = {
            "rank": rank,
            "index": int(j),
            "cosine_similarity": float(s),
            "wav_rel": str(store.wav_rel[j]),
            "true_speaker_slot": int(store.true_speaker_slot[j]),
            "v_id": int(store.v_id[j]),
        }
        if rk:
            hit["registry_id"] = rk
        out.append(hit)
    return out
