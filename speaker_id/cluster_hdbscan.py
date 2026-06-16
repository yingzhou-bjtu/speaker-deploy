"""HDBSCAN 聚类，和离线评测脚本对齐，不依赖 torch。"""
from __future__ import annotations

import inspect
import sys
from typing import Any, Dict, Tuple

import numpy as np

_SKLEARN_PATCHED = False


def _apply_sklearn_check_array_compat() -> None:
    """修补 hdbscan 和部分 sklearn 版本不兼容的问题。"""
    global _SKLEARN_PATCHED
    if _SKLEARN_PATCHED:
        return
    import sklearn.utils as skutils
    import sklearn.utils.validation as val

    _orig = val.check_array
    params = inspect.signature(_orig).parameters

    def _compat(
        X,
        accept_sparse="csr",
        ensure_all_finite=True,
        force_all_finite=None,
        **kwargs: Any,
    ):
        kw: Dict[str, Any] = {"accept_sparse": accept_sparse, **kwargs}
        if force_all_finite is not None and "force_all_finite" in params:
            kw["force_all_finite"] = force_all_finite
        elif ensure_all_finite is False:
            if "ensure_all_finite" in params:
                kw["ensure_all_finite"] = False
            elif "force_all_finite" in params:
                kw["force_all_finite"] = False
        elif ensure_all_finite is not True and "ensure_all_finite" in params:
            kw["ensure_all_finite"] = ensure_all_finite
        try:
            return _orig(X, **kw)
        except TypeError:
            kw.pop("ensure_all_finite", None)
            kw.pop("force_all_finite", None)
            return _orig(X, **kw)

    val.check_array = _compat  # type: ignore[method-assign]
    skutils.check_array = _compat  # type: ignore[attr-defined]
    _SKLEARN_PATCHED = True


def cluster_speakers(
    embs: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    n = int(embs.shape[0])
    mcs, ms = int(min_cluster_size), int(min_samples)
    eps = float(cluster_selection_epsilon)
    hdbscan_noise = np.zeros(n, dtype=bool)
    try:
        _apply_sklearn_check_array_compat()
        import hdbscan

        lab = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_epsilon=eps,
        ).fit_predict(embs)
        lab = np.asarray(lab, dtype=np.intp, order="C").copy()
        lab_dense = lab.copy()  # 重标噪声前备份，负一表示噪声点。
        hdbscan_noise = np.asarray(lab < 0, dtype=bool)
        n_noise_raw = int(np.sum(lab < 0))
        n_dense = int(len(np.unique(lab[lab >= 0]))) if np.any(lab >= 0) else 0
        if n_noise_raw > 0:
            nxt = int(np.max(lab[lab >= 0])) + 1 if np.any(lab >= 0) else 0
            for i in np.where(lab < 0)[0]:
                lab[i] = nxt
                nxt += 1
        n_total = int(len(np.unique(lab)))
        try:
            import importlib.metadata as im

            hdb_v = im.version("hdbscan")
        except Exception:
            hdb_v = "unknown"
        meta = {
            "clustering_backend": "HDBSCAN",
            "hdbscan_version": hdb_v,
            "metric": "euclidean",
            "min_cluster_size": mcs,
            "min_samples": ms,
            "cluster_selection_epsilon": eps,
            "n_hdbscan_dense_clusters": n_dense,
            "n_hdbscan_noise_each_unique_class": n_noise_raw,
            "n_distinct_pred_clusters": n_total,
        }
        return lab, meta, hdbscan_noise, lab_dense
    except (ImportError, TypeError) as e:
        from sklearn.cluster import AgglomerativeClustering

        k = max(2, min(32, max(2, n // 4)))
        print(
            f"[cluster] hdbscan 不可用 ({e!r})，回退 AgglomerativeClustering(n_clusters={k})",
            file=sys.stderr,
        )
        lab = AgglomerativeClustering(n_clusters=k).fit_predict(embs)
        meta = {
            "clustering_backend": "AgglomerativeClustering (fallback)",
            "n_clusters": k,
            "n_distinct_pred_clusters": int(len(np.unique(lab))),
        }
        lab = np.asarray(lab, dtype=np.intp, order="C")
        return lab, meta, hdbscan_noise, lab.copy()
