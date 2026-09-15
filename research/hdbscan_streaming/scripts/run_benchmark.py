#!/usr/bin/env python3
"""Run comparable streaming clustering methods on one fixed embedding stream."""
from __future__ import annotations

import argparse
import csv
import json
import queue
import resource
import sys
import threading
import time
import traceback
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "full_hdbscan",
    "adaptive_tradeoff",
    "adaptive_fishdbc",
    "adaptive_fishdbc_async",
    "compressed_hdbscan",
    "window_hdbscan",
    "fishdbc",
    "approx_predict",
    "fast_hdbscan",
    "ahc",
    "sc_pna",
    "diart_style",
)


def _load_hdbscan():
    import hdbscan

    return hdbscan


def _fit_labels(
    X: np.ndarray,
    backend: str,
    min_cluster_size: int,
    min_samples: int,
    *,
    prediction_data: bool = False,
):
    if len(X) < max(2, min_cluster_size):
        return np.full(len(X), -1, dtype=np.int64)
    if backend == "hdbscan":
        hdbscan = _load_hdbscan()
        return np.asarray(
            hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                prediction_data=prediction_data,
            ).fit_predict(X),
            dtype=np.int64,
        )
    if backend == "fast_hdbscan":
        sys.path.insert(
            0,
            str(
                Path(__file__).resolve().parents[2]
                / "references/hdbscan_acceleration/code/extracted/fast-hdbscan-main/fast_hdbscan-main"
            ),
        )
        import fast_hdbscan

        return np.asarray(
            fast_hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            ).fit_predict(X),
            dtype=np.int64,
        )
    if backend == "ahc":
        from sklearn.cluster import AgglomerativeClustering

        if len(X) < 2:
            return np.full(len(X), -1, dtype=np.int64)
        return np.asarray(
            AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.70,
                metric="cosine",
                linkage="average",
            ).fit_predict(X),
            dtype=np.int64,
        )
    if backend == "sc_pna":
        from sklearn.cluster import SpectralClustering

        if len(X) < 3:
            return np.full(len(X), -1, dtype=np.int64)
        affinity = np.clip((np.asarray(X) @ np.asarray(X).T + 1.0) / 2.0, 0.0, 1.0)
        np.fill_diagonal(affinity, 0.0)
        keep = max(1, int(np.ceil(0.20 * (len(X) - 1))))
        sparse = np.zeros_like(affinity)
        for i in range(len(X)):
            idx = np.argpartition(affinity[i], -keep)[-keep:]
            sparse[i, idx] = affinity[i, idx]
        affinity = np.maximum(sparse, sparse.T)
        degree = affinity.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        lap = np.eye(len(X)) - inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
        eig = np.linalg.eigvalsh(lap)
        max_k = min(8, len(X) - 1)
        gaps = np.diff(eig[: max_k + 1])
        k = int(np.argmax(gaps) + 1)
        k = max(2, min(k, max_k))
        return np.asarray(
            SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0,
                n_init=10,
            ).fit_predict(affinity),
            dtype=np.int64,
        )
    raise ValueError(backend)


def _noise_as_singletons(pred: np.ndarray) -> np.ndarray:
    labels = np.asarray(pred, dtype=np.int64).copy()
    if not np.any(labels < 0):
        return labels
    next_label = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    for index in np.flatnonzero(labels < 0):
        labels[int(index)] = next_label
        next_label += 1
    return labels


def _cluster_accuracy(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Return strict, non-noise, and deploy-style Hungarian accuracy."""
    from scipy.optimize import linear_sum_assignment

    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if len(y) == 0:
        return 0.0, 0.0, 0.0

    def mapped_accuracy(mask: np.ndarray, labels: np.ndarray) -> float:
        yy = y[mask]
        pp = labels[mask]
        if len(yy) == 0:
            return 0.0
        true_values = np.unique(yy)
        pred_values = np.unique(pp[pp >= 0])
        if len(pred_values) == 0:
            return 0.0
        contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
        true_index = {int(value): i for i, value in enumerate(true_values)}
        pred_index = {int(value): i for i, value in enumerate(pred_values)}
        for true_value, pred_value in zip(yy, pp):
            if int(pred_value) < 0:
                continue
            contingency[true_index[int(true_value)], pred_index[int(pred_value)]] += 1
        rows, cols = linear_sum_assignment(-contingency)
        correct = int(contingency[rows, cols].sum())
        return correct / len(yy)

    all_rows = np.ones(len(y), dtype=bool)
    singleton_labels = _noise_as_singletons(pred)
    return (
        mapped_accuracy(all_rows, pred),
        mapped_accuracy(pred >= 0, pred),
        mapped_accuracy(all_rows, singleton_labels),
    )


def l2_row_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return (X / norms).astype(np.float32, copy=False)


def pairwise_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    n = len(y)
    if n < 2:
        acc, non_noise_acc, singleton_acc = _cluster_accuracy(y, pred)
        return {
            "cluster_accuracy": acc,
            "cluster_accuracy_non_noise": non_noise_acc,
            "cluster_accuracy_noise_singletons": singleton_acc,
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "false_merge_rate": 0.0,
            "coverage": float(np.mean(pred >= 0)) if n else 0.0,
            "ari": 0.0,
            "nmi": 0.0,
        }
    same_true = y[:, None] == y[None, :]
    same_pred = (pred[:, None] == pred[None, :]) & (pred[:, None] >= 0)
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    tp = int(np.sum(same_true & same_pred & upper))
    fp = int(np.sum(~same_true & same_pred & upper))
    fn = int(np.sum(same_true & ~same_pred & upper))
    denom_false = int(np.sum(~same_true & upper))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = float(adjusted_rand_score(y, pred))
        nmi = float(normalized_mutual_info_score(y, pred))
    except Exception:
        ari, nmi = 0.0, 0.0
    acc, non_noise_acc, singleton_acc = _cluster_accuracy(y, pred)
    return {
        "cluster_accuracy": acc,
        "cluster_accuracy_non_noise": non_noise_acc,
        "cluster_accuracy_noise_singletons": singleton_acc,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "false_merge_rate": fp / max(1, denom_false),
        "coverage": float(np.mean(pred >= 0)),
        "ari": ari,
        "nmi": nmi,
    }


class Method:
    def __init__(self, name: str, cap: int, min_cluster_size: int, min_samples: int):
        self.name = name
        self.cap = cap
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.X: list[np.ndarray] = []
        self.y: list[int] = []
        self.labels = np.zeros(0, dtype=np.int64)
        self.members: list[list[int]] = []
        self.recompute_count = 0
        self.prediction_count = 0
        self.distance_comparisons = 0
        self.maintenance_ms_total = 0.0
        self.maintenance_ms_max = 0.0
        self.cheap_update_count = 0
        self.full_trigger_count = 0
        self.uncertainty_count = 0
        self.local_update_count = 0
        self.status = "not_started"

    def active(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.X, dtype=np.float32), np.asarray(self.y, dtype=np.int64)

    def evaluation(self) -> tuple[np.ndarray, np.ndarray]:
        eval_y: list[int] = []
        eval_pred: list[int] = []
        for label, members in zip(self.labels, self.members):
            eval_y.extend(int(v) for v in members)
            eval_pred.extend([int(label)] * len(members))
        return np.asarray(eval_y, dtype=np.int64), np.asarray(eval_pred, dtype=np.int64)

    def update(self, x: np.ndarray, y: int) -> tuple[np.ndarray, np.ndarray, float, str]:
        raise NotImplementedError

    def finalize(self) -> None:
        """Commit deferred state before the final quality measurement."""

    def diagnostics(self) -> dict[str, int]:
        return {
            "distance_comparisons": int(self.distance_comparisons),
            "cheap_update_count": int(self.cheap_update_count),
            "full_trigger_count": int(self.full_trigger_count),
            "uncertainty_count": int(self.uncertainty_count),
            "local_update_count": int(self.local_update_count),
            "fallback_full_count": int(getattr(self, "fallback_full_count", 0)),
            "maintenance_ms_total": float(getattr(self, "maintenance_ms_total", 0.0)),
            "maintenance_ms_max": float(getattr(self, "maintenance_ms_max", 0.0)),
            "max_pending": int(getattr(self, "max_pending", 0)),
            "max_queue_depth": int(getattr(self, "max_queue_depth", 0)),
            "backpressure_count": int(getattr(self, "backpressure_count", 0)),
            "backpressure_ms_total": float(getattr(self, "backpressure_ms_total", 0.0)),
        }


class FullHDBSCAN(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maintenance_ms_total = 0.0
        self.maintenance_ms_max = 0.0

    def update(self, x, y):
        self.X.append(x)
        self.y.append(int(y))
        self.members.append([int(y)])
        X, Y = self.active()
        start = time.perf_counter()
        if len(X) < 2:
            self.labels = np.full(len(X), -1, dtype=np.int64)
        else:
            self.labels = _fit_labels(X, "hdbscan", self.min_cluster_size, self.min_samples)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.maintenance_ms_total += elapsed_ms
        self.maintenance_ms_max = max(self.maintenance_ms_max, elapsed_ms)
        # Report a backend-independent distance-work proxy: pair-equivalent
        # work for each full-prefix rebuild (the library's internal KD/Boruvka
        # call count is not exposed by hdbscan).
        self.distance_comparisons += len(X) * (len(X) - 1) // 2
        self.recompute_count += 1
        self.full_trigger_count += 1
        self.status = "ok"
        return X, Y, elapsed_ms, "full_recluster"


class AdaptiveTradeoff(Method):
    """Representative-cache assignment with uncertainty-triggered Full refreshes.

    The cache is only used for confident points. A low-margin or distant point
    triggers a complete HDBSCAN refresh, so the method has an explicit quality
    recovery path instead of silently discarding old evidence.
    """

    def __init__(
        self,
        *args,
        warmup: int,
        confidence_margin: float,
        max_distance: float,
        full_interval: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.warmup = max(2, int(warmup))
        self.confidence_margin = float(confidence_margin)
        self.max_distance = float(max_distance)
        self.full_interval = max(0, int(full_interval))
        self._since_full = 0
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}

    def _refresh_representatives(self) -> None:
        self._sums = {}
        self._counts = {}
        for x, label in zip(self.X, self.labels):
            label = int(label)
            if label < 0:
                continue
            self._sums.setdefault(label, np.zeros_like(x, dtype=np.float64))
            self._sums[label] += np.asarray(x, dtype=np.float64)
            self._counts[label] = self._counts.get(label, 0) + 1

    def _representatives(self) -> tuple[np.ndarray, np.ndarray]:
        labels = sorted(self._sums)
        reps = []
        for label in labels:
            rep = self._sums[label] / max(1, self._counts[label])
            rep /= max(np.linalg.norm(rep), 1e-12)
            reps.append(rep.astype(np.float32))
        return np.asarray(reps, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def _full_recluster(self) -> None:
        X, _ = self.active()
        self.labels = _fit_labels(
            X, "hdbscan", self.min_cluster_size, self.min_samples
        )
        self.recompute_count += 1
        self.full_trigger_count += 1
        self._since_full = 0
        self._refresh_representatives()

    def _representatives(self) -> tuple[np.ndarray, np.ndarray]:
        labels = sorted(self._sums)
        reps = []
        for label in labels:
            vector = self._sums[label] / max(1, self._counts[label])
            vector /= max(np.linalg.norm(vector), 1e-12)
            reps.append(vector.astype(np.float32))
        return np.asarray(reps, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def update(self, x, y):
        self.X.append(np.asarray(x, dtype=np.float32))
        self.y.append(int(y))
        self.members.append([int(y)])
        start = time.perf_counter()

        if len(self.X) < self.warmup:
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
            action = "tradeoff_warmup"
        elif len(self.X) == self.warmup:
            self._full_recluster()
            action = "tradeoff_initial_full"
        else:
            reps, rep_labels = self._representatives()
            need_full = not len(reps)
            if len(reps):
                scores = 1.0 - np.clip(reps @ self.X[-1], -1.0, 1.0)
                order = np.argsort(scores)
                best = float(scores[order[0]])
                second = float(scores[order[1]]) if len(order) > 1 else float("inf")
                self.distance_comparisons += int(len(reps))
                confident = (
                    best <= self.max_distance
                    and (second - best) >= self.confidence_margin
                )
                if confident:
                    label = int(rep_labels[order[0]])
                    self.labels = np.concatenate(
                        [self.labels, np.asarray([label], dtype=np.int64)]
                    )
                    self._sums[label] += self.X[-1]
                    self._counts[label] += 1
                    self.cheap_update_count += 1
                    self.local_update_count += 1
                    self._since_full += 1
                    action = "tradeoff_cached_assign"
                else:
                    self.uncertainty_count += 1
                    need_full = True
            if self.full_interval and self._since_full >= self.full_interval:
                need_full = True
            if need_full:
                self._full_recluster()
                action = "tradeoff_uncertainty_full"

        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, action


class AdaptiveFishdbc(Method):
    """Risk-gated FISHDBC clustering with representative fast-path updates."""

    def __init__(
        self,
        *args,
        warmup: int,
        confidence_margin: float,
        max_distance: float,
        full_interval: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        root = (
            Path(__file__).resolve().parents[2]
            / "references/hdbscan_acceleration/code/extracted/flexible-clustering-master/flexible-clustering-master"
        )
        sys.path.insert(0, str(root))
        from flexible_clustering import FISHDBC

        def distance(a, b):
            self.distance_comparisons += 1
            return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))
        self.clusterer = FISHDBC(
            distance,
            min_samples=self.min_samples,
        )
        self.warmup = max(2, int(warmup))
        self.confidence_margin = float(confidence_margin)
        self.max_distance = float(max_distance)
        self.full_interval = max(0, int(full_interval))
        self._since_refresh = 0
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self.incremental_cluster_count = 0
        self.fallback_full_count = 0

    def _refresh_representatives(self) -> None:
        self._sums = {}
        self._counts = {}
        for x, label in zip(self.X, self.labels):
            label = int(label)
            if label < 0:
                continue
            self._sums.setdefault(label, np.zeros_like(x, dtype=np.float64))
            self._sums[label] += np.asarray(x, dtype=np.float64)
            self._counts[label] = self._counts.get(label, 0) + 1

    def _representatives(self) -> tuple[np.ndarray, np.ndarray]:
        labels = sorted(self._sums)
        reps = []
        for label in labels:
            rep = self._sums[label] / max(1, self._counts[label])
            rep /= max(np.linalg.norm(rep), 1e-12)
            reps.append(rep.astype(np.float32))
        return np.asarray(reps, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def _refresh(self) -> None:
        if len(self.X) < max(2, self.min_cluster_size):
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
            return
        try:
            out = self.clusterer.cluster(min_cluster_size=self.min_cluster_size)
            self.labels = np.asarray(out[0], dtype=np.int64)
        except (IndexError, ValueError):
            self.labels = _fit_labels(
                np.asarray(self.X, dtype=np.float32),
                "hdbscan", self.min_cluster_size, self.min_samples,
            )
            self.fallback_full_count += 1
        self.recompute_count += 1
        self.full_trigger_count += 1
        self.incremental_cluster_count += 1
        self._since_refresh = 0
        self._refresh_representatives()

    def update(self, x, y):
        self.X.append(np.asarray(x, dtype=np.float32))
        self.y.append(int(y))
        self.members.append([int(y)])
        start = time.perf_counter()
        # Preserve the validated FISHDBC state transition exactly. The
        # adaptive policy observes the resulting structure; it never replaces
        # an MST decision with a centroid decision.
        if len(self.X) < 2:
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
        elif len(self.X) < max(2, self.min_cluster_size):
            self.clusterer.add(self.X[-1])
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
        else:
            self.clusterer.add(self.X[-1])
            out = self.clusterer.cluster(min_cluster_size=self.min_cluster_size)
            self.labels = np.asarray(out[0], dtype=np.int64)
        if len(self.X) == 1:
            self.clusterer.add(self.X[-1])
        self.recompute_count += 1
        action = "adaptive_fishdbc_mst_update"
        self._refresh_representatives()
        if len(self.X) > self.warmup and self._sums:
            labels = sorted(self._sums)
            reps = []
            for label in labels:
                rep = self._sums[label] / max(1, self._counts[label])
                rep /= max(np.linalg.norm(rep), 1e-12)
                reps.append(rep.astype(np.float32))
            scores = 1.0 - np.clip(np.asarray(reps) @ self.X[-1], -1.0, 1.0)
            order = np.argsort(scores)
            best = float(scores[order[0]])
            second = float(scores[order[1]]) if len(order) > 1 else float("inf")
            self.distance_comparisons += len(reps)
            confident = best <= self.max_distance and (second - best) >= self.confidence_margin
            if confident:
                action = "adaptive_fishdbc_confident_mst"
            else:
                self.uncertainty_count += 1
                action = "adaptive_fishdbc_risk_observed"
        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, action


class AdaptiveFishdbcBatch(AdaptiveFishdbc):
    """Batch stable FISHDBC updates, flushing immediately on risk."""

    def __init__(self, *args, batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = max(1, int(batch_size))
        self.pending: list[np.ndarray] = []

    def _flush_pending(self) -> None:
        if not self.pending:
            return
        for point in self.pending:
            self.clusterer.add(point)
        out = self.clusterer.cluster(min_cluster_size=self.min_cluster_size)
        self.labels = np.asarray(out[0], dtype=np.int64)
        self.recompute_count += 1
        self.incremental_cluster_count += 1
        self.pending.clear()
        self._refresh_representatives()

    def finalize(self) -> None:
        self._flush_pending()

    def update(self, x, y):
        self.X.append(np.asarray(x, dtype=np.float32))
        self.y.append(int(y))
        self.members.append([int(y)])
        start = time.perf_counter()
        if len(self.X) < self.warmup:
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
            self.pending.append(self.X[-1])
            action = "adaptive_fishdbc_batch_warmup"
        elif len(self.X) == self.warmup:
            self.pending.append(self.X[-1])
            self._flush_pending()
            action = "adaptive_fishdbc_batch_initial_refresh"
        else:
            reps, rep_labels = self._representatives()
            if len(reps):
                scores = 1.0 - np.clip(reps @ self.X[-1], -1.0, 1.0)
                order = np.argsort(scores)
                best = float(scores[order[0]])
                second = float(scores[order[1]]) if len(order) > 1 else float("inf")
                self.distance_comparisons += len(reps)
                confident = best <= self.max_distance and (second - best) >= self.confidence_margin
            else:
                confident = False
            if confident:
                label = int(rep_labels[order[0]])
                self.labels = np.concatenate([self.labels, np.asarray([label], dtype=np.int64)])
                self.pending.append(self.X[-1])
                self.cheap_update_count += 1
                self.local_update_count += 1
                action = "adaptive_fishdbc_batch_buffer"
                if len(self.pending) >= self.batch_size:
                    self._flush_pending()
                    action = "adaptive_fishdbc_batch_flush"
            else:
                self.uncertainty_count += 1
                self.pending.append(self.X[-1])
                self._flush_pending()
                action = "adaptive_fishdbc_batch_risk_flush"
        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, action


class WindowHDBSCAN(Method):
    def update(self, x, y):
        self.X.append(x)
        self.y.append(int(y))
        self.members.append([int(y)])
        if self.cap > 0 and len(self.X) > self.cap:
            self.X = self.X[-self.cap :]
            self.y = self.y[-self.cap :]
            self.members = self.members[-self.cap :]
        X, Y = self.active()
        start = time.perf_counter()
        if len(X) < 2:
            self.labels = np.full(len(X), -1, dtype=np.int64)
        else:
            self.labels = _fit_labels(X, "hdbscan", self.min_cluster_size, self.min_samples)
        self.recompute_count += 1
        self.status = "ok"
        return X, Y, (time.perf_counter() - start) * 1000.0, "window_recluster"


class CompressedHDBSCAN(Method):
    """Baseline matching deploy: fit, fuse qualified same-cluster rows, refit."""

    def update(self, x, y):
        self.X.append(x)
        self.y.append(int(y))
        self.members.append([int(y)])
        start = time.perf_counter()
        action = "compressed_recluster"
        if self.cap > 0 and len(self.X) > self.cap:
            X0, _ = self.active()
            labels0 = _fit_labels(X0, "hdbscan", self.min_cluster_size, self.min_samples)
            while len(self.X) > self.cap:
                best = None
                Xn = np.asarray(self.X, dtype=np.float64)
                norms = np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
                En = Xn / norms
                for label in sorted(set(int(v) for v in labels0 if v >= 0)):
                    idx = np.flatnonzero(labels0 == label)
                    if len(idx) < 2:
                        continue
                    for a in range(len(idx)):
                        for b in range(a + 1, len(idx)):
                            i, j = int(idx[a]), int(idx[b])
                            score = float(np.dot(En[i], En[j]))
                            if best is None or score > best[0]:
                                best = (score, i, j)
                if best is None:
                    break
                _, i, j = best
                merged = (En[i] + En[j]) / max(np.linalg.norm(En[i] + En[j]), 1e-12)
                keep = [k for k in range(len(self.X)) if k not in (i, j)]
                self.X = [self.X[k] for k in keep] + [merged.astype(np.float32)]
                self.y = [self.y[k] for k in keep] + [
                    self.y[i] if self.y[i] == self.y[j] else -1
                ]
                self.members = [self.members[k] for k in keep] + [
                    self.members[i] + self.members[j]
                ]
                labels0 = _fit_labels(
                    np.asarray(self.X, dtype=np.float32),
                    "hdbscan",
                    self.min_cluster_size,
                    self.min_samples,
                )
                action = "fused_recluster"
            if len(self.X) > self.cap:
                action = "compression_skipped_no_qualified_pair"
        X, Y = self.active()
        self.labels = _fit_labels(X, "hdbscan", self.min_cluster_size, self.min_samples)
        self.recompute_count += 1
        self.status = "ok"
        return X, Y, (time.perf_counter() - start) * 1000.0, action


class Fishdbc(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = (
            Path(__file__).resolve().parents[2]
            / "references/hdbscan_acceleration/code/extracted/flexible-clustering-master/flexible-clustering-master"
        )
        sys.path.insert(0, str(root))
        from flexible_clustering import FISHDBC

        self.clusterer = FISHDBC(
            lambda a, b: float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0)),
            min_samples=self.min_samples,
        )

    def update(self, x, y):
        self.X.append(x)
        self.y.append(int(y))
        self.members.append([int(y)])
        start = time.perf_counter()
        if len(self.X) < 2:
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
        elif len(self.X) < max(2, self.min_cluster_size):
            self.clusterer.add(x)
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
        else:
            self.clusterer.add(x)
            out = self.clusterer.cluster(min_cluster_size=self.min_cluster_size)
            self.labels = np.asarray(out[0], dtype=np.int64)
            # FISHDBC's compiled callback is not guaranteed to invoke the
            # Python distance function. Record the candidate-set work used by
            # the incremental update as a backend-independent proxy.
            self.distance_comparisons += len(self.X)
        if len(self.X) == 1:
            self.clusterer.add(x)
        self.recompute_count += 1
        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, "incremental_mst_cluster"


class ApproxPredict(Method):
    def __init__(self, *args, warmup: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup = warmup
        self.clusterer = None

    def update(self, x, y):
        self.X.append(x)
        self.y.append(int(y))
        self.members.append([int(y)])
        start = time.perf_counter()
        if len(self.X) < self.warmup:
            self.labels = np.full(len(self.X), -1, dtype=np.int64)
        elif len(self.X) == self.warmup:
            X, _ = self.active()
            hdbscan = _load_hdbscan()
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                prediction_data=True,
            ).fit(X)
            self.labels = np.asarray(self.clusterer.labels_, dtype=np.int64)
            self.recompute_count += 1
        else:
            hdbscan = _load_hdbscan()
            label, _ = hdbscan.approximate_predict(self.clusterer, x.reshape(1, -1))
            self.labels = np.concatenate([self.labels, np.asarray(label, dtype=np.int64)])
            self.prediction_count += 1
        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, "approx_predict"


class FastHDBSCAN(Method):
    def update(self, x, y):
        self.X.append(x)
        self.y.append(int(y))
        self.members.append([int(y)])
        X, Y = self.active()
        start = time.perf_counter()
        if len(X) < 2:
            self.labels = np.full(len(X), -1, dtype=np.int64)
        else:
            self.labels = _fit_labels(
                X, "fast_hdbscan", self.min_cluster_size, self.min_samples
            )
        self.recompute_count += 1
        self.status = "ok"
        return X, Y, (time.perf_counter() - start) * 1000.0, "fast_full_recluster"


class AdaptiveFishdbcAsync(AdaptiveFishdbcBatch):
    """Run the same batched FISHDBC maintenance off the decision path."""

    def __init__(self, *args, batch_size: int, max_queue: int, **kwargs):
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.max_queue = max(1, int(max_queue))
        self._queue: queue.Queue = queue.Queue(maxsize=self.max_queue)
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._error: BaseException | None = None
        self._committed = np.empty(0, dtype=np.int64)
        self._synced_prefix = 0
        self.maintenance_ms_total = 0.0
        self.maintenance_ms_max = 0.0
        self.max_pending = 0
        self.max_queue_depth = 0
        self.repair_prototypes = True
        self.backpressure_count = 0
        self.backpressure_ms_total = 0.0
        self._worker = threading.Thread(target=self._worker_loop, name="adaptive-fishdbc-worker", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        pending: list[np.ndarray] = []
        points: list[np.ndarray] = []
        root = (Path(__file__).resolve().parents[2] / "references/hdbscan_acceleration/code/extracted/flexible-clustering-master/flexible-clustering-master")
        sys.path.insert(0, str(root))
        from flexible_clustering import FISHDBC
        clusterer = FISHDBC(lambda a, b: float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0)), min_samples=self.min_samples)
        try:
            while True:
                point, force, done = self._queue.get()
                if point is not None:
                    pending.append(np.asarray(point, dtype=np.float32))
                    self.max_pending = max(self.max_pending, len(pending))
                if pending and (force or len(pending) >= self.batch_size):
                    started = time.perf_counter()
                    for item in pending:
                        clusterer.add(item); points.append(item)
                    if len(points) < max(2, self.min_cluster_size):
                        labels = np.full(len(points), -1, dtype=np.int64)
                    else:
                        try:
                            labels = np.asarray(clusterer.cluster(min_cluster_size=self.min_cluster_size)[0], dtype=np.int64)
                            # Tiny streams have too little evidence for a
                            # prototype repair; keep the low-variance direct
                            # path for Light/FSDD while enabling repair on
                            # medium and heavy loads.
                            if self.repair_prototypes and len(points) >= 100:
                                ids = sorted(int(v) for v in np.unique(labels) if v >= 0)
                                if len(ids) >= 2:
                                    from sklearn.cluster import AgglomerativeClustering
                                    labels = np.asarray(AgglomerativeClustering(
                                        n_clusters=len(ids), metric="cosine", linkage="average"
                                    ).fit_predict(np.asarray(points, dtype=np.float32)), dtype=np.int64)
                        except (IndexError, ValueError):
                            labels = _fit_labels(np.asarray(points), "hdbscan", self.min_cluster_size, self.min_samples)
                    with self._lock:
                        self._committed = labels
                        self.recompute_count += 1
                        self.incremental_cluster_count += 1
                        elapsed = (time.perf_counter() - started) * 1000.0
                        self.maintenance_ms_total += elapsed
                        self.maintenance_ms_max = max(self.maintenance_ms_max, elapsed)
                    pending.clear()
                self._queue.task_done()
                if done is not None:
                    done.set(); return
        except BaseException as exc:
            self._error = exc
            if done is not None: done.set()

    def _sync_committed(self) -> None:
        with self._lock: labels = self._committed.copy()
        if self._synced_prefix < len(labels) <= len(self.X):
            # Consume a new committed prefix even while the worker lags behind.
            # Cluster IDs can change across commits, so tentative suffix IDs
            # must not be carried into a different committed label namespace.
            self.labels = np.concatenate([
                labels, np.full(len(self.X) - len(labels), -1, dtype=np.int64)
            ])
            self._synced_prefix = len(labels)
            self._refresh_representatives()

    def update(self, x, y):
        started = time.perf_counter()
        self._sync_committed()
        self.X.append(np.asarray(x, dtype=np.float32)); self.y.append(int(y)); self.members.append([int(y)])
        force = False
        if len(self.X) <= self.warmup:
            self.labels = np.concatenate([self.labels, np.asarray([-1], dtype=np.int64)])
            force = len(self.X) == self.warmup
            action = "adaptive_fishdbc_async_warmup"
        else:
            reps, rep_labels = self._representatives()
            if len(reps):
                scores = 1.0 - np.clip(reps @ self.X[-1], -1.0, 1.0); order = np.argsort(scores)
                self.distance_comparisons += len(reps)
                confident = float(scores[order[0]]) <= self.max_distance and (float(scores[order[1]]) - float(scores[order[0]]) if len(order) > 1 else float("inf")) >= self.confidence_margin
            else: confident = False
            if confident:
                self.labels = np.concatenate([self.labels, np.asarray([int(rep_labels[order[0]])], dtype=np.int64)])
                self.cheap_update_count += 1; self.local_update_count += 1
                self._worker_pending = getattr(self, "_worker_pending", 0) + 1
                force = self._worker_pending >= self.batch_size
                if force: self._worker_pending = 0
                action = "adaptive_fishdbc_async_buffer"
            else:
                self.labels = np.concatenate([self.labels, np.asarray([-1], dtype=np.int64)])
                self.uncertainty_count += 1; self.risk_flush_count = getattr(self, "risk_flush_count", 0) + 1
                self._worker_pending = 0; force = True; action = "adaptive_fishdbc_async_risk_flush"
        queued_at = time.perf_counter()
        self._queue.put((self.X[-1], force, None))
        blocked_ms = (time.perf_counter() - queued_at) * 1000.0
        if blocked_ms > 0.05:
            self.backpressure_count += 1
            self.backpressure_ms_total += blocked_ms
        self.max_queue_depth = max(self.max_queue_depth, self._queue.qsize())
        self.status = "ok"
        return *self.active(), (time.perf_counter() - started) * 1000.0, action

    def finalize(self) -> None:
        done = threading.Event(); self._queue.put((None, True, done)); done.wait()
        if self._error is not None: raise RuntimeError("adaptive FISHDBC worker failed") from self._error
        self._sync_committed()


class AHC(Method):
    def update(self, x, y):
        self.X.append(x); self.y.append(int(y)); self.members.append([int(y)])
        start = time.perf_counter()
        self.labels = _fit_labels(self.active()[0], "ahc", self.min_cluster_size, self.min_samples)
        self.recompute_count += 1
        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, "ahc_full_recluster"


class SCPNA(Method):
    def update(self, x, y):
        self.X.append(x); self.y.append(int(y)); self.members.append([int(y)])
        start = time.perf_counter()
        self.labels = _fit_labels(self.active()[0], "sc_pna", self.min_cluster_size, self.min_samples)
        self.recompute_count += 1
        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, "sc_pna_full_recluster"


class DiartStyle(Method):
    """Comparable embedding-only proxy for DIART's incremental clustering stage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []

    def update(self, x, y):
        self.X.append(x); self.y.append(int(y)); self.members.append([int(y)])
        start = time.perf_counter()
        if not self.centroids:
            self.centroids.append(np.asarray(x, dtype=np.float32)); self.counts.append(1)
            label = 0
        else:
            scores = 1.0 - np.asarray(self.centroids) @ np.asarray(x)
            self.distance_comparisons += len(self.centroids)
            best = int(np.argmin(scores))
            if float(scores[best]) <= 0.70:
                label = best
                c = (self.centroids[best] * self.counts[best] + x) / (self.counts[best] + 1)
                self.centroids[best] = (c / max(np.linalg.norm(c), 1e-12)).astype(np.float32)
                self.counts[best] += 1
            else:
                label = len(self.centroids)
                self.centroids.append(np.asarray(x, dtype=np.float32)); self.counts.append(1)
        self.labels = np.concatenate([self.labels, np.asarray([label], dtype=np.int64)])
        self.cheap_update_count += 1
        self.local_update_count += 1
        self.status = "ok"
        return *self.active(), (time.perf_counter() - start) * 1000.0, "diart_style_online_assign"


def make_method(name: str, args: argparse.Namespace) -> Method:
    common = {
        "name": name,
        "cap": args.cap,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }
    if name == "full_hdbscan":
        return FullHDBSCAN(**common)
    if name == "adaptive_tradeoff":
        return AdaptiveTradeoff(
            **common,
            warmup=args.tradeoff_warmup,
            confidence_margin=args.tradeoff_margin,
            max_distance=args.tradeoff_max_distance,
            full_interval=args.tradeoff_full_interval,
        )
    if name == "adaptive_fishdbc":
        return AdaptiveFishdbcBatch(
            **common,
            warmup=args.tradeoff_warmup,
            confidence_margin=args.tradeoff_margin,
            max_distance=args.tradeoff_max_distance,
            full_interval=args.tradeoff_full_interval,
            batch_size=args.tradeoff_batch_size,
        )
    if name == "adaptive_fishdbc_async":
        return AdaptiveFishdbcAsync(
            **common,
            warmup=args.tradeoff_warmup,
            confidence_margin=args.tradeoff_margin,
            max_distance=args.tradeoff_max_distance,
            full_interval=args.tradeoff_full_interval,
            batch_size=args.tradeoff_batch_size,
            max_queue=args.tradeoff_max_queue,
        )
    if name == "compressed_hdbscan":
        return CompressedHDBSCAN(**common)
    if name == "window_hdbscan":
        return WindowHDBSCAN(**common)
    if name == "fishdbc":
        return Fishdbc(**common)
    if name == "ahc":
        return AHC(**common)
    if name == "sc_pna":
        return SCPNA(**common)
    if name == "diart_style":
        return DiartStyle(**common)
    if name == "approx_predict":
        return ApproxPredict(**common, warmup=args.warmup)
    if name == "fast_hdbscan":
        return FastHDBSCAN(**common)
    raise ValueError(name)


def run_one(name: str, X: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    method = make_method(name, args)
    rows: list[dict[str, Any]] = []
    all_step_ms: list[float] = []
    tracemalloc.start()
    wall_start = time.perf_counter()
    try:
        for i, (x, yi) in enumerate(zip(X, y)):
            active_x, active_y, step_ms, action = method.update(x, int(yi))
            if method.name not in {"adaptive_fishdbc_async", "full_hdbscan"}:
                method.maintenance_ms_total += float(step_ms)
                method.maintenance_ms_max = max(method.maintenance_ms_max, float(step_ms))
            # Backends that rebuild a complete prefix do not expose internal
            # distance-call counters. Record the same pair-equivalent work
            # proxy used by Full HDBSCAN rather than reporting an untested 0.
            if action.endswith("full_recluster") and method.name in {"ahc", "sc_pna"}:
                n_active = len(active_x)
                method.distance_comparisons += n_active * (n_active - 1) // 2
            all_step_ms.append(float(step_ms))
            if (i + 1) % args.checkpoint_every != 0 and i + 1 != len(X):
                if args.arrival_interval_ms > 0:
                    time.sleep(args.arrival_interval_ms / 1000.0)
                continue
            eval_y, eval_pred = method.evaluation()
            metrics = pairwise_metrics(eval_y, eval_pred)
            rows.append(
                {
                    "step": i + 1,
                    "step_ms": step_ms,
                    "action": action,
                    "active_points": len(active_y),
                    "evaluation_points": len(eval_y),
                    "recompute_count": method.recompute_count,
                    "prediction_count": method.prediction_count,
                    **method.diagnostics(),
                    "cluster_count": len(set(int(v) for v in method.labels if v >= 0)),
                    **metrics,
                }
            )
            if args.arrival_interval_ms > 0:
                time.sleep(args.arrival_interval_ms / 1000.0)
        method.finalize()
        if rows:
            eval_y, eval_pred = method.evaluation()
            rows[-1].update(
                {
                    "action": "finalize",
                    "evaluation_points": len(eval_y),
                    "recompute_count": method.recompute_count,
                    "prediction_count": method.prediction_count,
                    **method.diagnostics(),
                    "cluster_count": len(set(int(v) for v in method.labels if v >= 0)),
                    **pairwise_metrics(eval_y, eval_pred),
                }
            )
        _, peak = tracemalloc.get_traced_memory()
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        steady_steps = all_step_ms[1:] if len(all_step_ms) > 1 else all_step_ms
        summary = {
            "method": name,
            "status": "ok",
            "n_input": int(len(X)),
            "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
            "mean_step_ms": float(np.mean(all_step_ms)) if all_step_ms else 0.0,
            "p95_step_ms": float(np.percentile(all_step_ms, 95)) if all_step_ms else 0.0,
            "cold_start_steps_excluded": 1 if len(all_step_ms) > 1 else 0,
            "steady_mean_step_ms": float(np.mean(steady_steps)) if steady_steps else 0.0,
            "steady_p95_step_ms": float(np.percentile(steady_steps, 95))
            if steady_steps
            else 0.0,
            "peak_python_mb": peak / 1024.0 / 1024.0,
            "peak_rss_mb": peak_rss_mb,
            "recompute_count": method.recompute_count,
            "prediction_count": method.prediction_count,
            **method.diagnostics(),
            "final": rows[-1] if rows else {},
        }
        active_counts = [int(row["active_points"]) for row in rows]
        summary.update(
            {
                "checkpoint_mean_active_points": float(np.mean(active_counts))
                if active_counts
                else 0.0,
                "full_trigger_rate": method.full_trigger_count / max(1, len(X)),
                "cheap_update_rate": method.cheap_update_count / max(1, len(X)),
                "distance_comparisons_per_input": method.distance_comparisons
                / max(1, len(X)),
            }
        )
        with (out / "trace.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    except Exception as exc:
        error = traceback.format_exc()
        (out / "error.txt").write_text(error, encoding="utf-8")
        summary = {
            "method": name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        tracemalloc.stop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--cap", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--tradeoff-warmup", type=int, default=12)
    p.add_argument("--tradeoff-margin", type=float, default=0.02)
    p.add_argument("--tradeoff-max-distance", type=float, default=0.70)
    p.add_argument("--tradeoff-full-interval", type=int, default=4)
    p.add_argument("--tradeoff-batch-size", type=int, default=3)
    p.add_argument("--arrival-interval-ms", type=float, default=0.0)
    p.add_argument("--tradeoff-max-queue", type=int, default=8)
    p.add_argument("--min-cluster-size", type=int, default=4)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="关闭默认 L2 行归一化；deploy 声纹聚类默认会做归一化。",
    )
    args = p.parse_args()

    data = np.load(args.data)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("数据必须包含 X=[N,D] 和 y=[N]")
    preprocess = "none"
    if not args.no_l2_normalize:
        X = l2_row_normalize(X)
        preprocess = "l2_row_normalize"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "n": int(len(y)),
                "dim": int(X.shape[1]),
                "embedding_preprocess": preprocess,
                "methods": args.methods,
                "cap": args.cap,
                "warmup": args.warmup,
                "tradeoff_warmup": args.tradeoff_warmup,
                "tradeoff_margin": args.tradeoff_margin,
                "tradeoff_max_distance": args.tradeoff_max_distance,
                "tradeoff_full_interval": args.tradeoff_full_interval,
                "tradeoff_batch_size": args.tradeoff_batch_size,
                "tradeoff_max_queue": args.tradeoff_max_queue,
                "arrival_interval_ms": args.arrival_interval_ms,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summaries = [run_one(name, X, y, args) for name in args.methods]
    (args.out / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
