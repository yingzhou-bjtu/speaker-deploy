# Optimization Method Selection

This project uses `full_hdbscan` as the only reference implementation. The
reference processes the complete prefix after every incoming embedding, applies
the same L2 row normalization as `deploy`, and is verified against the deploy
clusterer.

## Current Selection

| Method | Role | Keep for next round | Reason |
|---|---|---:|---|
| `full_hdbscan` | reference | yes | The complete-history target that every optimization must compare against. |
| `adaptive_fishdbc` | proposed risk-gated batched FISHDBC | yes | Stable points are buffered and flushed in batches; risk points flush immediately, preserving a conservative recovery path. |
| `adaptive_fishdbc_async` | proposed asynchronous risk-gated FISHDBC | yes | Moves batched maintenance off the decision path and reports queue staleness and maintenance cost separately. |
| `fishdbc` | incremental MST candidate | yes | Preserves the current smoke quality while reducing steady per-step time. |
| `ahc` | classic clustering baseline | baseline | A directly comparable cosine-distance agglomerative baseline. |
| `sc_pna` | recent spectral baseline | baseline | Embedding-level implementation of the ICASSP 2025 SC-pNA idea. |
| `diart_style` | online centroid baseline | baseline | Embedding-only proxy for the incremental clustering stage of DIART. |
| `compressed_hdbscan` | cache compression candidate | no, redesign | Current fusion changes the clustering geometry and causes a large quality drop. |
| `window_hdbscan` | bounded-window baseline | baseline only | It discards old evidence; the current 20-row window has zero final coverage on the audio stream. |
| `approx_predict` | fixed-model prediction baseline | baseline only | It avoids reclustering but does not recover new cluster structure; current final coverage is zero. |
| `fast_hdbscan` | static backend control | separate control | It still performs full-history recomputation at every step, so it is not an incremental optimization. |

## Evidence

- Full baseline: `results/fsdd_full_baseline/`
- Candidate summaries: `results/fsdd_optimization_candidates/`
- Relative table: `results/fsdd_optimization_candidates/compare_to_full.csv`
- Deploy equivalence check:
  `results/fsdd_full_baseline/deploy_equivalence.json`

The current audio smoke is only 30 utterances from FSDD and uses the existing
Chinese speaker embedding model on English digit speech. It is suitable for
pipeline and regression checks, not for the final paper claim.

The core comparison has five baselines: Full HDBSCAN, FISHDBC, AHC, SC-pNA,
and DIART-style online centroids. The current main-method candidate is
`adaptive_fishdbc_async`; the synchronous `adaptive_fishdbc` is an ablation.
O-EENC-SD and Streaming Sortformer are not
run in this embedding-only table because they are end-to-end diarization models
with their own segmentation/attractor networks; their published results should
be reported separately as external references, not mixed with this clustering
contract.
