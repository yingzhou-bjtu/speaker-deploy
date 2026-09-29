# FlowFish: Reproducible Streaming Speaker Clustering

This repository contains the reference implementation and reproducibility
package for **FlowFish**, a selective-maintenance policy for streaming density
clustering studied in an ICASSP 2027 manuscript.

Streaming speaker clustering has a structural tension: density-based methods
provide strong cluster structure, but a full maintenance step after every new
embedding is too expensive for low-latency streams. FlowFish addresses this by
separating latency-critical assignment from deferred density maintenance.
Confident arrivals are assigned through a representative distance-margin test,
while uncertain arrivals enter a bounded refresh path maintained by a
background FISHDBC worker.

The public repository is intentionally scoped to the paper-facing code and
evidence:

- benchmark implementations for FlowFish and comparison methods;
- committed synthetic and derived embedding fixtures;
- release result summaries, traces, and selected paper figures;
- scripts for reproducing the lightweight release bundle and main-table rerun;
- citation metadata, third-party notices, and artifact provenance.

Operational materials and exploratory records are not part of the public
research package.

## Repository Layout

```text
research/hdbscan_streaming/
  data/        synthetic fixtures and derived embedding fixtures
  figures/     selected paper figures for repository inspection
  results/     release summaries, traces, and tables
  scripts/     benchmark, validation, plotting, and release checks
  vendor/      third-party comparison backends with original licenses
  RiskAware.lean
```

The canonical paper entry point is:

[`research/hdbscan_streaming/README.md`](research/hdbscan_streaming/README.md)

The method name in the manuscript is **FlowFish**. The implementation key used
by the benchmark driver is `adaptive_fishdbc_async`.

## Reproduction

From the research package:

```bash
cd research/hdbscan_streaming
bash scripts/setup_research_env.sh
python3 scripts/check_release.py
bash scripts/reproduce_paper.sh release
```

The longer paper-scale rerun is:

```bash
bash scripts/reproduce_paper.sh main
```

The release commands use committed fixtures and do not require private audio
collections or organization-specific services. Raw FSDD audio is not
redistributed; provenance and artifact terms are documented in
[`research/hdbscan_streaming/data/fsdd/PROVENANCE.md`](research/hdbscan_streaming/data/fsdd/PROVENANCE.md).

## Results and Figures

The release measurements are summarized in
[`research/hdbscan_streaming/RESULTS.md`](research/hdbscan_streaming/RESULTS.md).
The selected paper figures are available under
[`research/hdbscan_streaming/figures/`](research/hdbscan_streaming/figures/).

The paired asynchronous comparison reports both foreground latency and
deferred-maintenance counters. On the committed release fixtures, FlowFish
reduces foreground P95 latency relative to FISHDBC while reporting quality,
queue depth, backpressure, and recomputation behavior. The repository does not
claim universal superiority across workloads or hardware platforms.

## Citation

If you use the implementation or results, cite the accompanying ICASSP 2027
paper and the upstream FISHDBC work. The software citation is provided in
[`research/hdbscan_streaming/CITATION.cff`](research/hdbscan_streaming/CITATION.cff).

## License and Artifact Terms

Original project source code is released under the MIT License. See
[`LICENSE`](LICENSE). Dataset-derived artifacts and vendored third-party code
retain separate terms documented in
[`research/hdbscan_streaming/THIRD_PARTY_NOTICES.md`](research/hdbscan_streaming/THIRD_PARTY_NOTICES.md).
