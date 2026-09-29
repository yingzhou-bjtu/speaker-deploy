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
bash scripts/reproduce_paper.sh all
```

`all` creates the research environment when needed, checks the release
contract, and runs the synthetic smoke, asynchronous comparison, arrival-rate
sweep, and five-seed protocol. Individual entry points are:

```bash
bash scripts/reproduce_paper.sh setup
bash scripts/reproduce_paper.sh data
FSDD_MODEL=/path/to/speaker_embedding.onnx bash scripts/reproduce_paper.sh audio
bash scripts/reproduce_paper.sh baseline
bash scripts/reproduce_paper.sh smoke
bash scripts/reproduce_paper.sh release
bash scripts/reproduce_paper.sh main
```

`data` downloads the public FSDD archive at a pinned upstream commit, verifies
the archive checksum, extracts the 3000 recordings, and updates the local
provenance record. The exact release benchmark uses committed embedding
fixtures, so raw audio is not required for the paper-scale reproduction.
Provenance and artifact terms are documented in
[`research/hdbscan_streaming/data/fsdd/PROVENANCE.md`](research/hdbscan_streaming/data/fsdd/PROVENANCE.md).

The raw-audio regeneration path is also public. After downloading FSDD, pass a
local ONNX speaker-embedding model to the complete audio smoke:

```bash
cd research/hdbscan_streaming
FSDD_MODEL=/path/to/speaker_embedding.onnx bash scripts/run_audio_smoke.sh
```

The pipeline is explicit and deterministic: `prepare_fsdd_audio.py` creates a
seeded balanced manifest, `audio_to_embeddings.py` writes the common `X`/`y`
embedding contract, and `run_audio_smoke.sh` runs the clustering methods. The
model is an external input and is not part of the public repository.

## Results and Figures

The release measurements are summarized in
[`research/hdbscan_streaming/RESULTS.md`](research/hdbscan_streaming/RESULTS.md).
Representative results from the committed release bundle are shown below.

| Stream | Method | Foreground P95 (ms) | ARI | Recompute | Max queue |
| --- | --- | ---: | ---: | ---: | ---: |
| FSDD, 30 points | FlowFish | 0.207 | 0.722 | 12 | 1 |
| FSDD, 30 points | FISHDBC | 1.422 | 0.722 | 30 | 0 |
| Synthetic, 320 points | FlowFish | 1.567 | 0.915 | 94 | 3 |
| Synthetic, 320 points | FISHDBC | 8.383 | 0.878 | 320 | 0 |

The complete tables and raw JSON/CSV paths are in
[`research/hdbscan_streaming/RESULTS.md`](research/hdbscan_streaming/RESULTS.md).

### Main comparison

![Latency and quality comparison](research/hdbscan_streaming/figures/fig_main_results.png)

### Arrival-rate sensitivity

![Arrival-rate sensitivity](research/hdbscan_streaming/figures/fig_stability_rate.png)

The paired asynchronous comparison reports foreground latency together with
quality, queue depth, backpressure, and recomputation behavior. The repository
does not claim universal superiority across workloads or hardware platforms.

## Reproduction Scripts

| Purpose | Command | Main artifact |
| --- | --- | --- |
| Environment | `bash scripts/reproduce_paper.sh setup` | `.venv/` |
| Dataset download | `bash scripts/reproduce_paper.sh data` | `data/fsdd/raw/` |
| Audio manifest and embedding | `FSDD_MODEL=... bash scripts/run_audio_smoke.sh` | `data/fsdd/selected/`, `data/fsdd/embeddings/` |
| Full HDBSCAN baseline | `bash scripts/run_full_baseline.sh` | `results/full_baseline/` |
| Synthetic smoke | `bash scripts/reproduce_paper.sh smoke` | `results/smoke_latest/` |
| Release bundle | `bash scripts/reproduce_paper.sh release` | `results/*_release/` |
| Paper-scale rerun | `bash scripts/reproduce_paper.sh main` | `results/final_rerun_release/` |

## Citation

If you use the implementation or results, cite the accompanying ICASSP 2027
paper and the upstream FISHDBC work. The software citation is provided in
[`research/hdbscan_streaming/CITATION.cff`](research/hdbscan_streaming/CITATION.cff).

## License and Artifact Terms

Original project source code is released under the MIT License. See
[`LICENSE`](LICENSE). Dataset-derived artifacts and vendored third-party code
retain separate terms documented in
[`research/hdbscan_streaming/THIRD_PARTY_NOTICES.md`](research/hdbscan_streaming/THIRD_PARTY_NOTICES.md).
