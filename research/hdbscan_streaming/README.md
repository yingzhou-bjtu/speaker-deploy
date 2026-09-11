# Streaming HDBSCAN Speaker Experiments

This directory contains the reproducible research materials for the streaming
speaker-clustering study in `speaker-deploy`.

## Contents

- `研究动机与贡献.md`: problem statement and intended paper positioning.
- `HDBSCAN流式折中算法_理论与证明草案.md`: connected assumptions, geometric
  lemmas, prefix-consistency theorem, and conditional tradeoff optimality.
- `HDBSCAN流式折中算法_实验记录.md`: experiment log and metric definitions.
- `scripts/`: benchmark, data preparation, validation, and plotting scripts.
- `data/`: synthetic streams and small embedding fixtures. Raw FSDD audio is
  intentionally excluded; see `data/fsdd/PROVENANCE.md` for provenance.
- `results/`: recorded summaries, tradeoff sweeps, stress tests, and figures.
- `RiskAware.lean`: Tau Ceti formal core for geometry, queue bounds, streaming
  prefix consistency, and pointwise gate-cost optimality.

## Reproduction

From this directory, install the dependencies from `requirements.txt`, then
run the scripts under `scripts/`. The exact commands and artifact paths are
documented in `HDBSCAN流式折中算法_实验记录.md` and the benchmark `README.md`.

The Lean file is a standalone excerpt. To compile it with Tau Ceti, copy it
under `TauCeti/Streaming/` in a Tau Ceti checkout and run:

```bash
lake env lean TauCeti/Streaming/RiskAware.lean
lake build
```

The formal results are conditional. They do not prove that nearest
representatives are universally equivalent to HDBSCAN or that the reference
clusterer is error-free. Those assumptions require calibration and streaming
experiments.
