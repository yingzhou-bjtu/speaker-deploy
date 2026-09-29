# Release Results

This page records the lightweight release bundle generated from the checked-in
fixtures and scripts. It is intended to make the headline claims inspectable
before opening the raw JSON and CSV files.

The measurements were collected on the current server-side CPU environment.
They are release-bundle measurements, not a replacement for the exact ICASSP
main-table configuration. The raw evidence remains authoritative.

## Matched Async Comparison

`FlowFish` is the `adaptive_fishdbc_async` implementation. `P95` is the
steady-state foreground decision latency. `Recompute` counts density refreshes,
and `Queue` is the maximum observed queue depth.

| Stream | Method | P95 (ms) | ARI | Recompute | Queue |
| --- | --- | ---: | ---: | ---: | ---: |
| FSDD, 30 points | FlowFish | 0.207 | 0.722 | 12 | 1 |
| FSDD, 30 points | FISHDBC | 1.422 | 0.722 | 30 | 0 |
| Synthetic, 320 points | FlowFish | 1.567 | 0.915 | 94 | 3 |
| Synthetic, 320 points | FISHDBC | 8.383 | 0.878 | 320 | 0 |

The paired run shows the intended separation: FlowFish moves routine work off
the foreground path while retaining the committed density reference and
exposing deferred maintenance through queue and recomputation counters.

Source:
`results/async_validation_release/async_comparison.json`

## Paper-Scale L/M/H Rerun

The longer rerun uses the committed FSDD embedding fixtures and evaluates all
seven methods in the paper-facing driver. The table below isolates the two
incremental density methods so that the selective-maintenance comparison is
easy to inspect.

| Prefix | Method | P95 (ms) | ARI | ACC | Recompute |
| --- | --- | ---: | ---: | ---: | ---: |
| L, 30 points | FlowFish | 0.216 | 0.722 | 0.767 | 12 |
| L, 30 points | FISHDBC | 1.890 | 0.722 | 0.767 | 30 |
| M, 300 points | FlowFish | 1.503 | 0.909 | 0.920 | 101 |
| M, 300 points | FISHDBC | 12.373 | 0.324 | 0.460 | 300 |
| H, 600 points | FlowFish | 8.198 | 0.822 | 0.805 | 186 |
| H, 600 points | FISHDBC | 24.457 | 0.421 | 0.515 | 600 |

All seven methods returned `status: ok` on each prefix. The complete
method-by-method summaries and logs are under:
`results/final_rerun_release/`.

## Five-Seed Robustness

The multi-seed run uses five paired synthetic streams with the seeds listed in
the reproduction command.

| Method | Mean P95 (ms) | Mean ARI | Mean ACC | Mean max queue |
| --- | ---: | ---: | ---: | ---: |
| FlowFish | 2.186 | 0.907 | 0.928 | 4.0 |
| FISHDBC | 10.505 | 0.819 | 0.870 | 0.0 |

The table reports means across seeds. Standard deviations and the complete
per-seed rows are in:
`results/multiseed_release/multiseed_summary.json` and
`results/multiseed_release/multiseed.csv`.

## Arrival-Rate Boundary

The rate sweep varies the inter-arrival interval while keeping the clustering
configuration fixed. At the most aggressive tested interval, FlowFish reaches
P95 `2.767 ms` with maximum queue depth `8`, while FISHDBC reaches P95
`9.686 ms`. At interval `20 ms`, the corresponding P95 values are `1.175 ms`
and `8.678 ms`.

The complete eight-row sweep is in:
`results/rate_sweep_release/rate_sweep.csv`.

## Reproduction

```bash
bash scripts/setup_research_env.sh
python3 scripts/check_release.py
bash scripts/reproduce_paper.sh release
```

The release command regenerates the smoke, async, rate, and multi-seed
artifacts. It does not download raw audio.

## Interpretation Boundary

These results evaluate the streaming clustering policy on committed embedding
fixtures. They do not establish cross-domain speaker-recognition quality, and
they do not imply that FlowFish dominates FISHDBC for every workload or
hardware platform. The queue depth and maintenance counters are part of the
result, not implementation details to be ignored.
