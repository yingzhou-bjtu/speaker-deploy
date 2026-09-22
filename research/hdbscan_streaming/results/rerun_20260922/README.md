# Full rerun after prototype-repair fix (2026-09-22)

All results below were produced by the repaired `run_benchmark.py`
(`noise-fraction-routed repair`, landed 2026-09-22 16:38) and reflect the
current method. They supersede the pre-fix synthetic numbers.

## What changed vs pre-fix
The repair routing only changes behavior on low-noise prefixes (noise fraction
< 0.25). FSDD prefixes keep 36--49% noise and are unaffected (verified: main
table and repair ablation are bit-identical). Synthetic prefixes (~17--19%
noise) change.

### Async policy-isolation (source: submission_experiments.tex policy paragraph)
- fsdd adaptive: mean 0.282 ms / p95 0.693 ms (was 0.405 / 0.665)
- fsdd fishdbc: mean 1.077 / p95 1.445 (was 2.319 / 4.156)
- synthetic adaptive: mean 0.501 / p95 1.284, ACC 0.9375 / ARI 0.9150
  (was 0.548 / 1.439, ACC 0.9156 / ARI 0.8781)
- synthetic fishdbc: mean 5.039 / p95 8.858 (was 5.971 / 9.713)

### Synthetic main (S320 / D320 / D640)
- S320: FlowFish ACC 0.9375 / ARI 0.9150 vs FISHDBC 0.9156 / 0.8781; P95 1.19 vs 8.56 ms.
- D320: both ACC/ARI 1.0000; P95 1.15 vs 8.75 ms.
- D640: FlowFish ACC/ARI 1.0000 vs FISHDBC 0.9969/0.9986; P95 3.94 vs 19.35 ms.

## Files
- rerun_summary.csv: aggregated metrics.
- async_*: async validation single-method runs.
- synthetic_S320 / synthetic_D320 / synthetic_D640: synthetic main comparisons.
- multiseed/: 5-seed paired validation.
- rate_sweep/: arrival-rate sweep.
