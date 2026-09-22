# Archived stale experiment directories

Moved here on 2026-09-22 to keep `results/` focused on the canonical reruns.
These are historical runs from earlier `run_benchmark.py` revisions whose
method paths no longer exist at the current script line numbers. None are
referenced by current scripts or docs.

- smoke_v4: old smoke with fast_hdbscan import failure.
- adaptive_fishdbc_batch1 / adaptive_fishdbc_batch3: old adaptive_fishdbc runs
  with a representative-refresh bug at the former line 535.

Active evidence lives in results/final_rerun (main table), results/scale_canonical
(scaling), and the reproduction_audit_20260914_122419 directory.
