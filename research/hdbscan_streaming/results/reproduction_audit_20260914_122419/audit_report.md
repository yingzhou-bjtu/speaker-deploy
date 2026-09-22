ICASSP 2027 reproduction audit
Date: 2026-09-14
Root: /home/user/Desktop/paper/hdbscan_streaming_bench/results/reproduction_audit_20260914_122419

Fresh benchmark method runs: 94 ({'ok': 94})
Fresh error.txt files: 0
Fresh logs: 17; CSV: 76; JSON: 139

Verified tests:
- compileall: PASS
- dataset_contracts: PASS
- deploy_dense_label_equivalence: PASS
- fsdd_audio_wav_to_onnx_to_cluster: PASS
- rate_sweep: PASS
- multiseed: PASS
- tradeoff_sweep: PASS
- long_stress_640: PASS
- plots: PASS

Dataset assets:
- data/fsdd/embeddings/fsdd_smoke.npz: n=30 dim=192 speakers=6
- data/synthetic_320.npz: n=320 dim=64 speakers=8
- data/synthetic_drift_320.npz: n=320 dim=64 speakers=8
- data/synthetic_drift_640.npz: n=640 dim=64 speakers=8
- data/synthetic_smoke.npz: n=120 dim=32 speakers=5

Environment:
- venv_python: Python 3.10.12
- system_python: Python 3.10.12
- venv_matplotlib: MISSING
- system_matplotlib: 3.10.9
- pip_check: PASS

Important scope boundary:
- The paper statement “174 tasks” is not independently verifiable from the local project: no formal 174-task manifest was found.
- The fresh audit contains 94 benchmark method runs plus contract/preflight/audio/deployment/plot checks; all recorded benchmark method statuses are ok.
- The venv satisfies pip check but lacks matplotlib; plotting was verified with system Python 3.10.9 / matplotlib 3.10.9.
