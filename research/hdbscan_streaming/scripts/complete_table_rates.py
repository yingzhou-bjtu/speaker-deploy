"""Complete FlowFish rate columns with the canonical workload parameters."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'results/reproduction_audit_20260914_122419'
OUT = ROOT / 'results/table_rate_completion_20260915'
METHODS = ['full_hdbscan', 'fishdbc', 'ahc', 'sc_pna', 'diart_style',
           'adaptive_fishdbc', 'adaptive_fishdbc_async']
for scenario, folder in [('FSDD', 'fsdd_audio_ours'), ('S320', 'synthetic_320'),
                         ('D320', 'drift_320'), ('D640', 'drift_640')]:
    config = json.loads((AUDIT / folder / 'config.json').read_text())
    for interval in ([0, 5, 20] if scenario == 'FSDD' else [0, 20]):
        methods = METHODS if interval == 5 else ['adaptive_fishdbc_async']
        dest = OUT / scenario / f'interval_{interval}'
        args = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/run_benchmark.py'),
                '--data', config['data'], '--out', str(dest), '--methods', *methods]
        for key in ['cap', 'warmup', 'tradeoff_warmup', 'tradeoff_margin',
                    'tradeoff_max_distance', 'tradeoff_full_interval', 'tradeoff_batch_size',
                    'tradeoff_max_queue', 'min_cluster_size', 'min_samples', 'checkpoint_every']:
            args += ['--' + key.replace('_', '-'), str(config[key])]
        args += ['--arrival-interval-ms', str(interval)]
        dest.mkdir(parents=True, exist_ok=True)
        print(scenario, interval, 'START', flush=True)
        with (dest / 'execution.log').open('w') as log:
            subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for method in methods:
            data = json.loads((dest / method / 'summary.json').read_text())
            assert data['status'] == 'ok', data
        print(scenario, interval, 'VERIFIED', flush=True)
