# Scaling (canonical config)

FlowFish (`adaptive_fishdbc_async`) vs incremental FISHDBC baseline on
synthetic 8-speaker streams at 1280/2560 utterances, 5 ms arrival interval,
canonical configuration: warmup 20, delta 0.70, margin 0.02, batch 4, queue 8,
min_cluster_size 3, min_samples 6, prototype repair ON.

FlowFish routes repair by FISHDBC noise fraction: over-split prefixes (high
noise fraction) use aggressive merge; clean prefixes (low noise fraction) keep
noise points and re-insert only points inside a cluster's spread.

## Result summary
The repaired FlowFish preserves or improves ACC/ARI/NMI over FISHDBC at both
scales while cutting mean latency, at the cost of a bounded queue that shows
backpressure under the 5 ms arrival rate.

## Files
- scale_canonical.csv: aggregate metrics.
- n{1280,2560}/{...}/summary.json: raw summaries and traces.
