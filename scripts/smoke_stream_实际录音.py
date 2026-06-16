#!/usr/bin/env python3
"""顺序 ingest 实际录音，检查首条不确定和后续聚类。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SMOKE_DIR = _ROOT.parent / "测试数据" / "实际录音"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

from speaker_id.cluster_panel import cluster_panel_counts, format_cluster_panel
from speaker_id.pipeline_factory import make_streaming_pipeline


def main() -> int:
    data_dir = Path(os.environ.get("SMOKE_DATA_DIR", str(_DEFAULT_SMOKE_DIR))).resolve()
    if not data_dir.is_dir():
        print(f"目录不存在: {data_dir}", file=sys.stderr)
        return 1

    n_take = int(os.environ.get("SMOKE_N", "12"))
    files = sorted(data_dir.glob("*.wav"))[:n_take]
    if not files:
        print("无 wav", file=sys.stderr)
        return 1

    pipe = make_streaming_pipeline(max_hdbscan_utterances=0, persist_registry=False)
    print(f"冒烟 ingest {len(files)} 条（共目录 {len(list(data_dir.glob('*.wav')))} 条 wav）\n")

    fails = 0
    for i, p in enumerate(files):
        r = pipe.ingest_wav(p, wav_key=p.name)
        vid = r.v_id_for_use if r.v_id_for_use is not None else "?"
        backend = (r.cluster_meta or {}).get("clustering_backend", "?")
        print(
            f"[{i+1:02d}] {p.name[:48]:48s} -> {vid!s:>4s}  "
            f"conf={r.confidence:.3f} emit={r.id_emitted} noise={r.hdbscan_was_noise} "
            f"backend={backend}"
        )
        if i == 0 and r.v_id_for_use is not None:
            fails += 1
            print("  FAIL: 首条应不确定（v_id_for_use=None）")

    assigned, uncertain, total = cluster_panel_counts(pipe)
    print("\n--- 最终面板 ---")
    print(format_cluster_panel(pipe))
    print(f"\n汇总: 确定簇 {assigned} | 不确定 {uncertain} | 总计 {total}")

    if fails:
        print("\n冒烟未通过", file=sys.stderr)
        return 1
    print("\n冒烟通过（首条未误发 id）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
