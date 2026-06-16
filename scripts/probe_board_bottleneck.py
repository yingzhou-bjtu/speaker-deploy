#!/usr/bin/env python3
"""板端瓶颈分析，逐条拆分嵌入、聚类和其它耗时。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import numpy as np

from speaker_id.audio_io import load_wav_16k_mono
from speaker_id.fbank_cpu import wav_to_fbank
from speaker_id.streaming_pipeline import StreamingClusterPipeline


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--ort-threads", type=int, default=4)
    ap.add_argument("--max-hdbscan-utterances", type=int, default=60)
    ap.add_argument("-o", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.audio_dir.glob("*.wav"))[: int(args.n)]
    if not files:
        print("无 wav", file=sys.stderr)
        return 1

    cap = int(args.max_hdbscan_utterances)
    kw = {
        "device": "cpu",
        "sv_backend": "onnx",
        "ort_threads": int(args.ort_threads),
        "min_cluster_size": 3,
        "min_samples": 2,
    }
    if cap > 0:
        kw["max_hdbscan_utterances"] = cap

    pipe = StreamingClusterPipeline(**kw)
    pipe._eng._ensure_onnx()

    rows = []
    # 预热一条。
    p0 = files[0]
    pipe.ingest_wav(p0, wav_key=p0.name)
    pipe.reset()

    for p in files:
        wav = load_wav_16k_mono(p)
        dur = len(wav) / 16000.0
        nfr = int(wav_to_fbank(wav).shape[0])

        t0 = time.perf_counter()
        emb = pipe._eng.embed_file(p)
        t_emb = time.perf_counter() - t0

        pipe._append_emb_row(np.asarray(emb, dtype=np.float32).reshape(-1))
        pipe._wav_keys.append(p.name)
        pipe._client_refs.append(None)
        pipe._y_true.append(-1)
        pipe._n_total_ingested += 1
        E = np.ascontiguousarray(pipe._E_prefix())
        n_buf = int(E.shape[0])

        from speaker_id.streaming_pipeline import _cluster_speakers_quiet

        t1 = time.perf_counter()
        cr = _cluster_speakers_quiet(pipe._eng, E, **pipe._hdb_kw())
        pipe._last_cr = cr
        pipe._rebuild_conf_cache(E)
        t_clu = time.perf_counter() - t1

        t_tot = t_emb + t_clu
        rows.append(
            {
                "wav": p.name[:48],
                "dur_s": round(dur, 2),
                "fbank_frames": nfr,
                "n_buffer": n_buf,
                "embed_s": round(t_emb, 4),
                "cluster_s": round(t_clu, 4),
                "total_s": round(t_tot, 4),
                "embed_pct": round(100 * t_emb / max(t_tot, 1e-9), 1),
            }
        )

    emb_s = sum(r["embed_s"] for r in rows)
    clu_s = sum(r["cluster_s"] for r in rows)
    tot_s = emb_s + clu_s

    print("=== 板端瓶颈（单条 ingest 拆分）===")
    print(f"配置: ort_threads={args.ort_threads} max_hdbscan={cap or '无上限'}")
    print(f"{'wav':42s} {'dur':>5s} {'fr':>5s} {'buf':>4s} {'embed':>7s} {'clus':>7s} {'%emb':>5s}")
    for r in rows:
        print(
            f"{r['wav']:42s} {r['dur_s']:5.1f} {r['fbank_frames']:5d} {r['n_buffer']:4d} "
            f"{r['embed_s']:7.3f} {r['cluster_s']:7.3f} {r['embed_pct']:5.1f}"
        )
    print("\n--- 合计 / 均值 ---")
    print(f"条数 {len(rows)}")
    print(f"总 embed   {emb_s:.2f}s  ({100*emb_s/tot_s:.1f}%)")
    print(f"总 cluster {clu_s:.2f}s  ({100*clu_s/tot_s:.1f}%)")
    print(f"总 wall    {tot_s:.2f}s  均值/条 {tot_s/len(rows):.3f}s")
    print(f"embed 均值 {emb_s/len(rows):.3f}s  cluster 均值 {clu_s/len(rows):.3f}s")
    print(f"cluster 最大 {max(r['cluster_s'] for r in rows):.3f}s (buf={rows[-1]['n_buffer']})")

    out = args.o
    if out:
        out = out.resolve()
        out.write_text(json.dumps({"rows": rows, "summary": {
            "embed_total_s": emb_s, "cluster_total_s": clu_s, "ort_threads": args.ort_threads,
            "max_hdbscan": cap,
        }}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
