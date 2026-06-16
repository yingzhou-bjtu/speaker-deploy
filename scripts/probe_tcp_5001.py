#!/usr/bin/env python3
"""探测 robotmedia 5001 口说话时是否在推 PCM，不加载声纹模型。"""
from __future__ import annotations

import argparse
import socket
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="probe robotmedia vad_pcm_socket")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--duration", type=float, default=120.0, help="监听秒数，0=不限")
    ap.add_argument("--idle-log-sec", type=float, default=5.0)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15.0)
    t0 = time.monotonic()
    print(f"connect {args.host}:{args.port} ...", flush=True)
    sock.connect((args.host, int(args.port)))
    sock.settimeout(0.5)
    print("connected. 请对着板子说话/唤醒后说话 ...", flush=True)

    total = 0
    sessions = 0
    in_sess = False
    last_data = t0
    last_idle_log = t0

    try:
        while args.duration <= 0 or (time.monotonic() - t0) < args.duration:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                now = time.monotonic()
                if in_sess and (now - last_data) >= 1.0:
                    print(
                        f"[pcm] session end total={sess_bytes}B "
                        f"(idle {(now - last_data):.1f}s)",
                        flush=True,
                    )
                    in_sess = False
                    sess_bytes = 0
                if (now - last_idle_log) >= float(args.idle_log_sec):
                    print(
                        f"[idle] cumulative={total}B sessions={sessions} "
                        f"(waiting speech...)",
                        flush=True,
                    )
                    last_idle_log = now
                continue
            if not chunk:
                print("server closed connection", flush=True)
                break
            if not in_sess:
                in_sess = True
                sessions += 1
                sess_bytes = 0
                print(f"[pcm] session #{sessions} start (+{len(chunk)}B)", flush=True)
            sess_bytes += len(chunk)
            total += len(chunk)
            last_data = time.monotonic()
            if sess_bytes <= len(chunk) or sess_bytes % 32000 < len(chunk):
                print(f"  recv +{len(chunk)} session={sess_bytes} total={total}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

    elapsed = time.monotonic() - t0
    print(
        f"done: {total} bytes in {elapsed:.1f}s, sessions={sessions}",
        flush=True,
    )
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
