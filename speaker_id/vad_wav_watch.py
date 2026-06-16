"""监听 robotmedia 落盘的 vad wav，和 TCP 二选一或并行兜底。"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

_MAX_SEEN_WAV = 2048
_MAX_PENDING_WAV = 256

# 文件名末尾数字是 16kHz 采样点数，不是秒。
_SILERO_SAMPLES_RE = re.compile(r"_(\d+)s\.wav$", re.IGNORECASE)


def silero_vad_samples_in_name(name: str) -> Optional[int]:
    m = _SILERO_SAMPLES_RE.search(name)
    return int(m.group(1)) if m else None

# 16kHz 单声道 PCM16 大约每秒 32000 字节。
_PCM16_BYTES_PER_SEC = 32000


def watch_vad_wav_loop(
    ingest_wav: Optional[Callable[[Path, str], None]] = None,
    *,
    on_vad_end: Optional[Callable[[Path, str], None]] = None,
    watch_dir: Path,
    pattern: str = "silero_vad_*.wav",
    poll_sec: float = 0.5,
    stable_sec: float = 0.3,
    skip_existing: bool = True,
    min_duration_sec: float = 0.2,
    max_duration_sec: float = 15.0,
) -> None:
    """新 wav 写稳后回调。ingest_wav 读文件做声纹，on_vad_end 只作段结束信号。启动时可跳过已有文件。"""
    if ingest_wav is None and on_vad_end is None:
        raise ValueError("ingest_wav 与 on_vad_end 至少提供一个")
    watch_dir = Path(watch_dir)
    seen: OrderedDict[str, None] = OrderedDict()
    pending_size: dict[str, int] = {}
    pending_at: dict[str, float] = {}

    def _mark_seen(key: str) -> None:
        seen[key] = None
        while len(seen) > _MAX_SEEN_WAV:
            seen.popitem(last=False)
    min_bytes = int(_PCM16_BYTES_PER_SEC * float(min_duration_sec))
    max_samples = int(16000 * float(max_duration_sec)) if float(max_duration_sec) > 0 else 0

    if skip_existing and watch_dir.is_dir():
        for p in watch_dir.glob(pattern):
            if p.is_file():
                _mark_seen(str(p.resolve()))
        tag = "signal" if on_vad_end and ingest_wav is None else "ingest"
        print(
            f"[wav-watch] dir={watch_dir} mode={tag} skip {len(seen)} existing; "
            f"only new VAD wav (>={min_duration_sec}s, <={max_duration_sec}s)",
            flush=True,
        )
    else:
        print(f"[wav-watch] dir={watch_dir} pattern={pattern}", flush=True)

    def _prune_pending(active: set[str]) -> None:
        if len(pending_size) <= _MAX_PENDING_WAV:
            return
        for key in list(pending_size.keys()):
            if key not in active:
                pending_size.pop(key, None)
                pending_at.pop(key, None)

    while True:
        try:
            active_keys: set[str] = set()
            if watch_dir.is_dir():
                for p in sorted(watch_dir.glob(pattern)):
                    if not p.is_file():
                        continue
                    key = str(p.resolve())
                    active_keys.add(key)
                    if key in seen:
                        continue
                    sz = int(p.stat().st_size)
                    if sz < min_bytes:
                        _mark_seen(key)
                        continue
                    if max_samples > 0:
                        ns = silero_vad_samples_in_name(p.name)
                        if ns is not None and ns > max_samples:
                            _mark_seen(key)
                            print(
                                f"[wav-watch] skip long VAD {p.name} "
                                f"({ns / 16000:.1f}s > {max_duration_sec}s)",
                                flush=True,
                            )
                            continue
                    if pending_size.get(key) != sz:
                        pending_size[key] = sz
                        pending_at[key] = time.monotonic()
                        continue
                    if time.monotonic() - pending_at.get(key, 0.0) < stable_sec:
                        continue
                    _mark_seen(key)
                    pending_size.pop(key, None)
                    pending_at.pop(key, None)
                    if on_vad_end is not None:
                        on_vad_end(p, p.name)
                    if ingest_wav is not None:
                        ingest_wav(p, p.name)
                _prune_pending(active_keys)
        except OSError as e:
            print(f"[wav-watch] {e!r}", flush=True)
        time.sleep(poll_sec)
