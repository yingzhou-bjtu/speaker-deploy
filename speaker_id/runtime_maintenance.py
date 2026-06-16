"""进程内自动裁剪日志和 vad wav，不用人工清理。"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

_trim_lock = threading.Lock()


def _format_ts(t: Optional[float] = None) -> str:
    t = time.time() if t is None else t
    ms = int((t % 1.0) * 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{ms:03d}"


def trim_log_file(
    log_path: Path,
    *,
    max_bytes: int,
    keep_bytes: int,
) -> bool:
    """日志太大就保留尾部并重写。"""
    log_path = Path(log_path)
    max_bytes = max(64 * 1024, int(max_bytes))
    keep_bytes = max(32 * 1024, min(int(keep_bytes), max_bytes))
    try:
        size = log_path.stat().st_size
    except OSError:
        return False
    if size <= max_bytes:
        return False
    with _trim_lock:
        try:
            size = log_path.stat().st_size
        except OSError:
            return False
        if size <= max_bytes:
            return False
        try:
            with open(log_path, "rb") as f:
                if size > keep_bytes:
                    f.seek(-keep_bytes, os.SEEK_END)
                tail = f.read()
        except OSError:
            return False
        marker = (
            f"[maint {_format_ts()} log auto-trim "
            f"keep={keep_bytes}B dropped={max(0, size - len(tail))}B]\n"
        ).encode("utf-8", errors="replace")
        try:
            fd = os.open(str(log_path), os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(fd, marker)
                if tail:
                    os.write(fd, tail)
            finally:
                os.close(fd)
        except OSError:
            return False
        for stream_fd in (1, 2):
            try:
                os.lseek(stream_fd, 0, os.SEEK_END)
            except OSError:
                pass
    return True


def prune_vad_wav_dir(
    watch_dir: Path,
    *,
    pattern: str = "silero_vad_*.wav",
    max_files: int,
    keep_files: int,
    min_age_sec: float,
) -> int:
    """wav 太多就删最旧且已写完的文件。"""
    watch_dir = Path(watch_dir)
    max_files = int(max_files)
    keep_files = int(keep_files)
    if max_files <= 0 or keep_files <= 0 or keep_files >= max_files:
        return 0
    if not watch_dir.is_dir():
        return 0
    try:
        files = sorted(
            (p for p in watch_dir.glob(pattern) if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return 0
    if len(files) <= max_files:
        return 0
    drop_n = len(files) - keep_files
    if drop_n <= 0:
        return 0
    cutoff = time.time() - max(60.0, float(min_age_sec))
    removed = 0
    for p in files[:drop_n]:
        try:
            if p.stat().st_mtime > cutoff:
                continue
            p.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def run_maintenance_once(
    *,
    log_path: Optional[Path],
    log_max_bytes: int,
    log_keep_bytes: int,
    wav_dir: Optional[Path],
    wav_max_files: int,
    wav_keep_files: int,
    wav_min_age_sec: float,
) -> None:
    if log_path is not None:
        if trim_log_file(log_path, max_bytes=log_max_bytes, keep_bytes=log_keep_bytes):
            print(
                f"[maint] log trimmed ({log_path}, cap={log_max_bytes // 1024}KB)",
                flush=True,
            )
    if wav_dir is not None and wav_max_files > 0:
        n = prune_vad_wav_dir(
            wav_dir,
            max_files=wav_max_files,
            keep_files=wav_keep_files,
            min_age_sec=wav_min_age_sec,
        )
        if n > 0:
            print(
                f"[maint] vad wav pruned {n} under {wav_dir} "
                f"(keep<={wav_keep_files})",
                flush=True,
            )


def maintenance_loop(
    *,
    log_path: Optional[Path],
    log_max_bytes: int,
    log_keep_bytes: int,
    wav_dir: Optional[Path],
    wav_max_files: int,
    wav_keep_files: int,
    wav_min_age_sec: float,
    interval_sec: float,
) -> None:
    interval_sec = max(30.0, float(interval_sec))
    run_maintenance_once(
        log_path=log_path,
        log_max_bytes=log_max_bytes,
        log_keep_bytes=log_keep_bytes,
        wav_dir=wav_dir,
        wav_max_files=wav_max_files,
        wav_keep_files=wav_keep_files,
        wav_min_age_sec=wav_min_age_sec,
    )
    while True:
        time.sleep(interval_sec)
        try:
            run_maintenance_once(
                log_path=log_path,
                log_max_bytes=log_max_bytes,
                log_keep_bytes=log_keep_bytes,
                wav_dir=wav_dir,
                wav_max_files=wav_max_files,
                wav_keep_files=wav_keep_files,
                wav_min_age_sec=wav_min_age_sec,
            )
        except Exception as e:
            print(f"[maint] {type(e).__name__}: {e}", flush=True)


def start_runtime_maintenance(
    *,
    log_path: Optional[Path] = None,
    wav_dir: Optional[Path] = None,
    interval_sec: float = 120.0,
    log_max_mb: float = 8.0,
    log_keep_mb: float = 4.0,
    wav_max_files: int = 2000,
    wav_keep_files: int = 1000,
    wav_min_age_sec: float = 600.0,
) -> threading.Thread:
    """启动后台维护线程，启动时先跑一轮。"""
    log_max_bytes = int(max(1.0, float(log_max_mb)) * 1024 * 1024)
    log_keep_bytes = int(max(0.5, float(log_keep_mb)) * 1024 * 1024)
    if log_keep_bytes >= log_max_bytes:
        log_keep_bytes = max(log_max_bytes // 2, 512 * 1024)
    t = threading.Thread(
        target=maintenance_loop,
        kwargs={
            "log_path": log_path,
            "log_max_bytes": log_max_bytes,
            "log_keep_bytes": log_keep_bytes,
            "wav_dir": wav_dir,
            "wav_max_files": int(wav_max_files),
            "wav_keep_files": int(wav_keep_files),
            "wav_min_age_sec": float(wav_min_age_sec),
            "interval_sec": float(interval_sec),
        },
        name="sv-maint",
        daemon=True,
    )
    t.start()
    return t
