"""声纹部署用的阶段计时和本进程资源采样。"""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _read_proc_rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
                if line.startswith("VmHWM:"):
                    pass
    except OSError:
        return None
    return None


def _read_proc_hwm_mb() -> Optional[float]:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except OSError:
        return None
    return None


def _rusage_mb() -> Dict[str, float]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux 上 ru_maxrss 单位是 KB，macOS 上是字节。
    rss = float(ru.ru_maxrss)
    if sys.platform == "darwin":
        rss /= 1024.0 * 1024.0
    else:
        rss /= 1024.0
    return {
        "cpu_user_s": float(ru.ru_utime),
        "cpu_sys_s": float(ru.ru_stime),
        "max_rss_mb_rusage": rss,
    }


@dataclass
class Sample:
    rss_mb: Optional[float] = None
    hwm_mb: Optional[float] = None
    tracemalloc_mb: Optional[float] = None


@dataclass
class PhaseRecord:
    name: str
    wall_s: float
    cpu_user_s: float
    cpu_sys_s: float
    rss_mb_start: Optional[float]
    rss_mb_end: Optional[float]
    rss_mb_delta: Optional[float]
    hwm_mb_end: Optional[float]
    tracemalloc_peak_mb: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProbeSession:
    """累积各阶段探测结果。"""

    def __init__(self) -> None:
        self.phases: List[PhaseRecord] = []
        self._peak_rss_mb: float = 0.0
        self.meta: Dict[str, Any] = {}

    def _touch_peak(self) -> None:
        r = _read_proc_rss_mb()
        if r is not None:
            self._peak_rss_mb = max(self._peak_rss_mb, r)

    @contextmanager
    def phase(self, name: str, **extra: Any) -> Iterator[Dict[str, Any]]:
        self._touch_peak()
        bag: Dict[str, Any] = {}
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        t0 = time.perf_counter()
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        rss0 = _read_proc_rss_mb()
        try:
            yield bag
        finally:
            t1 = time.perf_counter()
            ru1 = resource.getrusage(resource.RUSAGE_SELF)
            rss1 = _read_proc_rss_mb()
            hwm = _read_proc_hwm_mb()
            self._touch_peak()
            tm_peak = None
            if tracemalloc.is_tracing():
                _cur, peak = tracemalloc.get_traced_memory()
                tm_peak = peak / (1024.0 * 1024.0)
            rec = PhaseRecord(
                name=name,
                wall_s=round(t1 - t0, 6),
                cpu_user_s=round(ru1.ru_utime - ru0.ru_utime, 6),
                cpu_sys_s=round(ru1.ru_stime - ru0.ru_stime, 6),
                rss_mb_start=round(rss0, 3) if rss0 is not None else None,
                rss_mb_end=round(rss1, 3) if rss1 is not None else None,
                rss_mb_delta=(
                    round(rss1 - rss0, 3)
                    if rss0 is not None and rss1 is not None
                    else None
                ),
                hwm_mb_end=round(hwm, 3) if hwm is not None else None,
                tracemalloc_peak_mb=round(tm_peak, 3) if tm_peak is not None else None,
                extra={**extra, **bag},
            )
            self.phases.append(rec)

    def host_info(self) -> Dict[str, Any]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
        }

    def summary(self) -> Dict[str, Any]:
        ru = _rusage_mb()
        by_name = {p.name: p.to_dict() for p in self.phases}
        mount = by_name.get("mount_pipeline")
        return {
            "host": self.host_info(),
            "meta": self.meta,
            "process_totals": {
                **ru,
                "peak_rss_mb_sampled": round(self._peak_rss_mb, 3),
            },
            "phases": [p.to_dict() for p in self.phases],
            "highlights_s": {
                "import_deps": (by_name.get("import_deps") or {}).get("wall_s"),
                "mount_pipeline": (mount or {}).get("wall_s"),
                "embed_only": (by_name.get("embed_only") or {}).get("wall_s"),
                "first_ingest": (by_name.get("first_ingest") or {}).get("wall_s"),
                "stream_ingest_total": (by_name.get("stream_ingest") or {}).get("wall_s"),
            },
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def print_human(self) -> None:
        s = self.summary()
        hi = s.get("highlights_s") or {}
        pt = s.get("process_totals") or {}
        print("=== 声纹 deploy 探测报告 ===")
        print(f"PID {s['host']['pid']}  {s['host']['platform']}")
        print()
        print("【耗时摘要】")
        for k, label in (
            ("import_deps", "依赖导入"),
            ("mount_pipeline", "挂载 pipeline（含 ONNX）"),
            ("embed_only", "单条嵌入（ONNX，无聚类）"),
            ("first_ingest", "首条 ingest（嵌入+聚类）"),
            ("stream_ingest_total", "流式 ingest 合计"),
        ):
            v = hi.get(k)
            if v is not None:
                print(f"  {label}: {v:.3f} s")
        print()
        print("【进程资源（本程序）】")
        if pt.get("max_rss_mb_rusage"):
            print(f"  峰值 RSS (getrusage): {pt['max_rss_mb_rusage']:.1f} MB")
        if pt.get("peak_rss_mb_sampled"):
            print(f"  峰值 RSS (/proc 采样): {pt['peak_rss_mb_sampled']:.1f} MB")
        print(f"  累计 CPU 用户态: {pt.get('cpu_user_s', 0):.2f} s")
        print(f"  累计 CPU 内核态: {pt.get('cpu_sys_s', 0):.2f} s")
        print()
        print("【各阶段明细】")
        for p in self.phases:
            line = (
                f"  {p.name}: wall={p.wall_s:.3f}s "
                f"cpu={p.cpu_user_s + p.cpu_sys_s:.3f}s"
            )
            if p.rss_mb_end is not None:
                line += f" RSS={p.rss_mb_end:.1f}MB"
            if p.rss_mb_delta is not None:
                line += f" (Δ{p.rss_mb_delta:+.1f})"
            print(line)
            if p.extra:
                for ek, ev in p.extra.items():
                    if ek.startswith("_"):
                        continue
                    if ek == "per_file" and isinstance(ev, list) and len(ev) > 5:
                        print(f"      {ek}: <{len(ev)} 条，见 JSON>")
                        continue
                    print(f"      {ek}: {ev}")
