"""单线程 ingest 队列，PCM 可异步提交不阻塞收流。"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np


def _default_queue_maxsize() -> int:
    from .pipeline_factory import default_pipeline_queue_maxsize

    return default_pipeline_queue_maxsize()


@dataclass
class _Job:
    kind: str  # audio 或 wav
    wav: Optional[np.ndarray] = None
    path: Optional[Path] = None
    wav_key: Optional[str] = None
    client_ref: Any = None
    on_done: Optional[Callable[[Any], None]] = None
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class PipelineWorker:
    def __init__(self, pipe: Any, *, queue_maxsize: Optional[int] = None) -> None:
        self._pipe = pipe
        cap = _default_queue_maxsize() if queue_maxsize is None else max(1, int(queue_maxsize))
        self._q: queue.Queue[Optional[_Job]] = queue.Queue(maxsize=cap)
        self._t = threading.Thread(target=self._loop, name="sv-ingest-worker", daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                break
            try:
                if job.kind == "audio":
                    assert job.wav is not None
                    job.result = self._pipe.ingest_audio(
                        job.wav,
                        wav_key=job.wav_key,
                        client_ref=job.client_ref,
                    )
                else:
                    assert job.path is not None
                    job.result = self._pipe.ingest_wav(
                        job.path,
                        wav_key=job.wav_key,
                        client_ref=job.client_ref,
                    )
            except BaseException as e:
                job.error = e
            if job.error is None and job.on_done is not None:
                job.on_done(job.result)
            job.done.set()

    def submit_audio(
        self,
        wav_16k: np.ndarray,
        *,
        wav_key: Optional[str] = None,
        client_ref: Any = None,
        on_done: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._q.put(
            _Job(
                "audio",
                wav=np.asarray(wav_16k),
                wav_key=wav_key,
                client_ref=client_ref,
                on_done=on_done,
            )
        )

    def ingest_wav_path(
        self,
        path: Union[str, Path],
        *,
        wav_key: Optional[str] = None,
        client_ref: Any = None,
    ) -> Any:
        job = _Job("wav", path=Path(path), wav_key=wav_key, client_ref=client_ref)
        self._q.put(job)
        job.done.wait()
        if job.error:
            raise job.error
        return job.result
