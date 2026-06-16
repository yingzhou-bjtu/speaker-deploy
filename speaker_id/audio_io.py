"""读 16kHz 单声道 wav，不依赖 torch。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_wav_16k_mono(path: str | Path) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if int(sr) != 16000:
        from scipy import signal

        wav = signal.resample_poly(wav, 16000, int(sr)).astype(np.float32, copy=False)
    return np.asarray(wav, dtype=np.float32).reshape(-1)
