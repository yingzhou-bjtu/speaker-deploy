"""16kHz 单声道 PCM16 和 float32 wav 互转。"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE_HZ = 16000
BYTES_PER_SAMPLE = 2
DEFAULT_FRAME_BYTES = 3200  # 16kHz 单声道下约 100 毫秒一帧。


def pcm16le_bytes_to_float32(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("empty PCM")
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def segment_byte_length(segment_sec: float, *, sample_rate: int = SAMPLE_RATE_HZ) -> int:
    return int(float(segment_sec) * int(sample_rate) * BYTES_PER_SAMPLE)
