"""ONNX Runtime 声纹嵌入，板端不需要 torch。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio_io import load_wav_16k_mono
from .fbank_cpu import wav_to_fbank


class OnnxSpeakerEmbedder:
    def __init__(self, onnx_path: Path, *, ort_threads: int = 0) -> None:
        import onnxruntime as ort

        p = Path(onnx_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"缺少 ONNX 模型: {p}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if int(ort_threads) > 0:
            opts.intra_op_num_threads = int(ort_threads)
            opts.inter_op_num_threads = int(ort_threads)
        self.onnx_path = p
        self._sess = ort.InferenceSession(
            str(p),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._in_name = self._sess.get_inputs()[0].name

    def embed_wav_path(self, wav_path: str | Path) -> np.ndarray:
        wav = load_wav_16k_mono(wav_path)
        return self.embed_wav_numpy(wav)

    def embed_wav_numpy(self, wav_1d: np.ndarray) -> np.ndarray:
        feat = wav_to_fbank(wav_1d)
        if feat.shape[0] == 0:
            raise ValueError("音频过短，无法提取 FBank")
        x = feat[np.newaxis, :, :].astype(np.float32)
        out = self._sess.run(None, {self._in_name: x})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)
