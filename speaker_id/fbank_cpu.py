"""Kaldi FBank，对齐 3D Speaker，不依赖 torch。"""
from __future__ import annotations

import numpy as np


def wav_to_fbank(wav_1d: np.ndarray, *, n_mels: int = 80, sample_rate: int = 16000) -> np.ndarray:
    import kaldi_native_fbank as knf

    w = np.asarray(wav_1d, dtype=np.float32).reshape(-1)
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_rate)
    opts.frame_opts.dither = 0.0
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.round_to_power_of_two = True
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = int(n_mels)
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = 0.0
    opts.energy_floor = 1.0

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(int(sample_rate), w.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    if n <= 0:
        return np.zeros((0, n_mels), dtype=np.float32)
    frames = [fbank.get_frame(i) for i in range(n)]
    feat = np.asarray(frames, dtype=np.float32)
    feat = feat - feat.mean(axis=0, keepdims=True)
    return feat
