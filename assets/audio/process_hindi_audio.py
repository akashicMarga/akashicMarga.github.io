"""
Apply RVQ-level simulation to real Hindi audio (Mann Ki Baat clip).
Same quantization approach as generate_audio_examples.py but on real speech.
"""

import numpy as np
import struct
import sys
import wave
import os

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))


def read_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sr


def write_wav(filename, samples, sr):
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def quantize_to_bits(signal, n_bits):
    levels = 2 ** n_bits
    return np.clip(np.round(signal * (levels / 2)) / (levels / 2), -1.0, 1.0)


def moving_average(signal, window):
    kernel = np.ones(max(1, window)) / max(1, window)
    return np.convolve(signal, kernel, mode='same')


def simulate_rvq_level(signal, sr, level):
    """Simulate quality at different RVQ reconstruction levels."""
    if level == 1:
        # Coarse: 4-bit + heavy LP (simulates single codebook 1 — rough but recognizable)
        q = quantize_to_bits(signal, 4)
        q = moving_average(q, window=int(sr * 0.006))  # 6ms smoothing
        noise = np.random.default_rng(42).normal(0, 0.015, len(q))
        return np.clip(q + noise, -1, 1)
    elif level == 3:
        # Medium: 7-bit + gentle LP
        q = quantize_to_bits(signal, 7)
        q = moving_average(q, window=int(sr * 0.0015))
        return np.clip(q, -1, 1)
    elif level == 8:
        # Full: 12-bit — near lossless
        return np.clip(quantize_to_bits(signal, 12), -1, 1)


if __name__ == "__main__":
    src = os.path.join(AUDIO_DIR, "hindi_original.wav")
    print(f"Reading {src} ...")
    signal, sr = read_wav(src)
    print(f"  {len(signal)/sr:.1f}s  @  {sr}Hz")

    for level in [1, 3, 8]:
        out = simulate_rvq_level(signal, sr, level)
        fname = os.path.join(AUDIO_DIR, f"hindi_rvq_level_{level}.wav")
        write_wav(fname, out, sr)
        print(f"  -> hindi_rvq_level_{level}.wav")

    print("Done.")
