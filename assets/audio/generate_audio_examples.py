"""
Generate synthetic audio examples demonstrating RVQ coarse-to-fine reconstruction.
Uses numpy/scipy to create a speech-like signal, then simulates
progressive quantization at different numbers of codebook levels.
Outputs WAV files for embedding in the blog.
"""

import numpy as np
import struct
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000
DURATION = 2.5  # seconds


def write_wav(filename, samples, sr=SAMPLE_RATE):
    """Write float samples to a 16-bit PCM WAV file."""
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    n_samples = len(pcm)
    with open(filename, 'wb') as f:
        # RIFF header
        data_size = n_samples * 2
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))        # chunk size
        f.write(struct.pack('<H', 1))         # PCM
        f.write(struct.pack('<H', 1))         # mono
        f.write(struct.pack('<I', sr))        # sample rate
        f.write(struct.pack('<I', sr * 2))    # byte rate
        f.write(struct.pack('<H', 2))         # block align
        f.write(struct.pack('<H', 16))        # bits per sample
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm.tobytes())


def generate_speech_like(sr, duration, seed=42):
    """
    Synthesize a speech-like signal:
    - Voiced segments: harmonic series with vibrato (F0 ~150Hz)
    - Fricative noise bursts
    - Amplitude envelope (prosody)
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.linspace(0, duration, n)

    # Slowly varying F0 (natural speech vibrato)
    f0_base = 150.0
    f0 = f0_base + 8 * np.sin(2 * np.pi * 5 * t)  # ±8Hz at 5Hz vibrato rate

    # Voiced component: sum of harmonics
    phase = np.cumsum(2 * np.pi * f0 / sr)
    voiced = np.zeros(n)
    for harmonic in range(1, 12):
        amplitude = 1.0 / harmonic ** 0.8
        voiced += amplitude * np.sin(harmonic * phase)
    voiced /= np.max(np.abs(voiced) + 1e-8)

    # Unvoiced (fricative) noise
    noise = rng.standard_normal(n) * 0.3

    # Segment mask: alternate voiced/unvoiced
    mask = np.ones(n)
    seg_len = int(sr * 0.35)
    for i in range(0, n, seg_len * 2):
        end = min(i + seg_len, n)
        mask[i:end] = 0  # unvoiced here
    voiced_sig = voiced * mask
    noise_sig = noise * (1 - mask)

    # Amplitude envelope (prosody: rises and falls)
    envelope = 0.5 + 0.5 * np.sin(np.pi * t / duration)
    envelope *= np.clip(1 - np.abs(t - duration/2) / (duration/2 + 0.1), 0.2, 1.0)

    signal = (voiced_sig + noise_sig) * envelope
    signal /= np.max(np.abs(signal) + 1e-8) * 1.1
    return signal


def quantize_to_n_bits(signal, n_bits):
    """
    Simulate VQ quantization by scalar quantizing to 2^n_bits levels.
    Each level of RVQ contributes ~log2(N) bits; here we use total bit depth
    as a proxy for how many RVQ levels are used.
    """
    levels = 2 ** n_bits
    quantized = np.round(signal * (levels / 2)) / (levels / 2)
    return np.clip(quantized, -1.0, 1.0)


def apply_smoothing(signal, window_ms, sr):
    """Low-pass filter via simple moving average to simulate coarse codec."""
    window = max(1, int(window_ms * sr / 1000))
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def simulate_rvq_level(signal, sr, level, total_levels=8):
    """
    Simulate the perceptual quality of RVQ reconstruction at `level` codebook levels.

    Level 1 (coarse): heavily quantized + smoothed — speaker identity but rough
    Level 3 (mid):    moderate quality — intelligible, good prosody
    Level 8 (full):   near lossless
    """
    if level == 1:
        # Very coarse: 4-bit equivalent + heavy smoothing
        q = quantize_to_n_bits(signal, 4)
        q = apply_smoothing(q, window_ms=8, sr=sr)
        # Add slight buzz artifact
        t = np.linspace(0, len(signal)/sr, len(signal))
        q += 0.04 * np.sin(2 * np.pi * 200 * t)
        return np.clip(q * 0.85, -1, 1)
    elif level == 3:
        # Medium: 7-bit equivalent + mild smoothing
        q = quantize_to_n_bits(signal, 7)
        q = apply_smoothing(q, window_ms=2, sr=sr)
        return np.clip(q * 0.92, -1, 1)
    elif level == 8:
        # Full: 12-bit equivalent — near original
        q = quantize_to_n_bits(signal, 12)
        return np.clip(q * 0.98, -1, 1)
    else:
        bits = 4 + int(8 * level / total_levels)
        q = quantize_to_n_bits(signal, bits)
        smooth_ms = max(0, 8 - level * 0.8)
        if smooth_ms > 0:
            q = apply_smoothing(q, window_ms=smooth_ms, sr=sr)
        return np.clip(q, -1, 1)


if __name__ == "__main__":
    print("Generating speech-like signal...")
    original = generate_speech_like(SAMPLE_RATE, DURATION)
    write_wav(os.path.join(OUTPUT_DIR, "rvq_original.wav"), original)
    print("  -> rvq_original.wav")

    for level in [1, 3, 8]:
        reconstructed = simulate_rvq_level(original, SAMPLE_RATE, level)
        fname = f"rvq_level_{level}.wav"
        write_wav(os.path.join(OUTPUT_DIR, fname), reconstructed)
        print(f"  -> {fname}  (level {level}/8)")

    print("Done. 4 WAV files generated.")
