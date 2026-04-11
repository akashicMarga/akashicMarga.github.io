"""
Real EnCodec reconstruction at different codebook levels.

The bug in the previous version: zeroing codes to index 0 still adds
codebook_entry[0] (a non-zero vector) for each zeroed level — corrupting
the reconstruction with garbage embeddings from all 8 levels.

The fix: use only the first k codebook layers in vq.decode,
so the quantized embedding is exactly sum(embed_i for i in 0..k-1).

EnCodec internals:
  model.encoder(wav)          -> z: [B, D=128, T_frames]
  model.quantizer.vq.decode(
    q_indices: [K, B, T])     -> quantized: [B, D, T]  (sum of k embeddings)
  model.decoder(quantized)    -> audio: [B, 1, T_audio]

The encode path returns codes shaped [B, K, T].
vq.decode expects [K, B, T], so we transpose before slicing.
"""

import torch
import torchaudio
import numpy as np
import wave
import os

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))


def write_wav(filename, samples_np, sr):
    samples_np = np.clip(samples_np, -1.0, 1.0)
    pcm = (samples_np * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def reconstruct_at_level(model, encoded_frames, n_levels):
    """
    Proper level-k reconstruction:
      1. codes [B, K, T] -> transpose -> [K, B, T]
      2. slice first n_levels: [n_levels, B, T]
      3. vq.decode sums exactly n_levels embeddings
      4. pass through GAN decoder
    """
    codes, scale = encoded_frames[0]   # codes: [B, K=8, T]

    # Transpose to [K, B, T] — the format vq.decode iterates over
    codes_kbt = codes.permute(1, 0, 2)        # [K, B, T]
    codes_trimmed = codes_kbt[:n_levels]       # [n_levels, B, T]

    with torch.no_grad():
        # Sum embeddings from only the first n_levels codebooks
        quantized = model.quantizer.vq.decode(codes_trimmed)   # [B, D=128, T]

        # Apply per-frame scale (EnCodec normalises each frame)
        if scale is not None:
            quantized = quantized * scale.view(-1, 1, 1)

        # GAN decoder: latent -> waveform
        audio = model.decoder(quantized)   # [B, 1, T_audio]

    return audio.squeeze().cpu().numpy()


if __name__ == "__main__":
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    print("Loading EnCodec 24kHz model...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)   # 6 kbps = 8 RVQ levels
    model.eval()

    src = os.path.join(AUDIO_DIR, "hindi_original_raw.wav")
    if not os.path.exists(src):
        # fallback to the previously downloaded original
        src = os.path.join(AUDIO_DIR, "hindi_original.wav")
    print(f"Loading {src}...")
    wav, sr = torchaudio.load(src, backend="soundfile")
    print(f"  {wav.shape}, {sr} Hz")

    # Resample to 24kHz mono
    wav_24k = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav_24k = wav_24k.unsqueeze(0)   # [B=1, C=1, T]

    print("Encoding...")
    with torch.no_grad():
        encoded_frames = model.encode(wav_24k)
    codes, scale = encoded_frames[0]
    print(f"  Codes: {codes.shape}  →  [B={codes.shape[0]}, K={codes.shape[1]} levels, T={codes.shape[2]} frames]")

    target_sr = model.sample_rate  # 24000

    for n_levels in [1, 3, 8]:
        print(f"Reconstructing with {n_levels} codebook level(s)...")
        audio_np = reconstruct_at_level(model, encoded_frames, n_levels)

        # Resample 24kHz → 16kHz for smaller files
        audio_t = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        audio_16k = torchaudio.functional.resample(audio_t, target_sr, 16000).squeeze().numpy()

        out = os.path.join(AUDIO_DIR, f"hindi_rvq_level_{n_levels}.wav")
        write_wav(out, audio_16k, 16000)
        print(f"  -> {out}")

    # Save full reference (16kHz) for fair comparison
    ref_np = reconstruct_at_level(model, encoded_frames, 8)
    ref_t = torch.from_numpy(ref_np).unsqueeze(0).unsqueeze(0)
    ref_16k = torchaudio.functional.resample(ref_t, target_sr, 16000).squeeze().numpy()
    write_wav(os.path.join(AUDIO_DIR, "hindi_original.wav"), ref_16k, 16000)
    print("-> hindi_original.wav  (full round-trip reference)")
    print("\nDone.")
