---
layout: post
title: "From Spectrograms to Speech Codecs: A Deep Dive"
date: 2026-04-11
categories: [speech, deep-learning, audio]
tags: [codec, RVQ, spectrogram, TTS, voice-cloning, EnCodec, Mimi, SNAC, Orpheus, Chatterbox, speech-LM]
description: "A ground-up explanation of how audio is represented, compressed, and tokenized for modern speech AI — with animations, real audio examples, and an interactive bitrate calculator."
---

Think about the last time you listened to a voice note on WhatsApp, or joined a Zoom call that started dropping quality as your WiFi struggled. That degraded, garbled version of someone's voice — that's compression happening in real time. The system is throwing away information, but trying very hard to throw away *the right* information.

This post builds the full story of how that works, from first principles — and then goes further into how modern speech LMs like VALL-E, Orpheus, Chatterbox, and Fish Speech represent audio as sequences of integers that a language model can predict, exactly like next-word prediction in text.

The central thread throughout: **audio compression and image compression are solving the same problem with the same mathematical tools.** JPEG and MP3 were invented independently and are usually taught separately, but they're doing essentially the same thing. I'll draw that parallel at every step.

> **Prefer visual explanations?** The [Kyutai codec explainer](https://kyutai.org/codec-explainer) was instrumental in building my own understanding — highly recommended alongside this post. For Fourier fundamentals, [3Blue1Brown's visual introduction](https://www.youtube.com/watch?v=spUNpyF58BY) is the best starting point on the internet.

---

**Contents**

1. [The problem: raw audio is enormous](#1-the-problem-raw-audio-is-enormous)
2. [Seeing sound: the spectrogram](#2-seeing-sound-the-spectrogram)
3. [What speech actually encodes: the source-filter model](#3-what-speech-actually-encodes-the-source-filter-model)
4. [Classical compression: JPEG and MP3](#4-classical-compression-jpeg-and-mp3)
5. [Learned compression: from images to audio](#5-learned-compression-from-images-to-audio)
6. [Vector Quantization: the discrete bottleneck](#6-vector-quantization-the-discrete-bottleneck)
7. [Residual Vector Quantization: stacking quantizers](#7-residual-vector-quantization-stacking-quantizers)
8. [The GAN decoder: why it sounds good](#8-the-gan-decoder-why-it-sounds-good)
9. [Neural codec architectures: the evolution](#9-neural-codec-architectures-the-evolution)
10. [Codec comparison](#10-codec-comparison)
11. [Which codec should I actually use?](#11-which-codec-should-i-actually-use)
12. [Mimi: why flat RVQ wins for streaming](#12-mimi-why-flat-rvq-wins-for-streaming)
13. [SNAC: multi-scale RVQ for hierarchical generation](#13-snac-multi-scale-rvq-for-hierarchical-generation)
14. [Codec-based TTS: the full pipeline](#14-codec-based-tts-the-full-pipeline)
15. [What codecs encode: the coarse-to-fine hierarchy](#15-what-codecs-encode-the-coarse-to-fine-hierarchy)
16. [Open questions](#16-open-questions)

---

## 1. The problem: raw audio is enormous

Before we can talk about compression, it helps to understand just how much data an uncompressed audio signal actually is.

A single second of 24kHz float32 mono audio — the kind used in modern speech AI:

```
24,000 samples × 4 bytes = 96,000 bytes ≈ 768 kbps
```

A 10-minute podcast at this rate is ~57.6 MB. A one-hour lecture is ~346 MB. You cannot stream this efficiently, you cannot feed it to a language model (the context window would span millions of tokens), and you cannot train on it at scale without burning enormous compute.

The same problem exists for images. A raw 512×512 RGB image is 786 KB. JPEG compresses it to ~50 KB — a 15× reduction — with barely perceptible quality loss. **The question is: can we do something equivalent for audio, and can we do it in a way that produces a representation useful for generative models?**

To answer that, we first need to understand what information is actually *in* a sound.

---

## 2. Seeing sound: the spectrogram

Here is something worth pausing on: two people can say the word "hello" at exactly the same pitch, same speed, same volume — and you will *instantly* know which person said it. You've never thought about how you do that. The answer lies in the structure of the sound wave, and the spectrogram is how we make that structure visible.

### From waveform to image

A raw audio signal is a 1D pressure wave — one amplitude value per timestep. The problem is that this representation mixes together all frequencies at once. A single sample value of 0.3 tells you nothing about whether it came from a low bass note, a high-pitched consonant, or room noise. It's all collapsed into one number.

The **Short-Time Fourier Transform (STFT)** fixes this by analyzing the signal through a sliding window:

<figure>
  <img src="/assets/images/codec-animations/waveform_to_spectrogram.gif"
       alt="Waveform sliced into overlapping windows, each FFT'd and stacked into a spectrogram"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    The waveform is sliced into overlapping 25ms windows. Each window gets its own FFT — turning "what is the amplitude right now?" into "what frequencies are present right now?" Stack those FFT columns side by side and you get the spectrogram: time on the x-axis, frequency on the y-axis, brightness = energy.
  </figcaption>
</figure>

1. Cut the waveform into overlapping frames (typically 25ms window, 10ms hop)
2. Apply a **Hann window** to each frame — a smooth fade-in and fade-out that prevents artifacts at the frame edges
3. Take the FFT of each frame to get the frequency content at that moment

Result: a 2D matrix of shape `[n_frames, n_fft/2 + 1]` — time across one axis, frequency across the other. This is the **spectrogram**.

```python
import torch, torchaudio

waveform, sr = torchaudio.load("speech.wav")
stft = torch.stft(
    waveform.squeeze(),
    n_fft=1024,
    hop_length=160,    # 10ms hop at 16kHz
    win_length=400,    # 25ms window
    window=torch.hann_window(400),
    return_complex=True
)
magnitude = stft.abs()   # discard phase, keep energy
```

There's an unavoidable tradeoff baked in here: a larger `n_fft` gives you finer frequency resolution but coarser time resolution, and vice versa. This isn't a limitation of our algorithm — it's the **Heisenberg uncertainty principle applied to signals**. You cannot simultaneously know exactly *when* and exactly *what frequency* a sound occurred. The window size is a choice about which you care about more.

> *If this is the first time you're seeing the Fourier transform, [this 3Blue1Brown video](https://www.youtube.com/watch?v=spUNpyF58BY) builds the intuition visually before the math — worth 20 minutes before going further.*

**The image parallel is exact:** a spectrogram IS a 2D image. Frequency is the Y-axis, time is the X-axis, pixel intensity is energy. Every tool we have for image processing — 2D CNNs, spatial attention, image compression — applies directly.

### The mel scale: aligning with human hearing

Look at a raw STFT spectrogram and you'll notice most of the interesting action happens in the lower frequencies. The top half of the frequency axis — where most pixels live — is mostly silence for speech. That's because the FFT distributes its frequency bins linearly, but human hearing is logarithmic.

The difference between 100Hz and 200Hz sounds just as large as the difference between 1000Hz and 2000Hz — even though the second gap is 10× wider. The **mel scale** maps Hz to a perceptually uniform space:

```
mel(f) = 2595 × log₁₀(1 + f/700)
```

<figure>
  <img src="/assets/images/codec-animations/mel_filterbank.gif"
       alt="Linearly spaced vs mel-spaced triangular filters on the frequency axis"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Linear filters (top) waste resolution at high frequencies. Mel-spaced filters (bottom) are dense where speech energy lives — below 1kHz where vowels and fundamental frequency reside — and progressively coarser at higher frequencies where speech carries less perceptual information.
  </figcaption>
</figure>

A mel filterbank applies 80 triangular bandpass filters, spaced uniformly on the mel scale, to the STFT magnitude:

```python
mel_filterbank = torchaudio.functional.melscale_fbanks(
    n_freqs=513, f_min=80, f_max=8000, n_mels=80, sample_rate=16000
)
mel_spec = mel_filterbank @ magnitude       # [80, n_frames]
log_mel_spec = torch.log(mel_spec + 1e-8)  # compress dynamic range
```

The final log compression isn't just a detail — it matters enormously. Speech amplitude varies by 40–60dB between a whisper and a shout. Without log compression, loud sounds dominate the representation and quiet sounds (which carry plenty of linguistic information — think the difference between /s/ and /f/) get squashed to near-zero values. Log compression gives every part of the dynamic range fair representation, exactly like gamma correction does in image processing.

**The image parallel:** The mel filterbank is structurally identical to the DCT basis in JPEG compression. Both are linear projections that concentrate signal energy into fewer, more meaningful coefficients — and both are justified by the perceptual characteristics of their respective sensory system.

### Timbre: your voice's fingerprint in the spectrogram

Now we can answer the question from the beginning of this section — why you can instantly identify who said "hello."

The **spectral envelope** is the smooth curve that traces the peaks of the harmonic series in a spectrogram. It's shaped by your vocal tract: the length of your throat, the size of your mouth, the shape of your nasal cavity. These are anatomically unique to you. Two people saying the same word at the same pitch will produce identical harmonic spacing (same F0), but their spectral envelopes will be different — and that difference is what we perceive as **timbre**.

Here's why this matters for the rest of this post: when a neural codec encodes audio, the spectral envelope — your timbre — is exactly what the first codebook level learns to represent. More on that when we get to the hierarchy of what codebooks encode.

---

## 3. What speech actually encodes: the source-filter model

The spectrogram shows us *what* is in speech — but understanding *why* it looks the way it does requires one more model. And this model turns out to be the foundation of everything from voice conversion to speaker cloning.

Speech is the product of two independent physical processes multiplied together:

<figure>
  <img src="/assets/images/codec-animations/source_filter_model.gif"
       alt="Source-filter model: glottis pulse train × vocal tract filter = vowel spectrum"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    The glottis (source) produces a harmonic series at F0 — regular pulses whose spacing determines pitch. The vocal tract (filter) is a resonant tube whose shape determines which harmonics get amplified. The output is their product: a vowel. Change F0 → same vowel, different pitch. Change the filter → same pitch, different vowel.
  </figcaption>
</figure>

**The source** is your glottis — the vocal folds vibrating at your fundamental frequency F0. This produces a harmonic series: F0, 2F0, 3F0, 4F0, ... The rate of vibration encodes pitch. Male voices: ~85–180Hz. Female: ~165–255Hz. Children: ~250–400Hz.

**The filter** is your vocal tract — the roughly 17cm tube from your glottis to your lips, shaped by your tongue, jaw, and lips. This tube resonates at specific frequencies called **formants** (F1, F2, F3). F1 correlates with jaw height (low F1 = mouth closed like /i/, high F1 = mouth open like /a/). F2 correlates with tongue position (low F2 = tongue back like /u/, high F2 = tongue forward like /i/). The pattern of formants is what distinguishes every vowel from every other vowel.

Together, source × filter = what you hear. And here is the critical insight:

> *The source determines what phoneme is being spoken and at what pitch. The filter determines who is speaking. They are independent — which means they can, in principle, be separated.*

This is the mathematical foundation of voice cloning and voice conversion. If you can separate the source sequence (linguistic content, prosody) from the filter representation (speaker identity), you can recombine them: take the *what* from one person, the *who* from another. Every modern voice conversion system is trying to do exactly this separation, and neural codecs are increasingly being designed with this decomposition in mind.

---

## 4. Classical compression: JPEG and MP3

So we understand what speech looks like and what's in it. Now: how do we shrink it?

The first answer, developed through the 1990s, was to hand-engineer compression schemes around the known limits of human perception.

### JPEG (images)

1. Convert RGB to YCbCr — separating luminance (brightness) from color, because our eyes are far more sensitive to luminance detail
2. Divide into 8×8 pixel blocks
3. Apply the 2D Discrete Cosine Transform (DCT) to each block — this converts pixel values into frequency coefficients, concentrating most of the image energy into a few low-frequency terms
4. **Quantize**: divide each coefficient by a number from a quantization table — larger divisors for high-frequency coefficients (fine textures), smaller for low-frequency (overall brightness and color)
5. Entropy-code the result

The key insight is in step 4. Most of the information that your visual system actually uses is in the low-frequency DCT coefficients — broad color areas, large edges. High-frequency coefficients (fine texture, sharp detail) can be very coarsely quantized, sometimes to zero, with little perceptible effect.

### MP3 (audio)

MP3 applies the same logic but to audio's perceptual structure:

1. Apply a polyphase filterbank (conceptually similar to a mel filterbank) to identify frequency sub-bands
2. Model **psychoacoustic masking**: a loud sound masks quieter sounds at nearby frequencies — so those quieter sounds can be quantized more coarsely without you noticing
3. Allocate more bits to unmasked frequencies, fewer to masked ones
4. Quantize and entropy-code

**Both schemes are exploiting the same fundamental truth**: not all information is equally perceptible, so you don't need to preserve it all equally. The mel scale was literally designed to model the masking curves that MP3 uses.

The limitation that killed both schemes for AI purposes is that they're **handcrafted**. The quantization tables in JPEG and the masking model in MP3 were fixed by committee, optimized for perceptual quality as judged by humans. They cannot adapt to downstream tasks. A spectrogram compressed for visual quality is not necessarily optimal as input for ASR. And neither scheme produces discrete tokens that a language model can predict.

---

## 5. Learned compression: from images to audio

The shift from handcrafted to learned compression happened gradually in the 2010s, driven by deep learning's success on images.

### The image side: VAEs and latent spaces

Stable Diffusion's VAE compresses a 512×512×3 image (786KB) into a 64×64×4 latent tensor (~65KB) — a 12× compression ratio. The encoder is a convolutional network that learns, from data, which features matter. The decoder inverts it. The latent space is continuous and smooth — nearby latent vectors decode to visually similar images.

The key benefit for generative models: **diffusion over 64×64×4 is vastly cheaper than over 512×512×3**. The VAE does the heavy lifting of capturing low-level pixel structure; the diffusion model only needs to learn the high-level distribution of latent codes.

The same logic applies to audio, with one critical difference that changes the entire architecture.

### The discrete bottleneck

Language models predict discrete tokens. Text is discrete — characters, subword tokens, integers. The entire transformer machinery — the embedding tables, the softmax output head, the cross-entropy loss — is built around discrete sequences.

A continuous VAE latent doesn't give you this. You can't run `argmax(softmax(logits))` over a continuous space. To train an LLM to generate audio autoregressively — predicting token t+1 given tokens 1...t — you need audio to be a sequence of integers.

This is the core problem that neural audio codecs solve: **learn a compressed representation of audio that is both discrete (integers, for language models) and perceptually high-quality (reconstructable to good audio, via a learned decoder).**

The mathematical tool that makes this possible is Vector Quantization.

---

## 6. Vector Quantization: the discrete bottleneck

<figure>
  <img src="/assets/images/codec-animations/vector_quantization.gif"
       alt="Continuous encoder outputs snap to nearest codebook entry becoming integer indices"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Each continuous encoder output (white dot) computes its distance to every codebook entry, then snaps to the nearest one — becoming a single integer index k. The image parallel: GIF palette quantization, where each of 256 colors is assigned an index and every pixel becomes that index.
  </figcaption>
</figure>

The idea is simple: maintain a lookup table (a **codebook**) of N vectors. When the encoder outputs a continuous vector z, find the nearest entry in the codebook and replace z with that entry's index k. The decoder receives the entry at index k and reconstructs from it.

A continuous d-dimensional vector becomes a single integer 0 to N-1. That's the discrete bottleneck.

### Step by step: how one frame gets quantized

<figure>
  <img src="/assets/images/codec-animations/rvq_deep_dive.gif"
       alt="Full VQ pipeline: encoder → distance calc → argmin → residual → STE gradient → EMA update"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Every step in sequence: encoder outputs z → L2 distance to all N codebook entries → argmin selects nearest → that entry's index k is the code → quantization error (residual) is passed to the next level → the gradient problem at argmin → the straight-through estimator fix → EMA updates the codebook entries.
  </figcaption>
</figure>

**Step 1 — Encoder outputs z**, a continuous d-dimensional vector (d=128 in EnCodec).

**Step 2 — Codebook** `E ∈ ℝ^{N×d}`: N vectors, randomly initialized at the start of training.

**Step 3 — Find the nearest entry** by L2 distance:

```python
distances = torch.cdist(z.unsqueeze(0), codebook)   # [1, N]
k = distances.argmin()        # integer index 0..N-1
z_q = codebook[k]             # the quantized vector
```

**Step 4 — Residual**: `r = z - z_q` is the quantization error — the information that wasn't captured by this codebook entry. This residual is passed to the next level.

**Step 5 — The gradient problem.** To train the encoder end-to-end, gradients from the decoder's reconstruction loss need to flow back through the quantization step. But `argmin` is piecewise constant — its gradient is zero almost everywhere. Standard backprop stops cold.

**Fix: the Straight-Through Estimator (STE)**

```python
# Forward pass: use the quantized value (mathematically correct)
z_q = codebook[k]

# Backward pass: pretend the quantization never happened
z_q_st = z + (z_q - z).detach()
# .detach() prevents gradient from flowing through (z_q - z)
# so ∂(z_q_st)/∂z = 1 — gradient passes through as identity
```

This is an approximation — we're lying to the backward pass. But it works well in practice because once the codebook is reasonably trained, `‖z - z_q‖` is small, meaning the encoder's true gradient and the approximated gradient point in nearly the same direction.

### How codebooks learn: EMA updates

<figure>
  <img src="/assets/images/codec-animations/codebook_ema.gif"
       alt="EMA: codebook entries drift toward data cluster centroids, dead entries reinitialized"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Codebook entries are not updated by gradient descent — they drift via EMA toward the centroid of all encoder outputs assigned to them. Entries that never win any assignments (codebook collapse) are detected and reinitialized to random encoder outputs from the current batch.
  </figcaption>
</figure>

The codebook entries are updated via **Exponential Moving Average** — an online k-means algorithm running in parallel with encoder training:

```python
for i in range(N):
    assigned = z_batch[assignments == i]
    if len(assigned) > 0:
        # Move entry toward centroid of assigned vectors
        count_i = decay * count_i + (1 - decay) * len(assigned)
        sum_i   = decay * sum_i   + (1 - decay) * assigned.sum(0)
        codebook[i] = sum_i / count_i
    else:
        # Codebook collapse: this entry is never nearest to anything
        # Reinitialize to a random encoder output from this batch
        codebook[i] = z_batch[torch.randint(len(z_batch), (1,))]
```

Over thousands of training steps, codebook entries drift toward the natural clusters in encoder output space. Similar phonemes end up near each other. Similar voice qualities cluster together. The codebook self-organizes to tile the manifold of speech — even though it started random.

**Codebook collapse** is the main failure mode to watch for. If an entry is never the nearest neighbor to any encoder output, it stops updating. It's effectively dead. The reinitializaton trick above — moving dead entries to random batch vectors — is standard practice and essential for keeping the full codebook active.

---

## 7. Residual Vector Quantization: stacking quantizers

One codebook with N=1024 entries can only represent 1024 distinct vectors. Speech, with its enormous variation in phonemes × speakers × prosody × recording conditions, requires far more representational capacity than that. Reconstruction from a single VQ is too coarse to sound natural.

**Residual VQ (RVQ)** solves this by stacking Q quantizers in sequence, each one encoding the quantization error — the *residual* — that the previous level left behind:

<figure>
  <img src="/assets/images/codec-animations/rvq_residuals.gif"
       alt="RVQ levels: each quantizes the residual from the previous, approximation converges"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Level 1 gives a rough approximation. Each subsequent level refines the error. By level 4 the approximation is very close to the original — like progressive JPEG, where the first scan gives a blurry image and each pass sharpens the detail.
  </figcaption>
</figure>

```
z  = encoder output

Level 1:  nearest in codebook₁  →  index k₁,  quantized q₁
          residual r₁ = z - q₁

Level 2:  nearest in codebook₂  →  index k₂,  quantized q₂
          residual r₂ = r₁ - q₂

Level 3:  nearest in codebook₃  →  index k₃,  quantized q₃
          ...and so on...

Reconstruction = q₁ + q₂ + ... + qQ  ≈  z
```

One audio frame becomes the tuple `[k₁, k₂, …, kQ]` — Q integers. With Q=8 and N=1024 per codebook, the theoretical representational capacity is 1024⁸ combinations per frame. More practically, each additional level halves the residual energy — early levels capture coarse structure, later levels refine fine detail.

**The image parallel:** Progressive JPEG. The first JPEG scan transmits only the DC coefficient of each 8×8 block — you get a heavily blurred but recognizable image. Each subsequent scan adds higher-frequency AC coefficients, sharpening edges and texture. Level 1 of RVQ is the DC scan: rough speaker identity and phoneme structure. Level 8 is the final AC scan: fine acoustic texture that completes the picture.

*Pause for a moment here.* This progressive structure has a non-obvious implication: **you can generate high-quality speech without generating all levels autoregressively.** If you generate levels 1–2 with an expensive AR model (which captures semantics, prosody, speaker), you can fill in levels 3–8 with a much cheaper parallel model conditioned on the coarse codes. This is the architecture of MARS6 and Fish Speech — and it's why modern TTS can be both high-quality and fast.

### Bitrate calculator

Try different codec configurations and see how the numbers work out:

{% include bitrate_calculator.html %}

### Listen: the coarse-to-fine hierarchy in real audio

These are genuine **EnCodec reconstructions** — not simulations. The audio goes through the real EnCodec 24kHz encoder (CNN), gets encoded to 8 RVQ codebooks (N=1024 each), then gets decoded using only the first k levels through the real GAN decoder. The reference is the full 8-level round-trip.

**Codec:** [EnCodec](https://github.com/facebookresearch/encodec) by Meta AI — `encodec_24khz`, 6 kbps, 8 RVQ levels, N=1024 per level.
**Audio source:** PM Modi's *Mann Ki Baat* (30 Aug 2020) — [Internet Archive](https://archive.org/details/namo-speeches), public domain.

<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1.2rem;margin:1rem 0;">
  <div style="margin-bottom:0.8rem;">
    <span style="color:#8b949e;font-size:0.85em;display:block;margin-bottom:4px;">
      Reference — all 8 levels (full EnCodec round-trip)
    </span>
    <audio controls style="width:100%;accent-color:#58a6ff;">
      <source src="/assets/audio/hindi_original.wav" type="audio/wav">
    </audio>
  </div>
  <div style="margin-bottom:0.8rem;">
    <span style="color:#ff7b72;font-size:0.85em;display:block;margin-bottom:4px;">
      <strong>Level 1 only (coarse)</strong> — This is what the LLM must generate to get speaker identity and prosody right. Notice: the voice is recognizable, the rhythm of the sentence is preserved, but the quality is hollow and metallic — the GAN decoder is trying its best from only coarse latents.
    </span>
    <audio controls style="width:100%;accent-color:#ff7b72;">
      <source src="/assets/audio/hindi_rvq_level_1.wav" type="audio/wav">
    </audio>
  </div>
  <div style="margin-bottom:0.8rem;">
    <span style="color:#ffa657;font-size:0.85em;display:block;margin-bottom:4px;">
      <strong>Levels 1–3</strong> — Intelligible, natural prosody, good voice quality. Slight metallic tinge on fricatives (/s/, /sh/). Most TTS systems target roughly this quality range.
    </span>
    <audio controls style="width:100%;accent-color:#ffa657;">
      <source src="/assets/audio/hindi_rvq_level_3.wav" type="audio/wav">
    </audio>
  </div>
  <div>
    <span style="color:#3fb950;font-size:0.85em;display:block;margin-bottom:4px;">
      <strong>All 8 levels</strong> — Perceptually transparent to the reference. Fine consonant texture and breath sounds fully restored.
    </span>
    <audio controls style="width:100%;accent-color:#3fb950;">
      <source src="/assets/audio/hindi_rvq_level_8.wav" type="audio/wav">
    </audio>
  </div>
</div>

**What to listen for:** In level 1, the *timbre* — the spectral envelope that identifies the speaker — is already there. That's because the first codebook directly maps to the most energetic features of the encoder's latent space, which are dominated by speaker characteristics. What's missing is the fine harmonic texture that makes speech sound natural. Each subsequent level restores a little more of that texture.

---

## 8. The GAN decoder: why it sounds good

Getting from integer codes back to perceptually good audio is harder than it looks. A naive approach — train the decoder with L1 or L2 loss between predicted and real waveforms — produces over-smoothed, muffled output. The model learns to predict the *average* over all possible waveforms consistent with the codes, which is a blurry compromise.

The solution is a **Generative Adversarial Network (GAN)** decoder:

<figure>
  <img src="/assets/images/codec-animations/gan_decoder.gif"
       alt="GAN decoder: generator vs multi-scale and multi-period discriminators"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    The generator (decoder) must simultaneously fool 8 discriminators — 3 operating at different waveform resolutions (MSD) and 5 operating on periodic structure at different timescales (MPD). Being forced to fool all 8 pushes the generator toward perceptually sharp, realistic output.
  </figcaption>
</figure>

**Multi-Scale Discriminator (MSD):** Three discriminators run on the raw waveform at progressively pooled resolutions. The first sees the full-resolution waveform (catches fine-grained artifacts). The second sees every other sample (mid-range artifacts). The third sees every fourth (coarse structure). Together they enforce consistency across time scales.

**Multi-Period Discriminator (MPD):** Five discriminators reshape the 1D waveform into 2D grids at periods [2, 3, 5, 7, 11] before running 2D convolutions. This captures the periodic structure of speech harmonics — the very thing that makes speech sound like speech — which 1D discriminators tend to miss.

The training objective:

```python
# Generator (decoder) losses
L_recon  = mel_spectrogram_loss(real, fake)       # match spectral shape
L_feat   = feature_matching_loss(D, real, fake)   # match internal D representations
L_adv    = adversarial_loss(D, fake)              # fool all 8 discriminators

# Codebook losses
L_commit = ||z - z_q.detach()||²     # encoder "commits" to nearest codebook entry
L_vq     = ||z.detach() - z_q||²     # codebook entry moves toward encoder output

# Semantic distillation (Mimi / SpeechTokenizer only)
L_sem    = mse_loss(codec_level1, hubert_features)

L_total  = L_recon + λ_f*L_feat + λ_a*L_adv + λ_v*(L_commit + L_vq) + λ_s*L_sem
```

`L_feat` (feature matching) is underrated. Rather than just asking "does this fool the discriminator?", it asks "does this activate the discriminator's internal neurons in the same pattern as real audio?" This gives a much richer training signal — the discriminator becomes an involuntary perceptual quality metric.

---

## 9. Neural codec architectures: the evolution

Now we can talk about specific models. The key design choices are: how fast does the encoder downsample (fps), how many codebook levels, how big is each codebook, and what architecture generates the per-frame representations.

### Where frame rate comes from

The encoder is a 1D CNN with progressive strided convolutions. The **stride product** — multiply all the strides together — determines total temporal compression:

```
frames_per_second = sample_rate / stride_product

EnCodec:  24,000 / 320   = 75 fps   (1 token every 13ms)
Mimi:     24,000 / 1,920 = 12.5 fps (1 token every 80ms)
```

Lower fps means shorter token sequences, which means cheaper autoregressive generation. But each frame must now encode 6× more information — which requires a richer per-frame representation, which requires a more powerful encoder. This is the fundamental tradeoff.

### Architecture spectrum

**Pure CNN (EnCodec, DAC, SoundStream):** Stack of 1D convolutions with residual blocks. Fast, causal, streaming-friendly. But each frame only "sees" the local neighborhood defined by its receptive field (~40ms). No context about what came before or after. Compensates by needing more codebook levels — EnCodec needs 8 at 75fps.

The [DAC paper](https://arxiv.org/abs/2306.06546) (Descript Audio Codec) pushed this architecture further: Snake activations (better at capturing periodicity than ReLU), improved codebook training, and 44.1kHz support with 9 codebook levels at its default 8 kbps configuration. It's the current state-of-the-art for pure-CNN codec quality.

**CNN + Transformer (Mimi, SpeechTokenizer):** After the CNN downsampler, add Transformer layers with self-attention over the full sequence. Now each frame can attend to every other frame — it encodes not just "what frequencies are present at this moment" but "what's happening in the context of this sentence." Richer representations per frame → can run at lower fps with fewer levels.

**Semantic distillation:** Pure reconstruction training produces codebooks that cluster by acoustic similarity. But for LLM-based TTS, you need the first codebook level to cluster by *linguistic* content — similar phonemes should map to nearby entries regardless of speaker, pitch, or recording conditions.

```python
# Force level 1 to match representations from a pretrained SSL model
hubert_features = hubert_model(audio)           # frozen HuBERT
codec_level1    = codec_encoder(audio)          # before VQ

L_semantic = F.mse_loss(codec_level1, hubert_features)
```

After distillation, `k₁` tokens are linguistically meaningful. The LLM predicting them is predicting language structure, not raw acoustic noise. This is the difference that makes systems like Mimi work for real-time speech-to-speech.

---

## 10. Codec comparison

| Model | SR | fps | Levels | Codebook | Encoder | Semantic | Best for |
|---|---|---|---|---|---|---|---|
| SoundStream | 24k | 75 | 8 | 1024 | CNN | No | Original RVQ paper |
| EnCodec | 24k | 75 | 8 | 1024 | CNN | No | General audio compression |
| DAC | 44.1k | ~43 | 9 | 1024 | CNN (Snake) | No | HiFi music + speech |
| SpeechTokenizer | 16k | 50 | 8 | 1024 | CNN + HuBERT | Yes | Speech LLMs |
| Mimi | 24k | 12.5 | 8 | 2048 | CNN + Transformer | Yes (WavLM) | Real-time streaming |
| SNAC | 24k | 12.5/25/50 | 3 multi-scale | 4096 | CNN multi-scale | No | Hierarchical TTS |
| WavTokenizer | 24k | 40 | 1 | 4096 | CNN + Transformer | Yes | Single-stream LLMs |

**The pattern:** CNN-only codecs compensate for limited per-frame representations with more levels. CNN + Transformer codecs pack more into each frame and can run at lower fps. Semantic distillation is the essential ingredient for LLM-based TTS — without it, the LM is predicting acoustic texture, not language.

**A word on WavTokenizer** — it sits in a category of its own. By combining an extremely powerful CNN + Transformer encoder with a very large codebook (N=4096), it collapses the entire RVQ stack down to a single token per frame. At 40fps with 1 level, a 5-second clip becomes just 200 integers — a sequence any LLM can handle without any interleaving or multi-stream scheduling. The single-codebook direction is where several research groups are pushing, and WavTokenizer is currently the strongest representative. Quality is slightly below 8-level codecs at equivalent bitrate, but implementation simplicity is a real advantage.

---

## 11. Which codec should I actually use?

If you're about to train a speech model and you're staring at the comparison table wondering which row to start with — here's the honest answer:

**Use SNAC** if you're doing offline TTS and want the best quality-to-token-count ratio. It's a small, clean pip-installable repo, and its multi-scale structure maps directly onto a hierarchical AR+NAR architecture like MARS6. If your goal is "train a TTS that sounds like a human," start here.

**Use Mimi** if you need real-time or streaming output, or if you're building on top of Moshi. Its flat token structure means you can emit audio every 80ms without buffering. Level 1 tokens are linguistically structured out of the box (WavLM distillation is baked into the pretrained checkpoint), which matters a lot if your LLM backbone isn't huge.

**Use EnCodec** if you want maximum ecosystem compatibility. Almost every open-source TTS repo (VALL-E X, Voicebox, many others) ships with an EnCodec loader. The `encodec_24khz` checkpoint is the de facto standard reference point, which means benchmarks and community checkpoints are easy to compare against.

**Use DAC** (Descript Audio Codec) if audio quality is your top priority and you're not locked to 24kHz. Snake activations and improved codebook training push reconstruction quality noticeably above EnCodec. It supports 44.1kHz which matters for music or high-fidelity speech. `pip install descript-audio-codec` and it just works.

**Use WavTokenizer** if you want the simplest possible LLM setup. One token per frame, single stream — no interleaving, no multi-level scheduling, no delay patterns. For a first speech LLM experiment, the reduction in implementation complexity is worth the slight quality cost. The single-codebook trend is also where the field seems to be heading for pure LLM applications.

---

## 12. Mimi: why flat RVQ wins for streaming

<figure>
  <img src="/assets/images/codec-animations/mimi_streaming.gif"
       alt="Mimi flat token grid vs SNAC multi-rate, streaming frame by frame, CNN+Transformer encoder"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Left: SNAC requires 7 tokens across 3 timescales before it can decode 80ms — you must wait. Right: Mimi's flat structure means all 8 level-k tokens for one frame always arrive together — decode immediately, every 80ms. Bottom: the CNN + Transformer pipeline and why the larger N=2048 codebook compensates for lower fps.
  </figcaption>
</figure>

Mimi (Kyutai's codec, designed for Moshi real-time speech-to-speech) makes every design decision in service of streaming at low latency.

### Flat tokens = immediate decode

All 8 of Mimi's codebook levels run at the **same 12.5fps**. For any given audio frame, all 8 tokens `[k₁, …, k₈]` are produced together. The decoder receives all 8 and immediately reconstructs 80ms of audio — no waiting for tokens from other timescales.

Compare this to SNAC's multi-rate structure: to decode 80ms of audio you need `1 coarse + 2 mid + 4 fine = 7 tokens` that arrive at three different rates. In autoregressive generation you must buffer until the slowest stream catches up. This adds latency that's inherent to the architecture, not fixable by engineering.

### CNN + Transformer: context earns fewer tokens

```
Audio (24kHz)
  ↓ CNN downsampler (strided convolutions)
    → Local acoustic features, ~40ms receptive field
    → 24,000 / 1,920 = 12.5 fps

  ↓ Transformer (self-attention over full sequence)
    → Each frame attends to every other frame
    → Understands prosody, sentence rhythm, phoneme context
    → Much richer per-frame representation

  ↓ RVQ (8 levels, N=2048)
```

The Transformer doesn't add latency to *generation* (the codec encoder runs once offline to produce training tokens) but it does add compute to *inference encoding*. Moshi mitigates this with a causal Transformer variant that can be applied streaming.

### N=2048: compensating for lower fps

```
EnCodec:  75 fps  × 8 levels × log₂(1024) = 75 × 8 × 10  = 6,000 bps
Mimi:     12.5 fps × 8 levels × log₂(2048) = 12.5 × 8 × 11 = 1,100 bps
```

Mimi operates at **~1.1 kbps** — ~5.5× more compressed than EnCodec at comparable quality. The Transformer encoder earning richer per-frame representations is why this is possible.

### WavLM distillation

```python
wavlm_features = wavlm_model(audio)      # frozen WavLM, strong SSL model
codec_level1   = mimi_encoder(audio)     # before VQ, level 1 only
L_semantic     = F.mse_loss(codec_level1, wavlm_features)
```

After training, Mimi's `k₁` tokens are linguistically structured. The same phoneme spoken by different speakers maps to nearby entries in codebook 1. This is what makes Moshi's full-duplex speech-to-speech pipeline work — the LLM predicting Mimi tokens is predicting language, not waveform texture.

---

## 13. SNAC: multi-scale RVQ for hierarchical generation


<figure>
  <img src="/assets/images/codec-animations/snac_multiscale.gif"
       alt="SNAC three-tier streams: 1 coarse = 2 mid = 4 fine tokens per audio chunk"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    SNAC runs each codebook level at a different frame rate — the explicit multi-scale structure mirrors how information actually distributes across time in speech. 1 coarse token = 2 mid = 4 fine. This hierarchy maps directly onto a hierarchical generation architecture.
  </figcaption>
</figure>

Where Mimi optimizes for streaming, SNAC optimizes for generation quality in offline settings. The key idea: run each codebook level at a different temporal resolution, matching the natural information hierarchy of speech.

```
Coarse (12.5fps):  [c₀              c₁              c₂     ]
Mid    (25fps):    [m₀    m₁        m₂    m₃        m₄  m₅ ]
Fine   (50fps):    [f₀f₁f₂f₃       f₄f₅f₆f₇       f₈ ...  ]
```

Speaker identity and sentence prosody are slow-varying (coarse). Phoneme-level events are medium-speed (mid). Fine acoustic texture is fast (fine). By matching the timescale of each codebook level to the timescale of what it encodes, SNAC gets more efficient use of each token.

This multi-rate structure maps naturally onto a hierarchical generation strategy: an AR language model generates coarse tokens (12.5fps, semantically meaningful, cheap to produce), and a faster NAR model fills in mid and fine tokens in parallel. This is the pattern used by several SNAC-based systems — including **[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)** (Llama-3 backbone, runs on a single consumer GPU) which uses a SNAC-style codec and demonstrates that a well-chosen codec can give you voice cloning quality that would have required proprietary infrastructure two years ago.

Fish Speech takes a different path entirely — instead of SNAC, it uses their own **Firefly codec**: a GAN-based audio codec with residual VQ, trained specifically to handle multilingual phoneme diversity. The token structure is flat (closer to EnCodec), but the model pairs it with a Transformer that generates semantic tokens first and acoustic tokens second in two separate passes. Fish Speech 1.5 (the Firefly-era version) supports 13 languages — the codec's codebook is trained across that multilingual distribution, which is why the token space generalises better across languages than an English-only codec would.

**The streaming tradeoff:** To decode one 80ms audio chunk from SNAC, you need all 7 tokens (1+2+4). In a streaming AR system, you can't emit audio until all 7 are ready — which adds latency. This is why Mimi exists: for real-time speech, flat structure wins. For offline high-quality TTS, SNAC's hierarchy wins.

---

## 14. Codec-based TTS: the full pipeline

Once you have a good codec, building a TTS system on top of it is conceptually straightforward — it really is just next-token prediction:

```
Step 1 — Train codec offline (once)
  Large audio corpus → encoder + RVQ + GAN decoder
  Result: a frozen codec that converts audio ↔ integer sequences

Step 2 — Extract tokens from all training audio (offline)
  For each audio clip: run frozen encoder → store integer sequences

Step 3 — Train TTS model
  Input:  text token sequence (from a text tokenizer)
  Target: codec token sequence (pre-extracted in step 2)
  Loss:   cross-entropy over predicted vs actual codec indices
  Architecture: any LLM — Transformer, Mamba hybrid, etc.

Step 4 — Inference
  Text → TTS model → codec indices → codec decoder → waveform
```

The TTS model never touches a waveform or a spectrogram. It's pure integer-sequence prediction — structurally identical to language modeling. This is why the same architectural innovations that work for text generation (attention variants, KV caching, speculative decoding) transfer directly to TTS.

### Actually extracting tokens from audio

Here's what the code looks like for each major codec. Concretely, for a 5-second mono clip at 24kHz (120,000 samples):

**EnCodec:**
```python
# pip install encodec
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)   # 8 RVQ levels at 6 kbps
model.eval()

wav = wav.unsqueeze(0)  # [1, 1, 120000]
with torch.no_grad():
    encoded = model.encode(wav)

codes = encoded[0][0]  # [B=1, K=8, T_frames]
# For 5s at 24kHz with stride=320: T_frames = 120000/320 = 375
# codes.shape = [1, 8, 375]
# codes[0, 0, :]  → level-1 tokens, shape [375]  — integers 0..1023
# codes[0, 7, :]  → level-8 tokens, shape [375]  — integers 0..1023
```

**SNAC:**
```python
# pip install snac
from snac import SNAC
import torch

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

wav = wav.unsqueeze(0)  # [1, 1, 120000]
with torch.no_grad():
    codes = model.encode(wav)  # returns a list of 3 tensors

# For a 5-second clip at 24kHz:
# codes[0].shape = [1, 62]   coarse  (5s × 12.5fps = 62 tokens)
# codes[1].shape = [1, 125]  mid     (5s × 25fps  = 125 tokens)
# codes[2].shape = [1, 250]  fine    (5s × 50fps  = 250 tokens)
# Each integer is in range 0..4095 (N=4096)
# Integers 0..4095 at each level (N=4096)
```

**DAC:**
```python
# pip install descript-audio-codec
import dac
from audiotools import AudioSignal

# Note: DAC ships in three variants. The comparison table shows the 44.1kHz config
# (9 codebooks, stride 1024 → ~43 fps). This code loads the 24kHz variant
# (12 codebooks, stride 320 → 75 fps) — pick the variant that matches your pipeline.
model = dac.DAC.load(dac.utils.download(model_type="24khz"))
model.eval()

signal = AudioSignal("speech.wav")
signal.resample(model.sample_rate)
signal.to_mono()

with torch.no_grad():
    x = model.preprocess(signal.audio_data, signal.sample_rate)
    z, codes, latents, _, _ = model.encode(x)

# codes.shape = [B=1, K=12, T_frames]  — 12 levels for the 24kHz model, integers 0..1023
# (44.1kHz model: K=9 levels, ~43 fps; 16kHz model: K=12 levels, 50 fps)
```

### How tokens get arranged for the LLM

With Q levels and T frames, you have a `[Q, T]` integer matrix. There are three ways to flatten it into the 1D sequence a transformer needs:

**1. Flat interleaved (VALL-E style)**

```
frame 1:  [k1_t1, k2_t1, k3_t1, ..., k8_t1]
frame 2:  [k1_t2, k2_t2, k3_t2, ..., k8_t2]
...
→ sequence length = Q × T = 8 × 375 = 3,000 tokens for 5s
```

Simple to implement. But it makes the LLM predict all 8 levels autoregressively — levels 7 and 8 (perceptually subtle details) get the same expensive AR compute as level 1 (speaker identity). Wasteful, but works.

**2. Delay pattern (MusicGen / AudioCraft)**

Each level is shifted by one frame position before interleaving:

```
timestep:  1     2     3     4     5
level 1:  [k1_1, k1_2, k1_3, k1_4, k1_5]
level 2:  [  -, k2_1, k2_2, k2_3, k2_4]   ← shifted right by 1
level 3:  [  -,   -, k3_1, k3_2, k3_3]    ← shifted right by 2
```

At each transformer step, the model predicts all Q levels simultaneously (multi-head output, one head per level). Because of the delay, when predicting `k2_t`, the model has already seen `k1_t` in its context — so levels condition on each other without extra AR steps. Same sequence length as flat, but parallel multi-head prediction makes it faster.

**3. Hierarchical AR + NAR (MARS6 / Fish Speech)**

```python
# Stage 1: expensive AR model generates level-1 tokens only
coarse_tokens = ar_model(text_tokens)          # [T] integers, generated one by one

# Stage 2: cheap NAR model fills in levels 2–8 in parallel
#          conditioned on coarse tokens + text
fine_tokens = nar_model(coarse_tokens, text)   # [Q-1, T] integers, all at once
```

Level 1 is generated autoregressively (captures semantic / prosodic structure). Levels 2–8 are predicted in parallel given level 1 (captures acoustic detail, which is largely determined by the coarse structure anyway). This gives you the best of both: AR quality for what matters, NAR speed for what doesn't.

| Strategy | Sequence length | AR steps | When to use |
|---|---|---|---|
| Flat interleaved | Q × T | Q × T | Simple baseline, small models |
| Delay pattern | Q × T | T | MusicGen-style multi-head |
| AR + NAR | T (AR) + parallel (NAR) | T | Best quality/speed — MARS6, Fish Speech |

**Speaker conditioning** works by conditioning the LLM on a speaker embedding extracted from a short reference clip:

```python
spk_emb  = speaker_encoder(reference_audio)    # ECAPA-TDNN or similar
lang_emb = language_embedding[lang_id]          # learned per-language token

# Prepend to the LLM's input context
x = text_embeddings + spk_emb.unsqueeze(1) + lang_emb.unsqueeze(1)
codec_tokens = tts_model(x)
```

5–10 seconds of reference audio is enough for strong zero-shot voice cloning. The model has learned a general function: "given an embedding in this region of speaker space, produce codec tokens with these acoustic characteristics."

**Why strong multilingual LLM backbones matter:** If your TTS base model is Qwen3, Llama, or another model pretrained on multilingual text, it already deeply understands Hindi, Tamil, Arabic syntax and semantics. Fine-tuning on `{Hindi text → Hindi codec tokens}` pairs only teaches the *acoustic* mapping, not the language. You can get good multilingual TTS with 1–2 hours of target-language audio, rather than hundreds of hours from scratch.

### How much data do you actually need?

A question every practitioner hits immediately:

| Goal | Approx. data needed |
|---|---|
| Train a codec from scratch | 5,000–50,000 hours (diverse audio) |
| Fine-tune an existing codec | 100–500 hours (target domain) |
| Train a speech LM on top of a frozen codec | 500–5,000 hours |
| Fine-tune a pretrained speech LM, new speaker style | 10–100 hours |
| Zero-shot voice cloning at inference | 5–30 seconds (no training) |
| Multilingual fine-tune of an LLM-backbone TTS | 1–5 hours per language |

The last row is what makes multilingual speech LMs exciting right now. **Qwen3-TTS** (Alibaba, 2026) takes this to an extreme. It uses a dual-track architecture with two custom speech tokenizers: a 25Hz single-codebook tokenizer (vocabulary size 32,768, designed for maximum quality) and a 12.5Hz 16-layer RVQ tokenizer (1 semantic codebook + 15 acoustic RVQ layers, vocabulary 2,048 per layer, designed for streaming at <130ms first-packet latency). The Qwen3 LM backbone (0.6B or 1.7B parameters) predicts the semantic codebook autoregressively; a **Multi-Token Prediction (MTP)** module then fills all 15 residual acoustic codebooks in a single parallel step — eliminating the sequential AR bottleneck for fine-grained detail. Decoding runs through a flow-matching DiT to mel-spectrograms, then BigVGAN. Trained on 5 million+ hours across 10 languages; zero-shot voice cloning from just 3 seconds of reference audio. Achieves best English WER on the Seed-TTS benchmark and highest speaker similarity across all 10 test languages. The MTP pattern for parallel acoustic refinement is the cleanest published example of the AR+NAR split applied to a streaming-first codec design.

**[Chatterbox](https://github.com/resemble-ai/chatterbox)** by Resemble AI (and Chatterbox Turbo for lower latency) uses a clean three-stage pipeline: a Transformer LM predicts discrete speech tokens at 25Hz, a flow-matching model decodes those tokens into mel-spectrograms, and finally a HiFT vocoder renders the waveform at 24kHz. The key design decision is keeping these three stages separate — the flow-matching mel decoder can be fine-tuned independently for new speaker styles without retraining the LM, and the LM can be swapped for a larger one without touching the vocoder.

**Voicebox** (Meta) is worth knowing about as a contrast to the codec-token approach. Rather than discretizing audio into integer tokens, Voicebox applies flow matching directly over mel-spectrogram frames — continuous, not discrete. It's non-autoregressive (fills in masked audio segments) and requires no RVQ codec at all. The tradeoff: it's harder to condition on arbitrary text in an autoregressive way, and it can't leverage standard LLM infrastructure. It's a reminder that discrete codec tokens are not the only path to neural audio generation — just the most LLM-compatible one.

---

## 15. What codecs encode: the coarse-to-fine hierarchy

The most important empirical finding about RVQ codecs — and the one that makes the whole TTS system architecture make sense:

| Level | What it encodes |
|---|---|
| 1 (coarse) | Speaker identity, timbre, broad phoneme class, sentence prosody |
| 2–3 | Phoneme detail, voice quality, speaking style |
| 4–6 | Fine acoustic texture, room acoustics, noise floor characteristics |
| 7–8 | Perceptually subtle details, near quantization noise |

You can verify this by ear using the audio examples above. Level 1 alone gives you a recognizable voice — the speaker identity, rhythm, and sentence prosody are preserved — but the audio itself sounds hollow and metallic. The GAN decoder is doing its best from extremely sparse latents, and what's missing is the fine harmonic texture that makes speech sound natural. Level 3 is where most listeners would call it "intelligible and good enough." Levels 4–8 are the difference between "sounds good" and "sounds transparent."

This hierarchy has a direct architectural consequence: **the LLM must get level 1 right — everything else follows from that.** Level 1 errors mean the wrong speaker, wrong prosody, wrong phoneme. Levels 4–8 errors are practically imperceptible. This is why MARS6 and Fish Speech use AR generation only for coarse tokens and parallel NAR for fine tokens: the expensive sequential generation is focused exactly where it matters.

---

## 16. Open questions

**Mamba in codec encoders.** Transformers give global context but require O(T²) attention and can't stream causal context efficiently. Mamba's selective SSM is causal, O(T), and carries context in a fixed-size hidden state. A CNN + Mamba encoder could match Mimi quality with native streaming and lower compute. This is an active area of exploration.

**Cross-lingual codec token alignment.** A retroflex /ɖ/ in Hindi and an alveolar /d/ in English are acoustically similar — they should cluster near each other in codebook level 1 of a multilingual codec. If that alignment exists, multilingual TTS fine-tuning is easier: the LLM only needs to learn new text→token mappings, not new acoustic representations. Worth probing systematically.

**Streaming hierarchical codecs.** SNAC's multi-rate structure is the right inductive bias for offline generation. But can you make it streaming? A design where coarse tokens are causal (emit immediately), and fine tokens are filled by a non-causal model over a small lookahead buffer, could combine SNAC's quality with Mimi's latency. This is an open design problem.

---

## Summary

| Concept | Audio | Image |
|---|---|---|
| Raw representation | Waveform samples | Pixel values |
| 2D transform | STFT spectrogram | 2D pixel grid |
| Perceptual weighting | Mel filterbank | DCT basis in JPEG |
| Dynamic range | Log mel | Gamma correction |
| Classical compression | MP3 psychoacoustic masking | JPEG quantization tables |
| Learned compression | Neural codec encoder | VAE (Stable Diffusion) |
| Discrete bottleneck | Vector quantization | GIF palette quantization |
| Multi-level refinement | RVQ levels | Progressive JPEG passes |
| Coarse-to-fine hierarchy | SNAC token streams | Progressive JPEG scans |
| Semantic alignment | HuBERT/WavLM distillation | CLIP-guided latents |

---

## References and further reading

**Foundational videos**
- [3Blue1Brown — But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) — Essential visual intuition for the STFT and mel filterbank
- [Kyutai codec explainer](https://kyutai.org/codec-explainer) — The best visual walkthrough of neural audio codecs; highly recommended alongside this post

**Papers**
- [SoundStream — Zeghidour et al., 2021](https://arxiv.org/abs/2107.03312) — First RVQ-for-audio paper
- [EnCodec — Défossez et al., 2022](https://arxiv.org/abs/2210.13438) — Meta's 24kHz neural codec
- [DAC — Kumar et al., 2023](https://arxiv.org/abs/2306.06546) — Descript Audio Codec, pushes CNN encoder quality further
- [SpeechTokenizer — Zhang et al., 2023](https://arxiv.org/abs/2308.16692) — Semantic distillation for speech LLMs
- [Mimi / Moshi — Kyutai, 2024](https://arxiv.org/abs/2410.00037) — Real-time speech-to-speech with flat RVQ
- [SNAC — Siuzdak, 2024](https://github.com/hubertsiuzdak/snac) — Multi-scale RVQ for hierarchical generation
- [MARS6 — Baas et al., 2025](https://arxiv.org/abs/2501.05787) — SNAC-based hierarchical TTS
- [WavTokenizer — Ji et al., 2024](https://arxiv.org/abs/2408.16532) — Single-codebook extreme compression
- [HuBERT — Hsu et al., 2021](https://arxiv.org/abs/2106.07447) — SSL model used for semantic distillation
- [VALL-E — Wang et al., 2023](https://arxiv.org/abs/2301.02111) — First LLM-based TTS using neural codecs
- [kNN-VC — Baas et al., 2023](https://arxiv.org/abs/2305.18975) — Voice conversion via k-nearest neighbor in codec space
- [Orpheus TTS — Canopy AI, 2025](https://github.com/canopyai/Orpheus-TTS) — Llama-3 + SNAC-style codec, consumer GPU TTS
- [Fish Speech — Fish Audio, 2024](https://github.com/fishaudio/fish-speech) — Firefly codec + dual-pass semantic/acoustic generation
- [Chatterbox — Resemble AI, 2025](https://github.com/resemble-ai/chatterbox) — Codec-conditioned TTS with separate vocoder fine-tuning
- [Qwen3-TTS — Alibaba, 2026](https://arxiv.org/abs/2601.15621) — Dual-track LM (25Hz single-codebook + 12.5Hz 16-layer RVQ), MTP for parallel acoustic codebook generation, 5M+ hours training, 10 languages
- [Voicebox — Le et al., Meta 2023](https://arxiv.org/abs/2306.15687) — Non-autoregressive flow matching over mel spectrograms (no discrete codec)

---

*Animations generated with [Manim Community](https://www.manim.community/). Audio examples use [EnCodec](https://github.com/facebookresearch/encodec) by Meta AI. Code examples use PyTorch and torchaudio.*
