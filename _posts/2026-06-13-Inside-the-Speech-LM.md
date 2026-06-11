---
layout: post
title: "Inside the Speech LM: How a Language Model Learns to Speak"
date: 2026-05-20
categories: speech deep-learning audio
author: Akash Singh
description: >
  My last post ended exactly where the codec hands off to the language model.
  This one goes inside — what the LM is actually modeling, how different
  architectures handle multi-stream codec tokens, and why the inner monologue
  in Moshi changes the whole picture for speech-to-speech. A bridge between
  TTS and full duplex.
---

My last post built the full story of how raw audio becomes a sequence of
integers — RVQ codebooks, GAN decoders, the coarse-to-fine hierarchy. I ended
with a table and a claim: codec-based TTS is just next-token prediction.

That claim is technically true and also somewhat misleading. The moment you
look at *what* the language model is predicting over, and *how* different
systems structure that prediction problem, you realize the design space is far
richer than "it's just an LLM." The choice of token arrangement directly
determines latency, quality, and whether the system is streaming-capable at
all. And once you step from TTS into true speech-to-speech — where the same
model listens and speaks simultaneously — the architecture changes in ways that
text-centric thinking doesn't anticipate.

This post covers that ground. It bridges the codec post toward the
speech-to-speech post I'm planning next on Moshi and LFM2.5.

> **Prerequisites:** This post builds directly on *[From Spectrograms to Speech Codecs: A Deep Dive](/speech/deep-learning/audio/2026/04/11/Speech-Codecs-Deep-Dive.html)* — specifically the RVQ section, the coarse-to-fine hierarchy, and the three token arrangement strategies introduced there. If you haven't read that post, the codec concepts here will land better with that foundation.

---

## Contents

1. [What the LM is actually modeling](#1-what-the-lm-is-actually-modeling)
- [The two paradigms: AR vs continuous denoising](#the-two-paradigms-what-kind-of-generation-problem-is-this)
2. [The token arrangement problem](#2-the-token-arrangement-problem)
3. [The delay pattern: MusicGen's trick](#3-the-delay-pattern-musicgens-trick)
4. [AR + NAR: the split that changes everything](#4-ar--nar-the-split-that-changes-everything)
5. [Training the codec LM: what the loop actually looks like](#5-training-the-codec-lm-what-the-loop-actually-looks-like)
6. [How MARS6 does it](#6-how-mars6-does-it)
7. [Fish Speech and the Firefly codec](#7-fish-speech-and-the-firefly-codec)
8. [Qwen3-TTS: MTP as the cleanest AR+NAR expression](#8-qwen3-tts-mtp-as-the-cleanest-arnar-expression)
9. [The inner monologue: Moshi's insight](#9-the-inner-monologue-moshis-insight)
10. [CALM and Pocket TTS: Kyutai's edge-first thinking](#10-calm-and-pocket-tts-kyutais-edge-first-thinking)
11. [Speculative decoding for TTS](#11-speculative-decoding-for-tts)
- [Diffusion and flow matching for TTS](#diffusion-and-flow-matching-for-tts)
12. [Sampling for TTS: how inference differs from text LLMs](#12-sampling-for-tts-how-inference-differs-from-text-llms)
13. [Open questions](#13-open-questions)

[Appendix: Fine-tuning Qwen3-TTS on Apple Silicon with MLX](#appendix-fine-tuning-qwen3-tts-on-apple-silicon-with-mlx)

---

## 1. What the LM is actually modeling

Take a 5-second speech clip, encode it with Mimi (12.5fps, 8 RVQ levels,
N=2048). You get an integer matrix of shape `[8, 62]` — 8 codebook levels,
62 frames. Flatten it into a sequence and you have 496 tokens.

A TTS language model learns the conditional distribution:

```
P(codec_tokens | text_tokens, speaker_embedding)
```

Which looks identical to:

```
P(word_t+1 | word_1, ..., word_t, context)
```

The training loop is the same. Cross-entropy loss, AdamW, learning rate
warmup, the whole standard LLM recipe. What's different is what makes this
harder than text:

**Codebook interdependency.** In text, each token is from a single vocabulary.
In multi-level RVQ, each frame has Q tokens from Q different codebooks. Level
2 at frame t is conditionally dependent on level 1 at frame t — they encode
residuals of each other. A model that ignores this structure will produce
incoherent audio, even if individual token perplexity looks fine.

**Temporal alignment.** Text tokens and codec tokens live on different
timescales. A single word might correspond to 4–8 codec frames, but that
correspondence isn't fixed — it varies with speaking rate, prosody, emphasis.
The LM must learn this alignment implicitly, or you need explicit monotonic
alignment supervision (VALL-T, arXiv:2401.14891, does this with a transducer
formulation).

**The distribution is multi-modal in a different sense.** In text, "the cat sat
on the ___" has a handful of plausible completions. In speech, any given
phoneme can be produced with enormous variation in pitch, timing, breathiness,
and room acoustics — all of which correspond to different level 4–8 codec
tokens. The model must not over-commit at fine-grained levels while
being precise at coarse levels. This is why the AR+NAR split I'll describe
below isn't just an efficiency trick — it's a better match to the actual
structure of the problem.

---

## The Two Paradigms: What Kind of Generation Problem Is This?

Before diving into token arrangements and specific architectures, it helps to know the full design space. Codec-LM-based TTS — everything from §2 onwards — is one bet. There is a second paradigm with a completely different answer to the question "what kind of generation problem is speech synthesis?"

**Paradigm A — Autoregressive token prediction (this post)**

Discretize audio via a neural codec (RVQ) into integer tokens. Train an LM to predict `P(token_t | token_{<t}, conditioning)` via cross-entropy, exactly like a text LLM. Generate one token at a time, decode the sequence through the codec decoder. Everything from VALL-E to Moshi to Qwen3-TTS lives here.

**Paradigm B — Continuous denoising (diffusion and flow matching)**

Encode audio to a continuous latent space — no quantization, no codebook. Learn to transform Gaussian noise into data via N iterative refinement steps. Generate by running a neural network N times over the full output, not one token at a time. Each pass refines every frame simultaneously. Examples: AudioLDM, F5-TTS, Supertonic (arXiv:2503.23108), Voicebox, NaturalSpeech 2.

**The third category: one-pass non-AR**

Models like Kokoro-82M sit outside both paradigms. A single Transformer forward pass takes text and speaker conditioning and emits mel-spectrogram frames directly — no iteration, no tokens. Fast. Simple. Quality is limited by the parallel prediction assumption: the model cannot condition frame 10 on what frame 9 actually sounded like in the current utterance.

| Paradigm | Discretization | Generation | Native streaming | TTS examples |
|---|---|---|---|---|
| AR codec-LM | RVQ integer tokens | Token by token | ✅ Native | VALL-E, Moshi, Qwen3-TTS, MARS6 |
| Continuous denoising | None (continuous latent) | N NFE passes, all frames | ❌ Chunked only | F5-TTS, Supertonic, AudioLDM |
| One-pass NAR | Mel frames (direct) | Single forward pass | ❌ Chunked only | Kokoro-82M |

The rest of this post covers Paradigm A in depth. A full treatment of Paradigm B — how Gaussian noise, ODEs, DDPM, and flow matching actually work, and when you should choose them over AR — comes in the section titled *Diffusion and Flow Matching for TTS* near the end.

---

## 2. The token arrangement problem

With Q levels and T frames, you have a `[Q, T]` integer matrix. A transformer
needs a 1D sequence. There are three fundamentally different ways to flatten
it, and the choice has real consequences.

<figure>
  <img src="/assets/images/speech-lm-animations/token_arrangements.gif"
       alt="Three-panel comparison: flat interleaved (Q×T AR steps), delay pattern (T steps, Q heads), AR+NAR (T AR steps plus 1 NAR step)"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Three ways to flatten a [Q, T] codec matrix into a transformer sequence. Left: flat interleaved — every token predicted autoregressively, Q×T steps. Center: delay pattern — each level shifted by one frame, T steps with Q parallel heads. Right: AR+NAR — T sequential steps for coarse (level 1), then one parallel pass for all fine levels. The choice determines whether you're streaming-capable and how fast inference is.
  </figcaption>
</figure>

### Flat interleaved

```
[k1_t1, k2_t1, ..., k8_t1, k1_t2, k2_t2, ..., k8_t2, ...]
```

Every level at every frame, one by one. Sequence length: Q×T. The model
predicts all 8 levels autoregressively — the same expensive AR step for level
8 (perceptual noise) as for level 1 (speaker identity). Simple to implement.
VALL-E used this. Works, but wasteful.

### Delay pattern (MusicGen)

Each level is shifted by one frame position:

```
timestep:  1      2      3      4      5
level 1:  [k1_1, k1_2, k1_3, k1_4, k1_5]
level 2:  [  —,  k2_1, k2_2, k2_3, k2_4]   ← shifted right by 1
level 3:  [  —,    —,  k3_1, k3_2, k3_3]   ← shifted right by 2
...
```

At each transformer step, a multi-head output predicts all Q levels in
parallel. The delay ensures that when predicting level k at frame t, the model
has already seen level k−1 at frame t in its context window. Levels condition
on each other without extra AR steps. Sequence length stays Q×T but effective
decode steps drop to T.

### AR + NAR

```python
# Stage 1: AR model on level-1 tokens only
coarse_tokens = ar_lm(text_tokens, speaker_emb)   # [T] — generated one by one

# Stage 2: NAR model fills levels 2–Q in parallel
fine_tokens = nar_lm(coarse_tokens, text_tokens)   # [Q-1, T] — all at once
```

Level 1 captures speaker identity, sentence prosody, phoneme structure — the
things the LM must get right for the output to be coherent. Levels 2–Q are
mostly acoustic refinement conditioned on level 1. The NAR pass treats fine
token generation as a parallel classification problem: given the coarse
structure, predict all fine tokens simultaneously.

The quality-speed tradeoff:

| Strategy | Sequence length | AR steps | Quality ceiling |
|---|---|---|---|
| Flat interleaved | Q × T | Q × T | High (but slow) |
| Delay pattern | Q × T | T | High |
| AR + NAR | T (AR) + parallel (NAR) | T | Highest for speed |

The through-line for everything that follows: AR steps are expensive, sequential, and unavoidable for semantic coherence. The three strategies above differ only in how aggressively they shrink the AR count. Every real system in this post is a different answer to the same question: *what is the minimum AR budget that still produces coherent, natural-sounding speech?*

---

## 3. The delay pattern: MusicGen's trick

MusicGen (Copet et al., 2023) introduced the delay pattern as the practical
solution to multi-codebook parallel generation. The key insight is architectural:
replace a single output head with Q parallel heads, each predicting one level.

<figure>
  <img src="/assets/images/speech-lm-animations/delay_pattern.gif"
       alt="The delay pattern: each codec level row shifts right by its level index, creating a diagonal. At each step, Q output heads fire in parallel from the same hidden state."
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Step 1: the unshifted [Q, T] grid — all levels start at frame 1. Step 2: apply the delay — level q shifts right by q positions, leaving empty slots on the left. Step 3: at each transformer step, a single hidden state hₛ drives Q output heads simultaneously, one per level. Because of the diagonal structure, level k at step s can attend to level k−1 at the same audio frame (which sits one step back in the sequence). Cross-level conditioning for free.
  </figcaption>
</figure>

```python
class DelayedCodebookLM(nn.Module):
    def __init__(self, d_model, n_codebooks, vocab_size):
        super().__init__()
        self.transformer = TransformerDecoder(d_model, ...)
        # One output head per codebook level
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)
        ])

    def forward(self, x):
        h = self.transformer(x)
        # All heads run on the same hidden state
        return [head(h) for head in self.heads]
```

During training, inputs are shifted by level index before being packed into
the sequence. During inference, you maintain a sliding buffer per level and
emit audio as soon as all Q tokens for a frame are available.

The important latency consequence: with the delay pattern on a codec running
at 12.5fps (Mimi), you get a new audio frame every 80ms. That's the theoretical
floor for streaming TTS — you can't improve it by making the LM faster,
because the codec frame rate is the limit. This is why codec frame rate is a
first-class design decision, not just a compression detail.

---

## 4. AR + NAR: the split that changes everything

The delay pattern is clever, but it still runs all Q prediction heads at every
step. AR + NAR goes further by questioning whether levels 2–Q need to be
sequential at all.

The empirical justification: take any trained RVQ codec and measure mutual
information between level 1 tokens and level k tokens across a large corpus.
It drops sharply after level 3. Level 7 tokens, given level 1, are nearly
independent of the text. The LM is doing expensive sequential computation to
predict information that is largely determined by the local acoustic context,
not by the linguistic content.

<figure>
  <img src="/assets/images/speech-lm-animations/ar_nar_split.gif"
       alt="AR+NAR pipeline: left side shows AR generating coarse tokens one by one with sequential arrows; right side shows NAR filling all fine levels simultaneously with a single wide arrow"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Stage 1 (AR): coarse tokens generated sequentially — T steps, one per frame. Each step must see all previous coarse tokens to get speaker identity and prosody right. Stage 2 (NAR): fine levels generated simultaneously in a single forward pass conditioned on the complete coarse sequence. For Q=8, this turns 7 sequential AR steps into 1 parallel NAR step — essentially free.
  </figcaption>
</figure>

The NAR pass treats fine token generation as a parallel mask-prediction
problem, similar to BERT or MaskGIT:

```python
# NAR inference — all fine levels in one forward pass
# coarse_tokens: [B, T] — from the AR stage
# text_tokens:   [B, L]

fine_logits = nar_model(
    coarse_tokens=coarse_tokens,
    text_tokens=text_tokens,
    level_ids=torch.arange(1, Q)   # conditioning on which level we're predicting
)
# fine_logits: [B, Q-1, T, vocab_size]
fine_tokens = fine_logits.argmax(-1)   # [B, Q-1, T]
```

One forward pass generates Q−1 levels simultaneously. For Q=8, that's 7 levels
in one shot vs 7 sequential AR steps. On a modern GPU, the NAR pass takes
roughly the same wall-clock time as a single AR step — a 7× effective speedup
on the fine-level generation.

The quality cost is minimal for levels 4–8. For levels 2–3 there's a small but
audible gap vs full AR — you lose some voice quality precision. Whether that
matters depends on your target RTFX. For edge deployment on M5 Pro, the
tradeoff is clearly worth it.

---

## 5. Training the codec LM: what the loop actually looks like

The three strategies above describe how to arrange tokens during both training and inference. The training loop is the same regardless — cross-entropy over predicted codec indices — but several implementation details are non-obvious and not well documented outside of paper appendices.

<figure>
  <img src="/assets/images/speech-lm-animations/training_pipeline.gif"
       alt="Three-stage training pipeline: audio files of varying lengths through codec encoder to .codec.npy arrays; naive batching with padding waste vs length-bucketed tight packing; dual-channel input format with text tokens and codec tokens stacked, loss mask highlighting which positions count."
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Stage 1: audio clips of varying lengths encoded offline to integer arrays — the codec encoder never runs during training. Stage 2: length bucketing sorts clips by duration before batching, cutting padding from ~35% to ~8% of each batch and giving 3–5× throughput improvement. Stage 3: the dual-channel input packs text tokens (top row) and codec tokens (bottom row) with shared position indices; the cross-entropy loss is computed only on the codec channel after BOS.
  </figcaption>
</figure>

### Data pipeline: from raw audio to training tokens

The first step is offline pre-tokenization. You run the codec encoder once over your entire dataset and save the integer arrays to disk. Training the LM never touches a waveform:

```python
# Pre-tokenization (run once, save results)
for audio_path, text in dataset:
    wav, sr = load_audio(audio_path)
    wav = resample(wav, target_sr=24000)
    codec_tokens = codec_encoder(wav)         # [Q, T] integers
    np.save(audio_path.replace(".wav", ".codec.npy"), codec_tokens)
```

This matters because the codec encoder (especially CNN + Transformer variants like Mimi) is memory-hungry. Running it during training is a reliable path to OOM. Pre-tokenization also means the training loop sees identical inputs across epochs — no stochastic augmentation — which is usually fine for TTS because audio augmentation can damage the prosody signal.

**Length bucketing** is essential. A 2-second clip produces ~25 codec frames; a 15-second clip produces ~188. Batching these naively wastes the majority of your compute on padding. Sort the dataset by duration, then batch by groups of similar length. Most TTS training frameworks implement this as a `sort_by_length: true` config option — without it, training throughput is typically 3–5× slower.

**Min/max duration filtering.** Clips under 1 second are usually artifacts (truncated recordings, silence detection errors). Clips over 15 seconds force the LM to maintain very long context — useful for quality but hard on memory early in training. Start with a tight window (2–12 seconds) and expand once the model converges on shorter sequences.

### The dual-channel input format

At each training step, the LM sees a packed sequence of text tokens and codec tokens. The specific packing varies by system, but the Qwen3-TTS format illustrates the general pattern:

```
Text channel:
  [PAD PAD PAD | BOS | text_tokens... | EOS | PAD PAD ...]

Codec channel:
  [PAD PAD PAD | NOTHINK | THINK_BOS | THINK_EOS | SPEAKER(pos 6) | PAD | BOS | codec_0...]
```

`NOTHINK` disables chain-of-thought for the codec generation path. `SPEAKER(pos 6)` is the speaker embedding token injected at sequence position 6 — the position index matters because Qwen3-TTS uses absolute positional encodings and reserves this slot for voice identity.

The codec channel predicts `codec_0` — the first codebook — autoregressively. The remaining 15 acoustic codebook levels enter as additive embeddings on the input side, giving the model full knowledge of "what came before acoustically" at each step. The loss is computed only on `codec_0` predictions plus a 0.3× weighted sub-talker loss on the acoustic codebook heads.

### Curriculum learning and convergence

TTS LMs benefit from curriculum learning: train on short, clean, single-speaker audio first, then progressively introduce longer clips and more speaker diversity. The intuition is the same as for text LLMs — easy examples first establish the token distribution structure, hard examples (long sequences, diverse speakers, noisy recordings) refine it.

Loss curves for codec LMs have a characteristic shape: a steep drop in the first few hundred steps as the model learns the basic token frequency distribution, a plateau while it learns prosodic structure, then a slow decline as it refines speaker conditioning. If loss plateaus early and WER on held-out text is poor, the usual culprits are too-high learning rate (full fine-tuning) or too-low rank (LoRA fine-tuning with rank 4 when rank 8 is needed).

**Validation metrics.** Perplexity on codec tokens is necessary but not sufficient — low perplexity with bad prosody is common. The practical validation loop: every N steps, generate 10 held-out sentences, run ASR on the output, compute WER. A WER below 5% on clean text typically correlates with subjectively good quality. DNSMOS (a non-intrusive MOS predictor) gives a cheap objective quality estimate without human listeners.

---

## 6. How MARS6 does it

With the three strategies in hand, the next several sections are implementations — each one a different answer to the AR budget question, optimized for a different tradeoff of quality, speed, and streaming capability.

MARS6 (Baas et al., arXiv:2501.05787) is currently my reference implementation
for clean AR+NAR TTS. It uses SNAC as the codec — the multi-scale RVQ I
covered in the last post — and builds a two-stage pipeline on top of it.

**Stage 1 (AR):** A Transformer language model autoregressively generates
SNAC coarse tokens (12.5fps). Input: text BPE tokens + speaker embedding from
a 10-second reference clip. Output: coarse codec token sequence.

**Stage 2 (NAR):** A separate smaller Transformer runs in parallel over the
coarse tokens and fills in SNAC mid (25fps) and fine (50fps) tokens. The NAR
model sees all coarse tokens simultaneously — it's a non-causal encoder, not
a causal decoder. This matters: non-causal attention over the coarse sequence
gives the NAR model full context for predicting fine details.

What makes MARS6 interesting beyond the architecture is the speaker conditioning
approach. Rather than a single speaker embedding, MARS6 uses a **style
prefix** — a short sequence of codec tokens extracted from the reference audio
is prepended to the LM's input context. The model learns that "generate speech
that continues in the style of this prefix." This gives richer conditioning than
a single embedding vector: the model can see the reference speaker's actual
prosodic patterns, not just a compressed summary.

```python
# MARS6 speaker conditioning (conceptual)
ref_coarse_tokens = codec.encode(reference_audio)[0]   # level-1 only, shape [T_ref]
ref_prefix = ref_coarse_tokens[:ref_prefix_len]        # first N frames as prefix

# AR model input: [BOS, text_tokens..., SEP, ref_prefix..., SEP]
input_ids = torch.cat([bos, text_ids, sep, ref_prefix, sep])
coarse_out = ar_model.generate(input_ids, max_new_tokens=max_T)
```

The result: voice cloning quality competitive with XTTS-v2 (Coqui TTS's
open-source voice cloning system, the previous quality benchmark for open-source
TTS) at a fraction of the inference cost, because the NAR stage runs the
expensive fine-detail generation in one shot.

**Speaker conditioning: how different systems condition on voice.** MARS6 is a good place to pause and compare approaches across the systems in this post:

| System | Conditioning signal | Mechanism |
|---|---|---|
| VALL-E / most TTS | Speaker embedding vector | Single vector prepended to LM input |
| MARS6 | Level-1 codec tokens from reference | Real tokens prepended — LM continues the prosodic pattern |
| Qwen3-TTS | 3s reference → speaker embedding | Injected into MTP module alongside semantic tokens |
| Moshi | User audio stream | No explicit conditioning — speaking style is in the live token stream |

The progression from embedding vector to token prefix to implicit stream is also a progression in conditioning richness. Each approach gives the model finer-grained information about the target voice, at the cost of more context tokens or more architectural complexity.

---

## 7. Fish Speech and the Firefly codec

Fish Speech takes a different path on the codec side. Instead of SNAC or Mimi,
it uses its own **Firefly** codec — a GAN-based RVQ codec trained
specifically across 13 languages, with the multilingual phoneme distribution
explicitly in the training data. (The codec design is covered in detail in
the previous post; this section focuses on the LM strategy built on top of it.)

The architecture is flat-token (EnCodec-style) but the generation is two-pass:

**Pass 1 — Semantic token generation:** A Transformer predicts a sequence of
semantic tokens (discrete units extracted from a pretrained speech SSL model,
similar to HuBERT units). These are language-level tokens — they encode what
is being said and in what style, but not the fine acoustic texture.

**Pass 2 — Acoustic token generation:** A second Transformer (the Firefly
decoder) takes the semantic token sequence and generates the full RVQ codec
token sequence.

This two-pass structure differs from AR+NAR in an important way: both passes
are autoregressive. The first pass is a text→semantic-token LM. The second is
a semantic-token→acoustic-token LM. The advantage is that each pass can be
trained and optimized independently — and the first pass can be used as a
standalone speech understanding model.

<figure>
  <img src="/assets/images/speech-lm-animations/fish_speech_pipeline.gif"
       alt="Fish Speech two-pass AR: text tokens through Semantic LM to speaker-agnostic semantic tokens at 25fps, then through Acoustic LM (with speaker embedding) to RVQ codec tokens at 75fps showing 3× upsampling. Bottom panel: two speakers saying the same word produce identical semantic tokens but different codec level-1 tokens."
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Pass 1 (Semantic LM): text tokens produce speaker-agnostic semantic tokens at 25fps. Pass 2 (Acoustic LM): semantic tokens plus a speaker embedding produce RVQ codec tokens at 75fps — a 3× temporal upsampling the acoustic LM learns to perform. Key property: two different speakers saying the same word map to the same semantic token sequence but different level-1 codec tokens. This factorization is what makes Fish Speech efficient to fine-tune for new languages.
  </figcaption>
</figure>

### What semantic tokens actually are

Semantic tokens are not the same as codec tokens. A codec token at level 1 is
the nearest codebook entry to an encoder output — it captures speaker timbre,
speaking style, and broad phoneme class, but it's still heavily acoustic. A
semantic token, by contrast, is extracted from a frozen SSL model (HuBERT or
WavLM) trained to predict masked audio frames from surrounding context. Because
that self-supervised task is essentially forced to ignore speaker identity and
recording conditions (which change between utterances), the resulting
representations cluster by linguistic content rather than acoustic texture.

In practice this means semantic tokens from two different speakers saying the
same word will often be identical, while the same speaker whispering vs
speaking normally will produce different codec tokens at level 1 but identical
semantic tokens. For a TTS LM, predicting semantic tokens first is predicting
*what is being said and how the prosody flows* before committing to *who is
saying it* — a cleaner factorization of the problem.

```python
# Fish Speech semantic tokenization (conceptual)
# ssl_model is a frozen HuBERT or WavLM
ssl_features = ssl_model.encode(audio)           # continuous, shape [T_ssl, d]
semantic_tokens = vq_head(ssl_features)          # VQ on top of SSL features
                                                 # shape [T_sem] integers 0..N_sem-1
```

The VQ head is a simple codebook applied on top of the frozen SSL model's
outputs — trained jointly with the Fish Speech LM, not separately. This gives
Fish Speech its own semantic vocabulary tuned to its acoustic codec, rather
than borrowing token IDs from a generic SSL model.

### Why two-pass AR and not AR+NAR

Fish Speech's two-pass approach — text→semantic AR, then semantic→acoustic AR
— looks slower than AR+NAR at first glance: you're running two full
autoregressive models. The latency cost is real. But the advantage is
factorization: the text→semantic model is purely linguistic and speaker-agnostic,
while the semantic→acoustic model handles voice identity and recording acoustics.

For multilingual fine-tuning, this separation is valuable. Fine-tuning the
text→semantic model teaches it the phonology of a new language. Fine-tuning
the semantic→acoustic model teaches it a new voice. You can update one without
touching the other — which is why Fish Speech 1.5 achieves competitive quality
on a new language with as little as 30 minutes of fine-tuning data. The
multilingual Firefly codec provides the cross-lingual acoustic bridge (Hindi
/ɖ/ clusters near English /d/ in the codec's level-1 space), and the
text→semantic pass provides the linguistic bridge (similar phoneme contexts
map to similar semantic token sequences). Only the acoustic-generation model
needs language-specific updating.

The practical inference pipeline:

```python
# Fish Speech inference (conceptual, simplified)
text_tokens  = text_tokenizer(text)
spk_emb      = speaker_encoder(reference_audio)     # voice identity

# Pass 1: text → semantic tokens — speaker-agnostic, no spk_emb
sem_tokens   = semantic_lm.generate(text_tokens)

# Pass 2: semantic + speaker → acoustic codec tokens (AR, ~75 tokens/s at full RVQ fps)
codec_tokens = acoustic_lm.generate(sem_tokens, spk_emb)

# Decode
waveform     = firefly_decoder(codec_tokens)
```

The semantic pass runs at a lower frame rate than the acoustic pass — roughly
25 semantic tokens per second vs 75 RVQ frames per second. This means the
semantic→acoustic model must learn a 3× upsampling in token space, which it
handles through a standard autoregressive stride pattern.

---

## 8. Qwen3-TTS: MTP as the cleanest AR+NAR expression

Qwen3-TTS (Alibaba, arXiv:2601.15621) is the most production-polished example
of the AR+NAR split I've seen published. It uses a dual-track tokenizer design
that is worth studying carefully because it makes the tradeoffs explicit.

**Track 1 — 25Hz single-codebook tokenizer:** Large codebook (vocabulary size
32,768), one token per frame, designed for maximum quality in offline
generation. The LM backbone (Qwen3, 0.6B or 1.7B parameters) predicts this
stream autoregressively.

**Track 2 — 12.5Hz 16-layer RVQ tokenizer:** 1 semantic codebook + 15
acoustic RVQ layers, vocabulary 2,048 per layer, designed for streaming. The
semantic codebook is predicted AR by the LM. The 15 acoustic layers are filled
by a **Multi-Token Prediction (MTP) module** in a single parallel step.

The MTP module is the key innovation:

```python
# Qwen3-TTS MTP (conceptual)
# ar_lm predicts the semantic codebook autoregressively
semantic_tokens = ar_lm.generate(text_tokens, speaker_emb)   # [T]

# MTP fills all 15 acoustic RVQ layers in one shot
# Input: semantic tokens + text conditioning + speaker embedding
acoustic_tokens = mtp_module(
    semantic_tokens=semantic_tokens,
    text_tokens=text_tokens,
    speaker_emb=speaker_emb
)
# acoustic_tokens: [15, T] — all levels in one forward pass
```

After token generation, decoding runs through a **flow-matching DiT** to
mel-spectrograms, then BigVGAN to waveform. The flow-matching mel decoder is
an additional quality layer — it's not predicting tokens, it's denoising a
continuous mel representation conditioned on the predicted discrete tokens.
This gives VALL-E-style discrete token generation the mel-space refinement
that diffusion-based systems benefit from.

Trained on 5M+ hours across 10 languages. Best English WER on Seed-TTS
benchmark. Zero-shot voice cloning from 3 seconds of reference audio.

The 1.7B variant (available via mlx-audio) is my current primary target for
M5 Pro deployment, and Hindi fine-tuning is an open question I'm actively
thinking about — the Qwen3 backbone has strong Hindi text knowledge from
pretraining, but Hindi audio is absent from the TTS training data. The
MTP module and the semantic codebook both need to learn the Hindi acoustic
mapping.

---

## 9. The inner monologue: Moshi's insight

Everything above is TTS — text in, audio out. Moshi (Kyutai, arXiv:2410.00037)
asks a different question: what if the model doesn't see text at all? What if
it takes raw audio in and produces raw audio out, with full-duplex capability —
listening and speaking at the same time?

The core Moshi architecture is Helium (7B parameter language model) +
Mimi codec (24kHz, 12.5fps, 8 levels) + a Depth Transformer that handles
inter-codebook dependencies within each frame, and a Temporal Transformer that
handles sequence-level dependencies across frames.

The two transformers address different axes of dependency. The **Temporal Transformer** is the main sequence model — causal, attending across the full audio history, one step per 80ms frame. The **Depth Transformer** is a short 8-step AR chain running *within* each frame: given the Temporal Transformer's hidden state at timestep t, it predicts k₁_t first, then k₂_t conditioned on k₁_t, through k₈_t. The inner text token (introduced below) enters at the top of this depth chain, so text conditioning propagates to every codebook level — coarse-to-fine conditioning gets a semantic anchor at every frame, not just at level 1.

The full-duplex capability comes from modeling the user and system audio as
**two separate parallel streams** at every timestep. At each 80ms frame,
Moshi predicts:

```
[user_k1, user_k2, ..., user_k8, sys_k1, sys_k2, ..., sys_k8]
```

No turn-taking. No VAD-triggered switching. Both streams run simultaneously.
This is architecturally possible because Mimi's flat RVQ means both streams
have tokens at every frame — there's no "silence" slot that complicates the
modeling.

### The inner monologue

But here's the part that makes Moshi genuinely clever. At each timestep,
before predicting audio tokens, Moshi predicts **text tokens**:

```
timestep t:
  → predict inner_text_token_t   (from text vocabulary)
  → predict sys_audio_tokens_t   (from Mimi codebook)
```

The inner text token is Moshi "thinking in words" before speaking them.

<figure>
  <img src="/assets/images/speech-lm-animations/moshi_inner_monologue.gif"
       alt="Moshi inner monologue: two parallel streams (user audio, system audio) running simultaneously. At each 80ms step, the system first predicts an inner text token, then 8 audio tokens conditioned on it. Three benefits annotated: linguistic grounding, implicit ASR, streaming control."
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Full-duplex: user and system audio streams run at every 80ms frame — no turn-taking, no VAD. Per-timestep ordering: the system predicts an inner text token first (teal), then 8 audio tokens conditioned on it (red). The text token appears in the sequence before the audio tokens it conditions, so the causal attention mechanism handles the conditioning naturally. Inner text predictions taken as a sequence give you a free transcription of what the model is saying.
  </figcaption>
</figure>

This serves several purposes simultaneously:

1. **Linguistic grounding.** Predicting a text token forces the model's hidden
   state to represent what it's about to say at the semantic level before
   committing to acoustic tokens. The LM's text pretraining knowledge is
   directly active in the audio generation pathway.

2. **Implicit streaming ASR.** The inner text predictions, taken as a sequence,
   are a transcription of what the model is saying. No separate ASR model
   needed — the same forward pass gives you both audio output and text output.

3. **Streaming synthesis control.** Because the text token comes first in the
   timestep ordering, you can condition on it when predicting audio tokens.
   The model knows it's going to say "hello" before it decides exactly what
   acoustic texture "hello" should have for this particular speaker and context.

```python
# Moshi timestep (conceptual)
for t in range(T):
    # Inner monologue: text token first
    text_logits = text_head(hidden_state_t)
    text_token = text_logits.argmax()   # or sample

    # Audio tokens conditioned on text token
    audio_logits = audio_head(hidden_state_t, text_token)
    audio_tokens_t = [audio_logits[k].argmax() for k in range(8)]

    # Both outputs emitted; hidden state updated for t+1
```

This is the architectural reason why Moshi can be a real-time full-duplex
speech model while also producing transcription-quality text output. It's not
two separate pipelines sharing compute — it's one model with two output heads
at each step, where the text head conditions the audio head.

The practical consequence for the Depth Transformer: the 8 audio codebook
levels for each frame are predicted not just conditioned on the sequence
history, but also conditioned on the current timestep's inner text token.
Coarse-to-fine conditioning gets an additional semantic anchor at every frame.

---

## 10. CALM and Pocket TTS: Kyutai's edge-first thinking

Moshi is 7B parameters. Running it in real time on a MacBook Pro M5 Pro at
bf16 is feasible. Running it on a Raspberry Pi 5 is not.

Kyutai published two follow-up systems in 2025 that address the edge
deployment problem in fundamentally different ways.

<figure>
  <img src="/assets/images/speech-lm-animations/calm_architecture.gif"
       alt="Side-by-side: Moshi path (LM hidden state → RQ-Transformer with 8 sequential AR steps → discrete codebook indices k1..k8 → Mimi decoder → waveform, 701M params) vs CALM path (LM hidden state → Consistency MLP → continuous VAE latent → VAE decoder → waveform, ~10M params). Braces label 701M vs ~10M. Callout: 70× fewer parameters in the audio generation head."
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Left (Moshi): the LM hidden state feeds the RQ-Transformer, which runs 8 sequential AR steps to predict 8 discrete codebook levels per frame — 701M parameters. Right (CALM): the same hidden state feeds a 10M consistency MLP that predicts a continuous VAE latent in one shot — no codebook, no sequential steps. Both paths end at a decoder producing 80ms of audio. The 70× parameter reduction in the audio head is the edge deployment argument.
  </figcaption>
</figure>

### CALM (arXiv:2509.06926)

CALM replaces the discrete RVQ codec with a **continuous VAE latent space**.
Instead of quantizing encoder outputs to the nearest codebook entry (producing
integers), the VAE encoder outputs a continuous Gaussian distribution. The
decoder takes samples from that distribution and reconstructs audio.

Why does this matter for efficiency? The RQ-Transformer in Moshi's architecture
— the component that autoregressively predicts 8 discrete codebook levels per
frame — is replaced with a **consistency MLP** of approximately 10M parameters
(per the CALM paper). The consistency MLP takes the LM's hidden state and
directly predicts the VAE latent, bypassing the sequential per-level token
prediction entirely.

```
Moshi:   LM hidden state → RQ-Transformer (701M) → 8 codebook indices → Mimi decoder
CALM:    LM hidden state → consistency MLP (~10M) → VAE latent → VAE decoder
```

The quality tradeoff: continuous latents with a consistency MLP produce
slightly smoother but less controllable output than discrete tokens. The 70×
parameter reduction in the audio generation head is the edge deployment
benefit.

CALM also uses the Helium backbone — Kyutai's own 7B LM pretrained on
multilingual text. But the insight is that you could in principle swap in a
smaller backbone and keep the consistency MLP approach. The MLP itself is
not the bottleneck.

### Pocket TTS

Pocket TTS is Kyutai's explicit attempt to answer: how small can a real-time
TTS system be while remaining competitive?

The architecture is a **distilled student** derived from a larger teacher TTS
model through a two-stage process:

**Stage 1 — CFG distillation:** Classifier-free guidance (CFG) requires running
the model twice at inference — once with conditioning, once without — and
combining the outputs. CFG distillation collapses this into a single forward
pass with minimal quality degradation.

**Stage 2 — Capacity distillation:** The teacher model is progressively pruned
to a 6-layer student. The student is trained to match the teacher's output
distribution (not just the final audio, but the intermediate token
distributions) — a form of knowledge distillation that preserves
the teacher's learned behaviors in a much smaller model.

The result is approximately 100M parameters achieving competitive word error
rate, running at ~6× real-time on MacBook Air M4 CPU. That puts it in the
same parameter range as Kokoro-82M but with the full-duplex and inner
monologue architectural lineage of Moshi.

To make the latency question concrete: at 12.5fps, generating 5 seconds of audio requires T=62 coarse tokens. On an M5 Pro at bf16, a 1.7B transformer step takes roughly 8–12ms — so 62 AR steps ≈ 500–750ms of generation time for 5s of audio, giving ~7–10× real-time. A 100M model like Pocket TTS runs the same step in under 2ms: 62 steps ≈ 120ms, putting it at ~40× real-time on M5 Pro CPU and within reach of real-time on RPi 5, where the ~10ms/step budget requires the model to process at ≥8fps.

For RPi 5 deployment, Pocket TTS is the most plausible path to a real-time
Kyutai-quality TTS that actually runs on constrained hardware. As of this
writing, the GGUF export path doesn't exist — but the model is small enough
that once someone writes it, the latency math works out.

---

## 11. Speculative decoding for TTS

Speculative decoding for LLMs — where a cheap draft model proposes k tokens
and a target model verifies them in one parallel forward pass — transfers
directly to codec-based TTS, but with a twist that makes it more interesting
than a direct port.

In text LLM speculative decoding:
- Draft model proposes k tokens
- Target model verifies all k in one parallel forward pass
- Accepted tokens are kept; on rejection, resample from the target at the
  divergence point
- Net effect: ~2–3× throughput improvement

Three groups published TTS-specific speculative decoding implementations at
ICASSP 2025 and Interspeech 2025, and their results diverge in interesting
ways. A fourth paper reframes the acceptance criterion itself.

### Nguyen et al. (arXiv:2410.13839, ICASSP 2025): MTP + Viterbi

Multiple prediction heads added to the AR module (similar to MEDUSA for text
LLMs). The novel contribution is the verification step: instead of
token-by-token rejection sampling, they use a **Viterbi algorithm** to select
the optimal token sequence across all candidate paths generated by the draft
heads.

Why Viterbi instead of standard rejection? Codec token sequences have strong
Markov structure — the probability of token t+1 is heavily conditioned on
token t. Viterbi exploits this to select globally optimal sequences rather
than greedily accepting or rejecting each token independently. The result:
**4–5× speedup** with minimal quality tradeoff. Larger than typical text LLM
speculative decoding gains.

### Li et al. — VADUSA (arXiv:2410.21951, ICASSP 2025)

MEDUSA-style draft heads directly on the VALL-E backbone (4 or 6 heads, tree
attention mask). Reports strong generalization across diverse speech token
types and datasets. The tree attention approach — maintaining a branching tree
of draft candidates and verifying them all in one pass — is particularly well
suited to the multi-modal nature of codec token distributions (where multiple
tokens can produce perceptually equivalent audio).

### Lin et al. — SSD (arXiv:2505.15380, Interspeech 2025)

The most conservative approach: a separate lightweight draft model (not draft
heads) generates candidates, verified in parallel by the CosyVoice target
model. **1.4× speedup** — smaller than the other two. Built on a single-stream
codec (CosyVoice's single-codebook semantic tokenizer), which has less
temporal structure to exploit than multi-level RVQ.

### The perceptual acceptance insight

The most interesting recent paper in this space is arXiv:2511.13732 —
"Principled Coarse-Grained Acceptance for Speculative Decoding in Speech."

In text speculative decoding, the rejection criterion is exact: if the draft
token and the target distribution don't agree, reject. This is justified because
a wrong word is a wrong word. In speech, this is too strict. Two different
codec tokens at level 6 might decode to audio that is perceptually identical
to a human listener. Rejecting a draft token that would have produced
perceptually equivalent output wastes a good draft.

The paper proposes **coarse-grained acceptance**: define an acoustic similarity
metric over the codec's decoded audio, and accept draft tokens whose decoded
output falls within a perceptual threshold of what the target model would have
produced. Higher acceptance rates → better effective draft model utilization →
higher speedup ceiling.

This is the theoretically correct framing for speculative decoding in a
perceptual domain. The challenge is computing the similarity metric cheaply
enough that it doesn't eat the speedup you just gained.

### A note on edge deployment

All four papers benchmark on server GPUs. The specific question of whether
these speedups hold on Apple Silicon (MLX, unified memory, different
parallelism model) or on CPU-only inference (RPi 5) is completely unstudied.
On bandwidth-constrained hardware where compute is underutilized during decode
— exactly the regime where speculative decoding is designed to help — the
theoretical gains should be larger, not smaller. But the draft model
selection, tree size, and Viterbi depth all need to be tuned for the target
hardware. This is an open gap that directly affects whether any of these
techniques are practically usable on M5 Pro or RPi 5. My expectation: Nguyen et al.'s Viterbi-based approach will translate best to Apple Silicon, because its verification step is memory-bandwidth-bound rather than compute-bound — exactly the profile where MLX's unified memory model helps most.

---

## Diffusion and Flow Matching for TTS

The codec-LM approach discretizes audio and imports the entire LLM toolbox. Continuous denoising takes the opposite bet: speech is continuous, keep it continuous, and learn a path from Gaussian noise to real audio. This section builds that picture from first principles.

### Gaussian noise: why it's the universal starting point

Every denoising model starts from the same place: sample per-element random noise from a standard Gaussian:

```python
z_noise = torch.randn_like(z_clean)   # same shape as the latent we want to produce
```

Why Gaussian specifically? It's the maximum-entropy distribution for a given variance — meaning it's the most "uncertain" prior you can have while still being mathematically tractable. It has closed-form expressions for everything you need during training, and by the Central Limit Theorem it's what you get when you add many independent noise sources. Practically: there exists a well-defined path from any `z_noise` to any `z_clean`, and the math to describe that path is manageable.

### Mixing noise and data: the interpolation formulation

At any training step, you construct a "partially noisy" version of a real audio latent `z_clean` at time `t`:

```
z_t = α_t × z_clean + σ_t × ε,    ε ∼ N(0, I)
```

`α_t` and `σ_t` are time-dependent coefficients: at `t=0` you have pure data (`α=1, σ=0`); at `t=1` you have pure noise (`α≈0, σ≈1`). A concrete example with `z_clean = [0.8, −0.3, 1.2, −0.9, 0.5]`, `t=0.6`, `α_t=0.7`, `σ_t=0.5`, `ε=[−0.2, 0.8, −0.5, 1.1, 0.3]`:

```
z_t = 0.7 × [0.8, −0.3, 1.2, −0.9, 0.5]
    + 0.5 × [−0.2, 0.8, −0.5, 1.1, 0.3]
    = [0.46, 0.19, 0.59, −0.08, 0.50]
```

The network's job: given `z_t` and `t`, predict either the noise that was added (DDPM) or the velocity pointing from noise toward data (flow matching).

### DDPM: predict the noise

DDPM (Ho et al., arXiv:2006.11239) trains a network `ε_θ(z_t, t, conditioning)` to predict the added noise:

```python
loss = ((ε_actual - ε_θ(z_t, t, conditioning)) ** 2).mean()
```

Inference reverses this: start from pure noise `z_T ~ N(0, I)` and repeatedly subtract predicted noise. The catch: sampling is **stochastic** — fresh Gaussian noise is re-injected at every denoising step. This is mathematically required (reversing a Stochastic Differential Equation, not an ODE). The stochasticity stabilizes the reverse process but curves the trajectory: taking large steps causes errors that compound, so DDPM needs 50–1000 NFE (network function evaluations) to produce clean output. DDIM (Song et al., arXiv:2010.02502) removes per-step noise injection and enables ~50 NFE, but the path remains curved.

### Flow matching: straighten the path

Flow matching (Lipman et al., arXiv:2210.02747; Rectified Flow, Liu et al., arXiv:2209.03003) makes one insight: if the training paths from noise to data are **straight lines**, the trajectory becomes trivially integrable in 2–10 steps.

Training:

```python
z_0 = torch.randn_like(z_clean)          # noise sample
z_1 = z_clean                            # real audio latent
t = torch.rand(batch_size, 1, 1)         # t ∈ [0, 1] uniformly

z_t = (1 - t) * z_0 + t * z_1           # linear interpolation
v_target = z_1 - z_0                     # constant velocity along this line

v_pred = velocity_net(z_t, t, conditioning)
loss = ((v_pred - v_target) ** 2).mean()
```

`v_target = z_1 - z_0` is constant along each path — the network is regressing a fixed vector, not a function that changes with `t`. This makes optimization easier, and crucially the resulting ODE trajectory is straight:

```python
# Flow matching inference
z = torch.randn_like(target_latent)
for t in torch.linspace(1.0, 0.0, steps=8):
    v = velocity_net(z, t, conditioning)
    z = z - (1.0 / 8) * v              # Euler step; 8 NFE is typically enough
# z ≈ z_clean — decode to audio
```

**Stochasticity.** DDPM is a stochastic process (SDE): fresh Gaussian noise is re-injected at every sampling step, meaning the trajectory is different on every run even from the same starting point. You can make it reproducible by carefully seeding the RNG before every step, but the operational discipline required is non-trivial across 50–1000 iterations. Flow matching is a deterministic ODE — randomness lives only in the initial `z_0`. Fix that seed once and the same noise always produces the same audio. For production voice systems, this is a significant operational advantage.

### DDPM vs Flow Matching

| | DDPM | Flow Matching |
|---|---|---|
| Path shape | Curved (SDE) | Straight (ODE) |
| Network predicts | Noise `ε` | Velocity `v = data − noise` |
| Sampling | Stochastic (fresh noise each step) | Deterministic |
| Typical NFE | 50–1000 | 2–10 |
| Edge inference | Impractical | Practical |
| TTS examples | AudioLDM, NaturalSpeech 2 | Supertonic, F5-TTS, Voicebox |

### Why streaming is structurally hard for continuous denoising

This is the sharpest practical difference between the two paradigms.

AR generation is causally ordered — each token depends only on past tokens:

```
token_0 → token_1 → token_2 → ...
    ↓          ↓          ↓
  decode     decode     decode
    ↓          ↓          ↓
  play       play       play       ← play as you go, 80ms per Mimi frame
```

Flow matching generates all frames simultaneously in each NFE pass:

```
NFE pass 1:  [frame_0 noisy, frame_1 noisy, ..., frame_N noisy]
NFE pass 2:  [frame_0 refined, ...]
...
NFE pass 8:  [frame_0 clean, frame_1 clean, ..., frame_N clean]
                              ↓
                      decode all frames
                              ↓
                            play
```

Frame 0 at NFE pass 1 is essentially random noise — far from its final value. Its final value depends on what frames 50–100 look like during refinement, because the denoiser's attention sees the full output context at every pass. There is no "first frame ready" event during refinement. You must complete all N passes before any audio is playable.

**Practical streaming workarounds for flow-matching TTS:**

- **Sentence-level pipelining:** generate sentence N+1 while sentence N plays. Works when RTF ≪ 1 — Supertonic at ~0.012× RTF generates 1s of audio in ~12ms, so the player is never starved. Latency is one sentence, not one frame.
- **Chunked synthesis with crossfade:** generate fixed-length chunks independently, crossfade at boundaries. Loses cross-chunk prosody coherence.
- **Consistency-model distillation** (CALM-style): train the model to match the full denoiser's output in 1–2 NFE, sacrificing some quality ceiling for near-streaming capability.

**TTFA comparison on a 5-second utterance:**
- AR codec-LM (Moshi, 12.5fps): TTFA ≈ 80ms — first frame plays immediately
- Flow matching at 0.012× RTF: synthesis time = 5 × 0.012 = 60ms, then full audio delivered — similar TTFA, but no audio plays until synthesis completes

The TTFA numbers converge for short utterances. The difference emerges on long utterances: AR latency per frame stays constant; flow-matching synthesis time grows linearly with utterance length.

### When each paradigm wins

| Use case | Paradigm | Reasoning |
|---|---|---|
| Conversational AI, full-duplex | AR (Moshi, Pocket TTS) | Native per-frame streaming, sub-100ms TTFA, token-level conditioning on live user audio |
| Long-form narration, audiobooks | Flow matching (Supertonic, F5-TTS) | RTF ≪ 1, quality ceiling higher than AR for long utterances, sentence pipelining hides latency |
| Voice cloning, zero-shot | AR (Qwen3-TTS, MARS6) | Style prefix conditioning is most natural in AR; flow models require reference audio in the latent space |
| Edge deployment, smallest footprint | Kokoro-82M or Pocket TTS | One-pass NAR is simplest; Pocket TTS if codec quality is needed |
| Multilingual without fine-tuning | Supertonic 3 (31 languages) or Fish Speech 1.5 | Supertonic has widest coverage; Fish Speech if Hindi or other Indic languages are needed (Firefly codec cross-lingual alignment) |
| Seed-reproducible outputs | Flow matching | Deterministic after fixing initial noise; AR sampling requires seeding at every step |

**Current Hindi/Indic reality.** Supertonic 3 covers 31 languages but no Hindi. Qwen3-TTS covers 10 languages but no Hindi. Moshi and most AR systems are English-centric. Hindi support in any of these requires a full training run or fine-tuning on target-language audio — not just text-encoder adaptation. This is the actionable gap that the Appendix addresses.

---

## 12. Sampling for TTS: how inference differs from text LLMs

Codec-based TTS uses the same sampling primitives as text generation — temperature, top-p, greedy — but the implications are different enough that they're worth discussing separately.

### Temperature

In text generation, low temperature (0.3–0.5) makes output deterministic and repetitive; high temperature (1.2–1.5) makes it creative but incoherent. The optimal range for most tasks is 0.7–1.0.

In TTS, the picture splits by codebook level. For level-1 tokens (speaker identity, prosody, phoneme structure), **low temperature is critical** — typically 0.7–0.85. High temperature at level 1 causes the model to hallucinate wrong phonemes or produce incoherent prosody. Too low (below 0.5) is the other failure mode: every sentence collapses to the same flat, mid-pitch delivery — the statistical mode of the training distribution. For fine-level tokens (levels 4–8), temperature matters much less because the perceptual distance between nearby tokens is small — two level-7 tokens that both pass the perceptual acceptance criterion in the speculative decoding section are essentially interchangeable. If you're using AR+NAR, you can apply temperature only to the AR stage and use greedy decoding (temperature=0) on the NAR pass without audible quality loss.

```python
# AR stage — temperature matters, use 0.8
coarse_tokens = ar_model.generate(
    text_tokens, speaker_emb,
    temperature=0.8, top_p=0.95
)

# NAR stage — greedy is fine
fine_tokens = nar_model(coarse_tokens, text_tokens).argmax(-1)
```

### Top-p (nucleus) sampling

Top-p sampling at the text LM level restricts sampling to the smallest set of tokens whose cumulative probability exceeds p. At p=0.95, this typically means sampling from 10–50 tokens on any given text step.

For level-1 codec tokens the distribution is narrower — the model is usually quite confident about which phoneme comes next given the text. Top-p at 0.9–0.95 is a safe default. The bigger risk in TTS is not incoherence (the usual text-LLM concern) but *prosodic monotony* — the model always picks the highest-probability prosody pattern and every sentence sounds identical. Slightly higher top-p (0.97–0.99) or a small temperature boost helps introduce natural variation.

### Repetition penalty

This one matters more for TTS than for text generation. Codec tokens for silence or near-silence are high-probability fallback tokens at every frame — if the model gets confused about where a sentence should end, it can produce a stream of silence tokens that sounds like a hung process. A moderate repetition penalty (1.1–1.3) on the last 20–50 tokens prevents the model from looping on silence or on a repeated phoneme cluster.

In practice:
```python
# Typical TTS sampling config
sampling_params = {
    "temperature": 0.8,        # AR stage only
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "repetition_penalty_window": 30,  # look back 30 tokens
    "min_new_tokens": 10,       # prevent early EOS on short inputs
    "max_new_tokens": 1500,     # ~120s at 12.5fps — hard stop
}
```

### Why greedy decoding fails for TTS

Greedy decoding (always pick the highest-probability token) collapses prosody. The model has learned a distribution over prosody patterns; the single highest-probability pattern is roughly "flat, mid-pitch, medium-speed" — the statistical average of training data. Every sentence comes out with the same intonation, which sounds robotic within a paragraph even if each sentence sounds fine in isolation.

This is the TTS-specific analogue of the "neural text is bland" problem: the decoder's most likely output is a collapsed average, not a natural sample. Nucleus sampling or temperature > 1.0 on the prosody-encoding level-1 tokens is what restores natural variation. The intuition is the same as why diversity-promoting sampling helps creative text generation, just applied to a different modality.

---

## 13. Open questions

**Background: what the LM backbone already knows.** Every system above uses a pretrained LM backbone — Helium (Moshi/CALM), Llama-3 (Orpheus), Qwen3 (Qwen3-TTS) — trained on massive multilingual text corpora. Fine-tuning for TTS teaches the backbone *acoustics*, not language: it already knows Hindi phonology, syntax, and morphology from pretraining; the fine-tuning teaches it what Hindi text sounds like in codec token space. Fish Speech demonstrated this for 13 languages — if the codec's level-1 codebook has reasonable cross-lingual phoneme alignment, you need surprisingly little target-language audio. The LM knows what to say; the codec tokens teach it what that sounds like. This context motivates the language-specific questions below.

**Token consistency.** A known failure mode of codec-based TTS that barely gets
discussed: the same text, same speaker, same model can produce different codec
tokens on different runs — because sampling temperature and the multi-modal
distribution interact in unpredictable ways. This manifests as inconsistent
prosody across sentences in a long paragraph. Worth investigating before
designing any production TTS system; the problem is real but I haven't seen a
thorough published treatment of mitigations.

**Streaming hierarchical codecs.** SNAC's multi-rate structure is the right
inductive bias for offline generation. Making it streaming — where coarse
tokens emit immediately and fine tokens are filled from a small lookahead
buffer by a non-causal model — would combine SNAC quality with Mimi latency.
Nobody has published this.

**Mamba in the AR stage.** The AR backbone is the latency bottleneck in every
system I've described. Mamba's O(T) complexity and causal recurrent state make
it a natural candidate for replacing the Transformer in streaming TTS. A few
hybrid Transformer+Mamba architectures have shown competitive perplexity on
text — the question is whether the same holds for codec token distributions,
which have different autocorrelation structure than text.

**Pocket TTS GGUF.** The model is small enough to run on RPi 5. As of this
writing, the export path doesn't exist. This is an engineering problem, not a
research problem — but it's blocking a genuinely useful deployment.

**Hindi TTS open questions.** Three specific questions I want to answer before
committing to a fine-tuning approach for Qwen3-TTS:

1. Does Qwen3-TTS's semantic codebook (the 25Hz single-codebook track)
   generalize to Hindi phonemes out of the box, or does the codebook need
   fine-tuning?

2. Do retroflex consonants (/ɖ/, /ʈ/, /ɳ/) cluster near their dental
   equivalents (/d/, /t/, /n/) in the codec's level-1 space? This would
   determine how much acoustic data is needed to teach the model the
   phonological distinction.

3. Is the MTP module language-agnostic once the semantic codebook has been
   fine-tuned, or does it need separate fine-tuning for each language?

I don't have answers yet. These are the experiments I want to run.

---

## Summary

| System | Codec | Token strategy | Key innovation | Edge viable |
|---|---|---|---|---|
| VALL-E | EnCodec | Flat interleaved AR | First codec LM TTS | ❌ |
| MusicGen | EnCodec | Delay pattern | Parallel multi-head prediction | ❌ |
| MARS6 | SNAC | AR + NAR | Style prefix conditioning | ⚠️ M5 Pro |
| Fish Speech | Firefly | Two-pass AR | Multilingual codec alignment | ⚠️ M5 Pro |
| Qwen3-TTS | Custom dual-track | AR + MTP | Flow-matching mel decoder | ✅ M5 Pro (1.7B) |
| Moshi | Mimi | Delay + inner monologue | Full-duplex, text-before-audio | ✅ M5 Pro (q4) |
| CALM | VAE latents | Consistency MLP | No RVQ, ~10M audio head | ✅ M5 Pro |
| Pocket TTS | Mimi-derived | Distilled 6-layer | 100M params, 6× real-time | ✅ RPi 5 (soon) |

*Edge viability criteria: ❌ requires A100-class GPU for real-time inference; ⚠️ runs on M5 Pro at bf16 but may need quantization for consistent real-time; ✅ achieves ≥4× real-time on M5 Pro (bf16 or q4) or ≥1× real-time on RPi 5 with quantization.*

**Which system for your use case:**
- **Streaming TTS, lowest first-packet latency** → Moshi (q4) or Pocket TTS once GGUF lands
- **Best offline quality on a single consumer GPU** → Qwen3-TTS 1.7B (AR + MTP + flow-matching mel decoder)
- **Multilingual zero-shot, minimal fine-tuning data** → Fish Speech 1.5 (Firefly codec cross-lingual alignment)
- **Clean reference implementation to learn from** → MARS6 (SNAC + AR+NAR, MIT license, well-documented)
- **Edge deployment, lowest memory footprint today** → CALM (consistency MLP eliminates the 701M RQ-Transformer)

The through-line: every architectural decision is about where to spend your
AR budget. AR is expensive, sequential, and necessary for semantic coherence.
Everything else — fine acoustic detail, multi-level codec refinement,
vocoder pass — can and should be parallelized. The systems that internalize
this most aggressively (Qwen3-TTS MTP, CALM consistency MLP, Pocket TTS
distillation) are also the ones closest to viable edge deployment.

The next post will go from TTS into true speech-to-speech — how Moshi's
full-duplex architecture works end-to-end, how LFM2.5-Audio replaces the codec
decoder with a learned neural detokenizer, and what the latency math actually
looks like when you try to run either of these on an M5 Pro or Raspberry Pi 5.

---

## References

- [VALL-E — Wang et al., 2023](https://arxiv.org/abs/2301.02111)
- [VALL-T — Du et al., 2024](https://arxiv.org/abs/2401.14891)
- [MusicGen — Copet et al., 2023](https://arxiv.org/abs/2306.05284)
- [MARS6 — Baas et al., 2025](https://arxiv.org/abs/2501.05787)
- [Fish Speech — Fish Audio, 2024](https://github.com/fishaudio/fish-speech)
- [Qwen3-TTS — Alibaba, 2026](https://arxiv.org/abs/2601.15621)
- [Moshi — Défossez et al., 2024](https://arxiv.org/abs/2410.00037)
- [CALM — Kyutai, 2025](https://arxiv.org/abs/2509.06926)
- [Speculative decoding for TTS — Nguyen et al., 2024](https://arxiv.org/abs/2410.13839)
- [VADUSA — Li et al., 2024](https://arxiv.org/abs/2410.21951)
- [SSD — Lin et al., 2025](https://arxiv.org/abs/2505.15380)
- [Coarse-grained acceptance — arXiv:2511.13732](https://arxiv.org/abs/2511.13732)
- [SNAC — Siuzdak, 2024](https://github.com/hubertsiuzdak/snac)
- [mlx-audio-train — akashicMarga, 2025](https://github.com/akashicMarga/mlx-audio-train)
- [Qwen3-TTS official fine-tuning — QwenLM, 2026](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [SupertonicTTS — arXiv:2503.23108](https://arxiv.org/abs/2503.23108)
- [Flow Matching for Generative Modeling — Lipman et al., 2022](https://arxiv.org/abs/2210.02747)
- [Rectified Flow — Liu et al., 2022](https://arxiv.org/abs/2209.03003)
- [DDPM — Ho et al., 2020](https://arxiv.org/abs/2006.11239)
- [DDIM — Song et al., 2020](https://arxiv.org/abs/2010.02502)
- [AudioLDM — Liu et al., 2023](https://arxiv.org/abs/2301.12503)

---

*Animations generated with [Manim Community](https://www.manim.community/). Render scripts: `delay_pattern.py`, `ar_nar_split.py`, `moshi_inner_monologue.py`, `token_arrangements.py`, `training_pipeline.py`, `fish_speech_pipeline.py`, `calm_architecture.py` in `/assets/animations/`.*

---

## Appendix: Fine-tuning Qwen3-TTS on Apple Silicon with MLX

A few weeks before writing this post, I kept coming back to Qwen3-TTS. The architecture — AR backbone predicting a semantic codebook, MTP filling 15 acoustic layers in one shot, flow-matching mel decoder — felt like the cleanest expression of everything this post tries to explain. So I decided to test a specific question: what happens when you give it natural Indian language phonemes it was never trained on?

Hindi has retroflex consonants (/ɖ/, /ʈ/, /ɳ/) that don't appear in any of Qwen3-TTS's 10 training languages. These are produced by curling the tongue tip back to touch the roof of the mouth — acoustically similar to their dental counterparts (/d/, /t/, /n/) but perceptually distinct to any native Hindi speaker. The base model, predictably, either mapped them to the nearest English phoneme or produced something that sounded like a hesitation. The gap was obvious within the first few words.

That experiment is what led me to build [mlx-audio-train](https://github.com/akashicMarga/mlx-audio-train) — a fine-tuning framework for Qwen3-TTS that runs natively on Apple Silicon, without a CUDA cluster. The goal: teach the model what Hindi sounds like in codec space, starting from the Qwen3 backbone's strong Hindi text knowledge, and do it overnight on an M5 Pro 64GB.

The official Alibaba fine-tuning script (`sft_12hz.py`) assumes a CUDA environment and updates all 1.7B parameters. Two specific departures make the MLX version work on Apple Silicon.

### Departure 1: LoRA instead of full fine-tuning

mlx-audio-train applies **LoRA** to the attention and feedforward projections inside the `talker` submodule — the part of Qwen3-TTS that maps text and speaker context to codec tokens. The rest of the model (text encoder, speaker encoder, MTP module) stays frozen.

```python
# Target modules — applied recursively inside talker only
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # feedforward
]
# rank=8, alpha=16 for language adaptation
# rank=16, alpha=32 for speaker cloning (higher expressivity needed)
```

Only `lora_a` and `lora_b` accumulate gradients — roughly 1–2% of model weights. The adapter checkpoint is ~23–46MB instead of 1.7GB. Because the base weights stay frozen, they can be loaded in **8-bit quantization** (QLoRA): the base occupies ~800MB at int8 instead of ~3.4GB at bf16, with the LoRA deltas in bf16 on top. This is the difference between fitting comfortably in 64GB unified memory and thrashing.

A learning rate of 2e-5 that would risk destabilizing a full fine-tune is safe with LoRA — the low-rank constraint acts as implicit regularization on the parameter update.

### Departure 2: first codebook only as training input

The official training uses all 16 codec levels. Each frame in the codec channel is built as a sum of 16 embeddings — one per RVQ level — fed as the decoder input. mlx-audio-train trains on only the **first codebook** as input, trusting the MTP module to fill levels 2–15 at inference exactly as it does for the base model.

The first codebook is where speaker identity and phoneme structure live (covered in detail in the codec post). Fine-tuning the LM to predict level-1 tokens correctly for Hindi is the critical adaptation. Levels 2–15 are acoustic refinement the MTP module handles parametrically — introducing new gradients there risks disturbing the parallel prediction heads without quality upside.

### Two pipelines, different targets

**Language adaptation** (`configs/qwen3_tts_hindi.yaml`) — teaches the model what a new language sounds like in codec space:

```bash
# 1. Pre-tokenize audio offline (avoids loading the speech tokenizer during training)
python scripts/preprocess_dataset.py \
    --config configs/qwen3_tts_hindi.yaml   # writes *.codec.npy alongside each wav

# 2. Train — LoRA rank 8, cosine LR, grad accum 16 (effective batch 32)
python scripts/train.py --config configs/qwen3_tts_hindi.yaml

# 3. Evaluate: before/after comparison
python scripts/demo.py --adapter checkpoints/qwen3-hindi/checkpoint-best
```

Data format — one sample per line, no reference audio required:

```jsonl
{"audio": "data/hindi/clip.wav", "text": "नमस्ते दुनिया"}
```

Voice identity comes from a reference clip at inference time. The fine-tuning only teaches acoustic mappings for Hindi phonemes, not a particular speaker.

**Speaker cloning** (`configs/qwen3_tts_speaker.yaml`) — captures a specific voice:

```bash
# JSONL includes ref_audio per sample
# {"audio": "clip.wav", "text": "...", "ref_audio": "speaker_ref.wav"}

python scripts/train.py --config configs/qwen3_tts_speaker.yaml   # rank 16, 3 epochs

# Bake mean speaker embedding into codec_embedding[3000]
python scripts/bake_speaker_embedding.py \
    --config     configs/qwen3_tts_speaker.yaml \
    --checkpoint checkpoints/qwen3-speaker/checkpoint-final \
    --output     checkpoints/qwen3-speaker/custom_voice

# Inference — no ref_audio needed at runtime
from mlx_audio.tts.utils import load_model
model = load_model("checkpoints/qwen3-speaker/custom_voice")
audio = model.generate(text="Hello")
```

The bake step averages speaker embeddings across all training references and writes the mean into `talker.model.codec_embedding.weight[3000]` — the reserved custom-voice slot in Qwen3-TTS's codec embedding table. The config is patched to `tts_model_type: custom_voice`, telling inference to use that slot without a reference clip.

### Comparison with official fine-tuning

| | mlx-audio-train | Official `sft_12hz.py` |
|---|---|---|
| Framework | MLX (Apple Silicon) | PyTorch + Accelerate (CUDA) |
| Fine-tuning method | LoRA (rank 8–16) | Full fine-tune (all 1.7B params) |
| Base model precision | 8-bit quantized | bfloat16 |
| LoRA delta precision | bfloat16 | — |
| Adapter checkpoint | ~23–46 MB | ~1.7 GB (full model) |
| Codec input levels | Level 1 only | All 16 (additive embeddings) |
| Effective batch size | 16–32 (bs 2 × accum 8–16) | 8 (bs 2 × accum 4) |
| Speaker baking | Separate post-training script | Written at end of training loop |
| Target hardware | M1–M5 Pro/Max | NVIDIA A100/H100 |

Language adaptation for Hindi runs overnight on an M5 Pro 64GB. Inference RTFX is unchanged from the base model — LoRA weights can be fused into the base at serving time, so the deployed model is identical in speed to the unmodified Qwen3-TTS 1.7B.
