---
layout: post
title: "Teaching a Speech Model to Judge Itself: GRPO Post-Training for TTS on Apple Silicon"
date: 2026-08-08
categories: speech deep-learning audio
author: Akash Singh
description: >
  Part three of the speech series. SFT taught a Qwen3-TTS LoRA adapter to speak
  Hindi, then plateaued. This post uses GRPO — the same verifiable-reward RL
  behind math and code models — to optimize CER and naturalness directly on an
  M5 Pro, and reports the negative results alongside the wins.
tags: [GRPO, RLVR, TTS, RL, post-training, Qwen3-TTS, Hindi, MLX, LoRA, DNSMOS]
---

The last post ended in an appendix: a LoRA adapter that taught Qwen3-TTS to
speak Hindi. It worked — and then it stopped getting better. This post is about
what you do when supervised fine-tuning has given you everything it can.

> **Prerequisites:** This is the third post in the series — *representation*
> ([From Spectrograms to Speech Codecs](/speech/deep-learning/audio/2026/04/11/Speech-Codecs-Deep-Dive.html)),
> *generation* ([Inside the Speech LM](/speech/deep-learning/audio/2026/05/20/Inside-the-Speech-LM.html)),
> and now *refinement*. It assumes the codec-token picture from the first post
> (RVQ, level-1 vs. fine levels) and the Qwen3-TTS AR+MTP architecture and the
> LoRA SFT setup from the second post's appendix. If "the talker predicts
> codebook 0 autoregressively and the MTP module fills levels 1–15 in one shot"
> is familiar, you're ready.

---

## Contents

1. [Where SFT plateaus](#1-where-sft-plateaus)
2. [Why GRPO — and where it sits among RL-for-LLM techniques](#2-why-grpo--and-where-it-sits)
3. [Anatomy of a training step](#3-anatomy-of-a-training-step)
4. [The reward stack](#4-the-reward-stack)
5. [The alignment problem: when the model games the judge](#5-the-alignment-problem)
6. [The KL reference bug](#6-the-kl-reference-bug)
7. [Results — including a live open question in GRPO itself](#7-results)
8. [What's still open](#8-whats-still-open)

---

## 1. Where SFT plateaus

The Hindi adapter from the last post works. Feed it Devanagari text and it
produces fluent, correctly-articulated Hindi — the retroflex consonants that
the base model mangled now land where a native speaker expects them. LoRA rank
8 on the talker's attention and feedforward projections, ~1–2% of the weights,
trained overnight on an M5 Pro. As a proof that the Qwen3 backbone already knew
Hindi *text* and only needed to be taught what that text *sounds like* in codec
space, it's conclusive.

And then the curves flatten. You keep training and the cross-entropy loss keeps
dropping — slowly, but it drops — while the thing you actually measure at
evaluation time stops moving. On my held-out Hindi set the SFT adapter settles
around **CER ≈ 0.205** and sits there. More epochs buy a lower training loss and
nothing measurable on eval CER. The intelligibility you care about has decoupled
from the objective you're optimizing.

<figure>
  <img src="/assets/images/grpo-animations/sft_plateau.gif"
       alt="Two curves against training steps: cross-entropy training loss keeps drifting down while eval CER flatlines at 0.205"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    The two curves that motivate this whole post. Cross-entropy training loss (blue) keeps falling — the optimizer is doing its job. But eval CER (orange), computed by running ASR on the generated audio, drops early and then pins at ~0.205. The objective and the metric have decoupled: more optimization of the proxy buys nothing on the thing you actually evaluate.
  </figcaption>
</figure>

This is not a bug in the training loop. It's the objective doing exactly what
it was told, which turns out to be the wrong thing.

### Cross-entropy optimizes a proxy, not the goal

SFT minimizes token-level cross-entropy: at each frame, maximize the log-
probability of the *one* codebook-0 token that appeared in the ground-truth
recording, given everything before it. That's teacher forcing. The gradient
only ever sees the reference token; it never sees what the model would actually
sample, and it never scores a *sequence* — only individual next-token guesses
against a single "correct" answer.

But there is no single correct answer in speech. The same Hindi sentence has an
enormous set of perfectly intelligible, perfectly natural renderings — different
pitch contours, different micro-timing, different emphasis, all mapping to
different codebook-0 sequences (this is the multi-modality point from the second
post). Cross-entropy treats every one of those alternatives that *isn't* the
recorded token as an error to be pushed down. It's optimizing "match this
specific recording token-by-token," when what I actually want is "be
intelligible and sound natural."

The two are correlated — matching real recordings does tend to produce
intelligible speech, which is why SFT works at all — so the plateau arrives
exactly when cross-entropy has extracted all the intelligibility that correlates
with token-matching. Worse, the metric I evaluate on is computed by a completely
different process: **run an ASR model on the generated audio, compare its
transcript to the input text, count the character errors.** Nothing in
cross-entropy training ever runs the decoder, produces audio, or transcribes it.

So the question that opens this post is simple: what if we optimized the metric
directly? Instead of "make the model assign high probability to the recorded
token," what if the training signal were "generate the audio, transcribe it,
and reward the model when the transcript matches"? That's a different kind of
training loop — one that samples, scores, and updates — and building it
correctly, on a laptop, is what the rest of this post is about.

---

## 2. Why GRPO — and where it sits

"Generate, score, update" is reinforcement learning. The moment you commit to
optimizing a metric you can only compute *after* producing a full output, you've
left supervised learning behind — there's no per-token label to regress against,
only a scalar that rates the finished sample.

That leaves two independent choices, and it's worth keeping them apart because
they're usually conflated: **where the reward comes from**, and **how you turn
that reward into a gradient**. GRPO is an answer to the second. RLVR is an
answer to the first. This setup uses both, for different reasons.

### The reward is computed, not learned

In RLHF as originally practiced — InstructGPT, the early chat models — you can't
write down "good response" as a function, so you *learn* one: collect human
preference judgments, train a reward model to imitate them, then optimize the
policy against that learned model. The reward model is a neural network, it's
expensive to train, and it can be gamed (the policy finds inputs where the
reward model is wrong).

The reasoning-model wave (DeepSeek-R1, and the math/code RL that preceded it)
threw that out for a large class of problems. If you're training on math, you
don't need a learned judge — you have a *verifier*. Did the final answer equal
the ground truth? Did the code pass the unit tests? The reward is a
deterministic program, not a model. This is **RLVR — RL with Verifiable
Rewards** — and its whole appeal is that a programmatic reward can't be gamed
the way a learned reward model can, because it isn't approximating a judgment,
it *is* the judgment.

TTS turns out to fit this frame almost exactly. I don't need a learned model of
"good Hindi speech." I have verifiers:

- **Intelligibility** — run ASR on the generated audio, compute the character
  error rate against the input text. `1 − CER` is a program.
- **Speaker similarity** (for the voice-cloning pipeline) — cosine distance
  between a speaker embedding of the output and one of the reference clip.
  Another program.

These stand in for the human judge the way "does it pass the tests" stands in
for a code reviewer. The catch — which gets its own section — is that a verifier
is only a *proxy* for what you actually want, and proxies can be gamed even when
they can't be "learned around." Hold that thought until §5.

### The policy is just an LM

The second reason this works cleanly: the thing I'm training has the same shape
as a text LLM. The Qwen3-TTS talker predicts codebook-0 tokens autoregressively
— `P(codec_token_t | codec_{<t}, text, speaker)` — which is structurally
identical to `P(word_t | word_{<t})`. A rollout is a sampled sequence of discrete
tokens; a "prompt" is the text to be spoken; a "completion" is the codec-token
sequence. Every piece of GRPO's group-rollout machinery was built for exactly
this object. I'm swapping word-pieces for codec indices and a math checker for an
ASR pass; the RL scaffolding is untouched.

### Three ways to turn reward into a gradient

| | Advantage baseline | Rollout reuse | Extra model in memory | Needs |
|---|---|---|---|---|
| **PPO** | learned critic | yes, multi-step → needs clipping | +1 critic (≈ policy size) | scalar reward |
| **DPO** | n/a — pairwise objective | n/a | frozen reference only | *paired* preferences |
| **GRPO** | group mean of G rollouts | no — strictly on-policy | none | scalar reward |

DPO drops out immediately on the last column. It needs *paired* preference data
— for each prompt a "chosen" and a "rejected" completion — and optimizes the
model to prefer one over the other. This setup produces an *absolute scalar
reward per rollout* (a CER, a cosine), so using DPO would mean throwing away the
numeric reward to manufacture pairs from it. That's a GRPO/RLVR-shaped problem,
not a preference-optimization one.

PPO is the real alternative.

### What GRPO drops

**The critic.** PPO trains a second network — the same size as the policy — to
estimate the expected reward from each state, so it can compute a per-token
advantage. GRPO replaces that whole network with a *statistic*: sample a group of
G completions for the same prompt, and use the group's own mean reward as the
baseline. The advantage of rollout `i` is just how far its reward sits above the
group mean:

```python
# train/grpo/rewards.py — the critic, replaced by a groupwise z-score
groups = reward.reshape(-1, group_size)
mean   = groups.mean(axis=1, keepdims=True)
adv    = groups - mean
if adv_norm == "std":
    adv = adv / (groups.std(axis=1, keepdims=True) + eps)   # DeepSeek default
```

No second network, no critic training, no critic to go stale. On a 64GB laptop,
*not* holding a second 1.7B model in memory isn't a nicety — it's the difference
between the run fitting and not.

**The importance-ratio clip.** PPO's signature `clip(π_θ/π_old, 1±ε)` exists for
one reason: PPO takes *several* gradient steps on the same batch of rollouts,
because rollouts are expensive and you want to amortize them. After the first
step the policy has moved, so those rollouts are now *off-policy* — sampled from
a stale policy — and the clip is what keeps that reuse from blowing up.

I don't reuse rollouts. The setup is **strictly on-policy**: sample a batch of
rollout groups, take one gradient step, throw the rollouts away, sample again.
In `train/grpo/trainer.py` it falls out of a single knob:

```
Set grad_accumulation = prompts_per_step (B): one optimizer step
== B prompt groups == B·G rollouts, the standard strictly-on-policy
GRPO update.
```

The rollouts sampled during one accumulation window are always on-policy with
respect to the weights that produced them, because the step happens *after* the
window closes. If you never go off-policy, the entire reason for the importance
ratio and its clip evaporates — it's not that I disabled clipping, there's
simply no stale distribution for it to correct.

To be clear about the trade: rollout sampling dominates step time (§3), so reuse
would genuinely buy something. I'm declining it because reuse is what forces the
importance ratio, the clipping, and the off-policy correctness questions that
come with them — at this scale, one fewer class of bugs is worth more than the
throughput.

### Inspiration, and what doesn't transfer

The push to try this on speech at all came from [*Multi-Faceted Interactivity
Alignment in Full-Duplex Speech Models*](https://arxiv.org/abs/2606.11167)
(Ohashi, Zeghidour, Défossez and Kharitonov — three of them Moshi authors),
which post-trains Moshi and PersonaPlex for *interactivity* with RL: axis-specific
rewards shaping when the model takes turns, backchannels, and yields the floor in
a live conversation. I want to be
straight about what transfers and what doesn't. Their **problem** doesn't
transfer: turn-taking is a property of a two-party conversation over time, and
single-utterance TTS has no turns to take. What transfers is the **structural
insight** — that a speech token-LM is an RL-able policy like any other, that you
can define a computable reward over its decoded audio, and that on-policy
group-relative updates are a stable way to push on it. That insight is
algorithm-shaped, not task-shaped, which is why it survives the move from
turn-taking to intelligibility.

---

## 3. Anatomy of a training step

One GRPO step is two phases with completely different characters. Phase A
generates and judges; no gradients exist anywhere in it. Phase B is an ordinary
teacher-forced forward/backward that happens to be weighted by what Phase A
learned. Almost all the wall-clock time is Phase A — sampling dominates, and the
ASR pass that everyone assumes is the bottleneck is only ~3% of step time.

<figure>
  <img src="/assets/images/grpo-animations/grpo_step.gif"
       alt="One GRPO step: a prompt fans out into four sampled rollouts, each decoded to audio, transcribed by Whisper, scored to a CER, converted to a reward, then to a group-relative advantage that drives the policy update"
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    One prompt → G=4 sampled codec-token rollouts → decode each to audio → transcribe → CER → reward. The group's own mean reward is the baseline (no critic), so advantages are just how far each rollout sits above or below its siblings. Phase B then pushes up the rollouts that beat the group and down the ones that didn't, with a KL term anchoring everything to the frozen SFT reference.
  </figcaption>
</figure>

### Phase A: rollout, no gradients

For one prompt, sample G rollouts, decode them to waveforms, and score them.
The defaults are `group_size=4`, `temperature=0.9`, `top_p=0.95`, `top_k=50`,
`max_new_tokens=240` frames. The temperature matters more than it looks: GRPO
learns from *disagreement within the group*, so sampling has to be hot enough
that four rollouts of the same prompt actually differ. At temperature 0.3 the
group collapses to four near-identical sequences, every advantage is ~0, and the
step teaches nothing.

The whole phase is three calls in `train/grpo/trainer.py`:

```python
out = sample_rollouts_interleaved(
    self.model, prompt["text"], lang_code=lang, group_size=G,
    max_new_tokens=self.max_new_tokens, temperature=self.temperature,
    top_p=self.top_p, top_k=self.top_k,
    compute_ref=True, ref_params=self.ref_params,
)
audios = decode_codes_to_audio(self.model, out["full_codes"], out["codec_mask"])
scored = combine_rewards(
    audios, [prompt["text"]] * G, out["gen_lengths"].tolist(),
    self.max_new_tokens, self.reward_cfg, sample_rate=self.sample_rate,
    model=self.model, ref_mel=prompt.get("ref_mel"),
)
```

**Only codebook 0 is the policy.** This is the single most important
implementation fact in the whole setup. The talker samples cb0 autoregressively;
codebooks 1–15 come from the *frozen* `code_predictor` at decode time. That
mirrors the SFT regime from the last post — level 1 carries phoneme structure and
speaker identity, levels 2–16 are acoustic refinement — and it means the reward,
computed on fully-decoded 16-codebook audio, is attributed entirely to the cb0
choices the policy actually made. The gradient never touches the predictor.

Rewards then become group-relative advantages, and one guard runs before Phase B
is allowed to happen at all:

```python
# Zero-variance group: every rollout drew the same reward → advantages are
# all 0 → no policy-gradient signal (only KL would act).
if self.skip_zero_variance and float(reward.std()) < self.zero_var_eps:
    self._n_skipped += 1
    return None
```

If all four rollouts score identically — common early on, when every rollout is
degenerate and every CER is capped at 1.0 — the advantages are all zero and the
only surviving term is KL, which would pull the model toward the reference for
no reason. Standard DeepSeekMath practice is to drop those groups. Phase A is
already paid for, but skipping saves the Phase B forward/backward and, more
importantly, avoids diluting the accumulation window with groups that teach
nothing. The skip *rate* is logged per window, because a run can look perfectly
alive while 60% of its groups are being thrown away.

### Phase B: the update

Now gradients. Re-run the sampled tokens through the policy teacher-forced,
gather the log-probability of each token that was actually sampled, and weight
it by that rollout's advantage. The entire objective is four lines of
`train/losses/grpo_loss.py`:

```python
logp = gather_token_logprobs(logits, codec_ids)      # [G, T] = logπ_θ(sampled)

# advantage-weighted NLL — the only term that optimises the reward
pg = _masked_reduce(-(adv * logp), mask, n_valid, pg_norm)

# k3 KL(π_θ ‖ π_ref): exp(d) − d − 1,  d = logπ_ref − logπ_θ  (≥0)
d  = mx.clip(mx.stop_gradient(ref_logprobs) - logp, -kl_clip, kl_clip)
kl = _masked_reduce(mx.exp(d) - d - 1.0, mask, n_valid, pg_norm)

loss = pg + kl_beta * kl
```

Read `pg` as: for a rollout that beat its group (`adv > 0`), minimizing
`-adv·logπ` raises the likelihood of exactly those tokens; for one that lost,
the sign flips and the same tokens get pushed down. That's the whole learning
signal. Everything else is an anchor.

Two numerical details that are not optional. The KL uses the **k3 estimator**
(`exp(d) − d − 1`) rather than the naive log-ratio, because k3 is unbiased and
guaranteed non-negative — a raw mean log-ratio can go negative on a finite
sample and hand you a "reward" for drifting. And the log-ratio is clamped to
±10 before the exponential: a token sampled deep in the tail produces a large
`d`, and `exp(d)` will take the gradient with it. Relatedly,
`gather_token_logprobs` casts to float32 before `log_softmax` regardless of the
logits dtype — bfloat16 `log_softmax` is wildly inaccurate in the
low-probability tail, which is precisely where sampled-token log-probs live.

One knob controls how per-token quantities are reduced: `pg_norm` is either
`"token"` (global token mean, the known GRPO length bias) or `"sequence"` (every
rollout weighted equally regardless of length, the Dr. GRPO / DAPO correction),
applied identically to the PG and KL terms so `kl_beta` keeps the same meaning
across modes. That knob comes back in §7.

And the §2 argument is now concrete: Phase B consumes exactly the rollouts Phase
A produced, under the weights that produced them, then discards them. No second
gradient step, no stale distribution, nothing to clip.

### The layout trap

Here's the wrinkle that cost me the most time, and it's specific to doing this on
a TTS model rather than a text LLM.

Phase A samples through one code path. Phase B recomputes log-probs through
another. **If those two paths condition the model differently, `logπ_θ` in
Phase B is not the probability of what Phase A actually sampled** — you're
weighting the wrong distribution by the advantage, and the gradient is subtly,
silently wrong. Nothing crashes. The loss goes down. The model gets worse.

Qwen3-TTS has two legitimate conditioning layouts, and they are not
interchangeable:

- **Concatenated** — `[text_embeds | codec_prefix | codec_embeds]`, one packed
  sequence. This is what SFT trains on.
- **Interleaved** — what `model.generate()` actually does at inference: text
  streams in frame-by-frame via `trailing_text_hidden`, and all 16 codebook
  embeddings sum together as the feedback signal for the next frame.

The first version used concatenated for both phases, on the reasoning that
matching the SFT training layout was what mattered. The alternative is to match
the *deployment* path instead — sample via `generate()` and make Phase B replay
it exactly. That works because the input embeddings come from frozen tables
(codec embeddings, text projection), so they're constants with respect to the
LoRA parameters: you can record the per-frame input embeddings during sampling
and simply feed them back in Phase B, guaranteeing the two phases agree by
construction rather than by careful reimplementation.

```python
if "tf_input_embeds" in batch:
    # Interleaved: replay the recorded generate()-path input embeds
    # (frozen, so constant w.r.t. LoRA); grad flows through the transformer.
    logits_full, hidden = talker(batch["tf_input_embeds"])
    codec_offset = int(batch["cb0_offset"])
    logits = logits_full[:, codec_offset: codec_offset + T, :]
else:
    # Concatenated (SFT-matched): rebuild the forward from ids.
    logits, codec_offset, hidden = grpo_codec_logits(
        model, batch["text_ids"], codec_ids, lang_codes, spk_embeds=spk_embeds
    )
```

Which of those two bets was right is an empirical question with an unambiguous
answer, and it's the first thing §7 reports.

---

## 4. The reward stack

Everything the model learns comes through the scalar that Phase A hands to Phase
B. Getting that scalar right is most of the work.

Each reward is a named component registered in `train/grpo/rewards.py`, and the
total is a plain weighted sum over whichever components a run's YAML activates:

```
total_i = Σ_name  weight(name) · r_name,i
```

| Reward | Pipeline | Signal | Default weight |
|---|---|---|---|
| `intelligibility` | 1 (language) | `1 − err(ASR(audio), text)` via mlx-whisper; CER for Hindi | 1.0 |
| `speaker_similarity` | 2 (cloning) | `cosine(speaker_encoder(mel), ref_mel)` | 0.0 |
| `length_penalty` | both | −1 if no EOS; silence, rate and overrun penalties | 0.5 |
| `naturalness` | both | DNSMOS P.835, rescaled `(MOS−1)/4` | 0.0 |
| *KL* | both | per-token penalty — **not** a reward | β = 0.02–0.1 |

A weight of 0 deactivates a component entirely, which is why the table's
defaults are a design statement rather than a config detail. Intelligibility
drives learning. Length is a guard. Speaker similarity and naturalness are off
unless a run deliberately turns them on — and in the naturalness case, that
choice is the whole subject of §5.

### Why CER, not WER

Both are computed on every rollout — they're cheap string operations sitting
next to an ASR pass that dominates their cost — but only one drives the reward.
For Hindi it has to be CER, and the reason is orthographic.

Devanagari marks vowels on consonants with *matras* — diacritics attached above,
below, or beside the base character. कल (*kal*) and काल (*kaal*) differ by a
single matra and mean "yesterday/tomorrow" and "time." A model that gets the
vowel length slightly wrong has made a one-character error that flips the entire
word: 100% error on that token under WER, one edit in three under CER.

For evaluation that would be a reporting preference. For RL it's structural.
GRPO learns from *spread within a group* — if four rollouts all score 1.0
because each mangled a different matra, the advantages are zero and the step is
discarded by the zero-variance guard from §3. WER saturates early in training,
exactly when you most need signal; CER degrades gracefully and keeps producing
usable gradient.

Text normalization runs on both sides before scoring — NFC, strip punctuation
including the Devanagari danda (।॥), collapse whitespace, lowercase — so the
metric reflects phonetic content rather than punctuation the TTS never voices.

### Shaping the error rate

Mapping an error rate to a reward is not as innocent as it looks:

```python
def _shape_intel_reward(err, reward_shape, reward_k):
    if reward_shape == "tanh":
        return float(1.0 - np.tanh(reward_k * err))
    return 1.0 - min(1.0, err)
```

The linear form has flat sensitivity and a hard clamp at `err ≥ 1`, which hurts
at both ends of training. Early on, rollouts routinely land above 1.0 (ASR
hallucinating extra text on garbage audio) and the clamp flattens them all to
0.0 — no spread, no signal. Late on, hovering around CER 0.12, it gives almost
no contrast between a good rollout and a slightly better one.

`tanh` fixes both ends: it stretches within-group spread roughly 2.6× at
CER ≈ 0.12, and it saturates smoothly instead of clamping, so insertions with
`err > 1` are handled gracefully. Break-even against linear is around
CER ≈ 0.38. §7 has the ablation that settled this.

### Speaker similarity, and an honest note about it

Pipeline 2 (voice cloning) adds a cosine similarity between the speaker
embedding of the generated audio and that of the reference clip, reusing the
frozen `speaker_encoder` already loaded for SFT. Degenerate rollouts — near-
silent, or short enough to overflow the mel — are guarded to the worst possible
similarity (−1.0) rather than allowed to emit a NaN that would poison the whole
group's advantages through the mean and std.

The honest finding: **it barely does anything.** Once a loader bug was fixed
(the reference was being passed as a path string instead of a 24 kHz waveform),
speaker similarity saturated near 0.995 cosine almost immediately. The base
model already clones voices near-perfectly, so the term functions as a guard
against the policy *losing* the voice while chasing CER, not as a learning
driver. Intelligibility remains the signal that actually moves the model.

### The degeneracy guard

`length_penalty` is the only component that is purely negative — a penalty, never
a bonus. It exists because an unconstrained CER reward has some deeply
unattractive optima, and it accumulates four terms: a flat −1 if generation hit
the token cap without emitting EOS; a silence penalty when more than 60% of the
tail is below −40 dBFS; a speaking-rate floor; and a graded over-length penalty
proportional to how far voiced duration exceeds what the text warrants. The last
two close reward-hacking routes specifically, and both measure *voiced* duration
with trailing silence excluded, so they target articulation rather than the
padding the silence term already covers. Why those routes need closing is §5.

### KL is not a reward

The KL term is in every diagram of this system, but it is structurally different
from everything above, and the table marks it separately for a reason: it never
enters `combine_rewards`. It lives in the loss, as its own term with its own
coefficient:

```python
loss = pg + kl_beta * kl
```

That distinction matters. A KL folded into the reward sum would be
group-normalized along with everything else — it would become relative,
tradeable against CER, and its meaning would shift with the group's variance. As
a separate per-token loss term it does exactly one job: penalize drift from the
frozen SFT reference, everywhere, at a rate set by β (0.02–0.1; 0.05 is the
default).

It's an anti-degradation anchor. GRPO on a narrow reward will happily walk the
model somewhere the reward likes and everything else hates, and the SFT adapter
is a known-good point in weight space that took a full training run to reach.
β controls how far the policy is allowed to wander from it. Set it too high and
nothing moves; too low and the run finds a degenerate optimum and stays there.

Which raises the obvious question — what stops the model from finding a
degenerate optimum the reward *does* like? That's the next section.

---

## 5. The alignment problem: when the model games the judge

RLVR's appeal, from §2, was that a programmatic reward can't be gamed the way a
learned reward model can. That claim needs an asterisk, and this section is the
asterisk.

### The verifier is a proxy, not ground truth

In math RL the verifier is close to ground truth: the answer either equals 42 or
it doesn't. Speech has no such luck. My verifier is a Whisper transcript, and
what I actually care about is "would a Hindi speaker find this natural and
clear?" Those are different questions, and the reward only ever sees the first
one.

So there are two layers of slippage. ASR-CER is a proxy for intelligibility, and
intelligibility is a proxy for quality. **Optimizing hard against a proxy pushes
the model toward wherever the proxy and the true objective disagree** — that's
not a speech-specific problem, it's the general shape of reward hacking, and
having a programmatic reward doesn't exempt you from it. It only means the
failure mode is legible in advance rather than emergent from a learned model's
quirks.

The verifier has its own fragilities, too. Whisper's default decoding retries
with a temperature fallback, and on degenerate audio it will happily hallucinate
long repetitive transcripts — a different CER on every pass over the same clip.
A nondeterministic verifier is a noisy teacher, so the ASR call pins
`temperature=0.0` and `condition_on_previous_text=False`: the reward for a given
waveform is now deterministic, and degenerate rollouts stop getting re-decoded
six times each.

### What gaming would look like

Take a sentence — आज मौसम अच्छा है। ("the weather is nice today"). Here are two
ways to say it.

**Natural**, the way a person would, about 2.1 seconds:

```
आज मौसम अच्छा है।
```

**Over-articulated**, every syllable isolated and stretched, with pauses between
them, about 4.3 seconds:

```
आ····ज····मौ····स····म····अ····च्छा····है
```

The words are identical. Whisper transcribes both correctly. And that's the
entire problem: **the second one scores marginally *better* on CER**, because
isolated, over-enunciated syllables are easier for an ASR model to segment than
natural connected speech with coarticulation and reduction.

<figure>
  <img src="/assets/images/grpo-animations/reward_hacking.gif"
       alt="The same Hindi sentence rendered two ways: natural at 2.1 seconds, and over-articulated at 4.3 seconds with isolated syllables. CER is 0.04 versus 0.03 — the robotic version scores better — while speaking rate halves and DNSMOS drops from 3.27 to 2.41."
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    Same words, same transcript, twice the duration. CER cannot separate these two — it slightly prefers the robotic one — while speaking rate halves and DNSMOS collapses. A reward built only on <code>1 − CER</code> would walk the model steadily toward the bottom panel. (Illustrative of the failure mode; §5's finding is that it did not occur in this run.)
  </figcaption>
</figure>

So `1 − CER` doesn't merely fail to *penalize* the bottom panel — it actively
prefers it. Every step, the group-relative advantage would nudge the policy
further toward slower, flatter, more separated speech while the reward reported
improvement; you'd watch CER fall for hours and end up with a model nobody wants
to listen to.

That's an alignment problem rather than a tuning problem. The reward isn't wrong;
it's radically incomplete — it measures one real thing (can this be understood?)
and is blind to the other (does this sound like a person?). GRPO will find that
blind spot considerably faster than I will.

### Instrumenting the catch

The defense is to measure the thing the reward is blind to, and *not* optimize
it. `naturalness` (DNSMOS P.835, reference-free) is registered as a reward but
ships at weight 0 — computed and logged every step, never in the gradient.

That restraint is the entire point. If DNSMOS were in the reward, the policy
would optimize it too, and it could no longer serve as an independent check —
I'd be asking the same signal to both teach and grade. Held out of the
objective, it becomes a witness.

The signature of reward hacking is then a specific, falsifiable pattern:

> **CER improves while DNSMOS degrades.** That would mean the model bought
> transcribability with naturalness — exactly the over-articulation trade.

Speaking rate is the second witness, and a more direct one, because
over-articulation has an unavoidable physical signature: it takes longer. The
length reward already measures characters per second of *voiced* audio, so a
drift downward in cps is the over-enunciation hypothesis stated in a number.

### What actually happened

The gaming pattern did not appear.

| Signal | SFT baseline | After GRPO | Read |
|---|---|---|---|
| CER | 0.205 | 0.183 | improved (−11% rel.) |
| DNSMOS OVRL | 3.237 | 3.271 | up, not down |
| DNSMOS SIG | 3.573 | 3.608 | up, not down |
| Speaking rate | 7.9 cps | 7.7–8.2 cps | held |

DNSMOS moved *with* CER rather than against it, and speaking rate stayed put
across the configuration sweep instead of sliding downward. Both witnesses point
the same way: this run improved intelligibility without trading away
naturalness.

I want to be careful about how much that carries. **+0.034 DNSMOS is small.**
The honest reading is not "GRPO improved quality" — it's "GRPO improved CER and
quality did not degrade, with a slight positive drift." The value of the
measurement is the *absence of the gaming signature*, which is a weaker and more
useful claim than a quality win.

And DNSMOS itself is a proxy with a known defect here: **it's English-trained.**
Using it on Hindi makes it a relative indicator — fine for "did this get worse
than it was," unreliable as an absolute MOS. Catching reward hacking with a
metric that may not fully understand the language is exactly the kind of
reassurance that deserves a second opinion, which is why §7 brings in an
independent multilingual benchmark rather than resting on this table.

### The babbling failure mode

The other degenerate optimum has nothing to do with articulation: rollouts that
simply never stop. The model wanders past the end of the sentence, and because
the ASR transcript still contains the target text somewhere inside the babble,
CER can stay deceptively low.

The original guard was a binary cliff — hit the 240-frame cap without emitting
EOS, take a flat −1. That catches the extreme case and nothing else. A rollout
that runs 1.8× longer than the text warrants but *does* terminate gets no
penalty at all, while one that grazes the cap takes the full hit. Across most of
the range where over-generation actually lives, the guard is silent, and where it
does fire it's a step function with no gradient to descend.

So the cliff was upgraded to a graded penalty: compute the expected voiced
duration from the character count, allow a tolerance multiplier, and then ramp
the penalty in proportion to the overrun, saturating at the far end.

```python
if length_target_cps > 0:
    expected_s = n_chars / length_target_cps
    # overrun = 1.0 exactly at the tolerance edge, >1 past it.
    overrun = voiced_s / (expected_s * length_overrun_tol)
    if overrun > 1.0:
        pen -= overrun_penalty * min(1.0, overrun - 1.0)
```

Now every degree of over-generation carries a proportional cost, and the policy
gets a dense signal telling it *which direction* is better rather than a single
tripwire at the boundary. It's the same lesson as the `tanh` reward shaping in
§4: for RL, a smooth signal that discriminates between two mediocre outcomes is
worth more than a sharp one that only fires at the extreme.

---

## 6. The KL reference bug

This one deserves its own section because the obvious implementation is one line,
looks correct, runs without error, and is wrong in two independent ways at once.

<figure>
  <img src="/assets/images/grpo-animations/kl_reference_bug.gif"
       alt="Left: disabling LoRA gives a base-model reference in bf16, and KL explodes. Right: a frozen fp32 snapshot of the SFT adapter, and KL starts at zero and stays bounded."
       style="max-width:100%;border-radius:6px;background:#0d1117;" />
  <figcaption style="text-align:center;color:#6e7681;font-size:0.85em;margin-top:6px;">
    The same KL term, two references. Disabling the LoRA adapter anchors to the base model — a policy that can't speak Hindi — and does so through a bf16 path while the policy runs fp32, so KL is already nonzero before training starts. Snapshotting the SFT adapter instead keeps both sides on the identical fp32 path: KL is exactly 0 at initialization and measures only adapter drift.
  </figcaption>
</figure>

### What the reference is actually for

The KL term anchors the policy to a reference so GRPO can't wander somewhere the
narrow reward likes and everything else hates (§4). That makes the choice of
reference a design decision, not an implementation detail: **whatever you anchor
to is the model you're saying is worth staying near.**

For this setup the answer should be obvious in hindsight. The SFT adapter is the
known-good point — it took an overnight run to reach, and it's the thing GRPO is
supposed to *refine*, not replace. So π_ref should be the SFT policy.

### The one-line version that looks right

LoRA makes an alternative reference seem free. The adapter is additive, so you
can recover "the model before fine-tuning" by just switching it off — and a
`lora_disabled` context manager is already sitting in the codebase for exactly
that. Wrap the reference forward in it, done:

```python
with lora_disabled(model):                 # ← the bug
    ref_logits, *_ = grpo_codec_logits(model, ...)
```

This produces numbers. Training runs. Nothing warns you.

### Two independent failures

**It anchors to the wrong policy.** Disabling LoRA gives you the *base* model —
Qwen3-TTS before it learned any Hindi. The KL term is now pulling the policy back
toward a model that mangles retroflex consonants, every single step. The anchor
isn't stabilizing the run; it's actively fighting the SFT adaptation I spent a
night producing. Turn β up to stabilize training and you turn up the force
undoing your own fine-tune.

**It silently changes dtype.** The disabled path ran bfloat16 while the enabled
(policy) path ran float32. So the two forwards differ *even with identical
weights*, purely from precision — and that disagreement compounds through 28
transformer layers into roughly 3 nats of log-prob divergence.

The second failure is the more insidious one, because of what it does to the k3
estimator from §3. With `d ≈ 3`, `exp(d) − d − 1` is around 16 **per token** —
enormous next to the policy-gradient term, so KL dominates the loss. And the
gradient is trying to close a gap that the policy *cannot* close, because the gap
isn't adapter drift; it's a rounding difference between two numeric paths. The
optimizer will happily distort the adapter chasing it. (This is also why the ±10
log-ratio clamp exists — it bounds the damage, but it doesn't make a wrong
reference right.)

### The invariant that catches it

There's a clean tell here, and it's worth internalizing beyond this bug:

> If the reference is "where training started," then at step 0 the reference
> *is* the policy, so **KL must be exactly 0.** Not small. Zero.

That's a quantity with a known analytic value at a known point — a free
assertion. Any nonzero KL at initialization means the two paths disagree about
something, and disagreement at step 0 can only come from the setup, never from
learning. Logging KL and looking at the first step would have caught this
immediately; it took much longer because a large-but-decreasing KL curve looks
plausible if you assume the reference is correct.

The general lesson: whenever an RL setup holds a frozen copy of a model, verify
the reference path and the policy path are identical at initialization — same
weights *and* same precision. A "disabled" code path is exactly the kind of thing
that quietly runs in a different dtype, because nothing errors when it does.

### The fix

Don't disable anything. Deep-copy the SFT adapter weights once at GRPO start and
temporarily install them for the reference forward, leaving the code path
otherwise untouched:

```python
def lora_snapshot(model: nn.Module) -> Dict[str, mx.array]:
    """Deep-copy the current LoRA adapter weights — the frozen GRPO reference.

    Take this once at GRPO start (the SFT-trained adapters) and feed it to every
    rollout via `lora_swapped`. The KL term then measures drift from SFT, on the
    same float32 forward the policy uses.
    """
    return {k: mx.array(v) for k, v in get_trainable_params(model).items()}
```

Taken once in the trainer (`ref_params = lora_snapshot(model)`), then swapped in
per rollout:

```python
with lora_swapped(model, snapshot):
    ref_logits, _, _ = grpo_codec_logits(
        model, text_ids, codec_ids, lang_codes, spk_embeds=spk_embeds
    )
    ref_logprobs = mx.stop_gradient(gather_token_logprobs(ref_logits, codec_ids))
```

Same LoRA path, same float32, same everything — only the adapter *values* differ,
and at initialization they don't differ at all. KL starts at exactly 0, and from
then on it measures the one thing it was supposed to measure: how far the policy
has drifted from SFT.

---

## 7. Results — including a live open question in GRPO itself

A note on reading the numbers below, because there are three different evaluation
protocols in this section and cross-comparing them would be a mistake. The
validated run and the configuration sweep use **different held-out sets and
different scoring rules** — the sweep caps per-sentence CER at 1.0, which removes
the hallucinated-transcript outliers that inflate an uncapped mean, so its SFT
baseline reads 0.163 where the validated run's reads 0.205. Same adapter, same
direction of travel, different rulers. The ablation in the third subsection is
different again. **Compare within a table, never across them.**

### The negative result: the concatenated layout lost

The v1 setup used the concatenated layout for both phases — the §3 bet that
matching the SFT *training* layout was what mattered.

It lost to plain SFT. Not "improved less than hoped": the GRPO-trained adapter
was worse than the checkpoint it started from, on the same held-out eval.

The reasoning behind the bet wasn't stupid — Phase A and Phase B genuinely did
agree, which was the constraint I'd identified as critical. But they agreed on a
pathway the model never uses at inference, so every improvement the policy
earned was improvement in the wrong coordinate system.

Switching to the interleaved layout — sampling through the real `generate()`
path and replaying the recorded input embeddings in Phase B — is what turned the
result positive. **The lesson isn't "make Phase A and Phase B agree." It's "make
them agree *on the deployment path*."**

### The validated run

Interleaved layout, held-out Hindi eval:

| Metric | SFT baseline | GRPO | Change |
|---|---|---|---|
| **CER** | 0.205 | **0.183** | **−11% relative** |
| DNSMOS OVRL | 3.237 | 3.271 | +0.034 |
| DNSMOS SIG | 3.573 | 3.608 | +0.035 |

Significance is where this gets interesting, and where I want to be careful:

- **Wilcoxon signed-rank: p ≈ 7e-4.** The improvement is robust in rank terms —
  sentence by sentence, GRPO wins far more often than chance.
- **Paired t-test: p ≈ 0.09.** The mean-based test does *not* clear 0.05.

Both are true and the gap between them is informative rather than embarrassing.
CER distributions have heavy right tails: a handful of catastrophic rollouts
(ASR hallucinating on a bad clip, CER > 1) dominate the mean and swamp a
consistent improvement across the other ninety-odd sentences. The rank test is
robust to that; the t-test isn't. So the honest claim is **"GRPO improves the
typical sentence, with high confidence"** — not "GRPO improves mean CER, with
high confidence." The naturalness read on the DNSMOS rows is §5's.

### The configuration sweep

A follow-up sweep at group size 4, 200 steps, 100 held-out sentences, 3 seeds,
CER capped at 1.0. The two knobs are `pg_norm` (`token` vs `sequence`, the
length-bias question from §3) and the SFT-mixin weight `sft_lambda`:

| Configuration | CER | Δ vs SFT | Relative | Wilcoxon p |
|---|---|---|---|---|
| SFT baseline | 0.163 | — | — | — |
| `sequence` + sft 0.1 | 0.122 | −0.041 | −25.4% | 2.5e-05 |
| `sequence` + sft 0 | 0.120 | −0.043 | −26.2% | 2.1e-09 |
| `token` + sft 0.1 | 0.119 | −0.045 | −27.2% | 6.7e-10 |
| **`token` + sft 0** | **0.114** | **−0.049** | **−29.8%** | **5.7e-09** |

Every cell beats SFT at p ≪ 1e-4. The spread between configurations (−25% to
−30%) is much narrower than the gap between *any* of them and the baseline,
which is the more useful takeaway: the method is doing the work, not a lucky
hyperparameter. DNSMOS stayed flat to slightly positive (3.28 → 3.29–3.33) and
speaking rate held at 7.7–8.2 cps against 7.9 — the §5 gaming check, repeated
across the sweep.

Two mild surprises. `token` normalization edged out `sequence` despite carrying
the known length bias that Dr. GRPO and DAPO argue against — at 240-frame caps
and short Hindi sentences there may simply not be enough length variance for the
bias to bite. And the SFT mixin *didn't help*: `sft_lambda = 0` beat `0.1` in
both pairs. The KL anchor was apparently sufficient regularization on its own,
and adding a cross-entropy pull toward the training distribution just fought the
reward.

### The `/std` question: real evidence on an unsettled debate

Vanilla GRPO (DeepSeekMath) normalizes advantages by the group's standard
deviation: `A = (r − mean) / (std + eps)`. **Dr. GRPO** (Liu et al.,
*Understanding R1-Zero-Like Training*) argues that `/std` introduces a
difficulty bias. Consider a group where all four rollouts score nearly
identically — dividing by a tiny std rescales those trivial differences up to
unit magnitude, so a low-information group ends up shouting as loudly as one with
genuine spread. DAPO makes a parallel argument about token-level length bias.
The proposed fix is to drop the `/std` and use `A = r − mean`.

The codebase supports both, so the question is answerable rather than
theoretical. A 150-step ablation crossing `adv_norm` with the §4 reward shaping
(**these CER values are on a different protocol again — read them only against
each other**):

| Configuration | CER best | CER final | KL |
|---|---|---|---|
| `std` + `linear` | 0.081 | 0.081 | 0.019 |
| **`std` + `tanh`** | **0.078** | **0.078** | **0.011** |
| `none` + `linear` | 0.109 | 0.341 | 0.99 |
| `none` + `tanh` | 0.090 | 0.183 | 0.40 |

Look at the `none` rows: **best CER and final CER diverge badly.** Those runs
reach a decent score and then fall apart. The failure is a late KL cliff around
step 125 — KL explodes, CER degrades, and the run ends worse than it peaked.

The obvious explanation is that dropping `/std` shrinks advantage magnitudes
roughly 10×, so the learning rate is simply mismatched. That explanation is
wrong, or at least incomplete: **the cliff persisted at 8× higher learning
rates.** It isn't a scaling problem you can tune away in this setting.

So for this problem — small groups, LoRA adapters, a bounded reward — vanilla
`/std` normalization is clearly the stable choice, and `std` + `tanh` (k=3) is
the recommended default. I want to be precise about the scope of that claim: it
is *not* a refutation of Dr. GRPO. Their argument is about large-scale reasoning
RL with large groups and highly variable difficulty, which is a different regime
in every dimension that matters. What this ablation contributes is a data point
from a corner of the space nobody was looking at — the `/std` term is genuinely
unsettled, and here removing it destabilized training in a way that higher
learning rates couldn't fix.

### TTSDS2: a second opinion on naturalness

§5's naturalness verdict rested on DNSMOS, which is English-trained — a real
weakness when the output is Hindi. So the check was repeated with **TTSDS2**, a
distributional benchmark whose multilingual factor set (mHuBERT-147, XLSR,
Whisper, Allosaurus) actually covers Hindi. 50 held-out sentences, IndicVoices-R
as the real-speech reference, 0–100:

| Factor | Real reference | SFT | GRPO |
|---|---|---|---|
| GENERIC | 99.3 | 90.7 | **92.7** |
| SPEAKER | 97.9 | 55.1 | 56.1 |
| PROSODY | 97.9 | 82.2 | *81.6* |
| INTELLIGIBILITY | 82.8 | 79.8 | **81.0** |
| **OVERALL** | **94.5** | 77.0 | **77.9** |

The corroboration that matters: GENERIC — how close the output distribution sits
to real speech — rose 90.7 → 92.7, independently confirming the DNSMOS-flat read
with a metric that isn't English-centric. Intelligibility rose too, which is the
axis GRPO was actually optimizing.

Two things I'm not going to paper over. **Prosody went down** (82.2 → 81.6). It's
a small move and I don't want to over-read a 0.6 change on 50 sentences, but it's
the one factor that moved the wrong way, and it's plausibly the axis most at risk
from optimizing intelligibility. **Speaker similarity is poor for both models**
(~55 against a 97.9 reference) — that's an artifact of this eval comparing
against reference speakers the adapter was never conditioned on, not a
regression, but it's a reminder that a 77.9 overall is a long way from the 94.5
that real speech scores.

---

## 8. What's still open

### What this run actually establishes

Stated as narrowly as the evidence allows: **on a Hindi LoRA adapter for
Qwen3-TTS, on-policy GRPO against an ASR-derived reward reduces character error
rate by 11–30% relative depending on protocol, robustly in rank terms, without
the naturalness degradation that the obvious reward-hacking failure mode would
produce — as measured by two automatic proxies.**

Everything load-bearing in that sentence is a measurement I can point at.
Everything outside it is not established, and the rest of this section is the
part I'd want a skeptical reader to hold me to.

### The gap that matters most: nobody has listened

There is no human evaluation. No Hindi-native MOS study, no A/B listening test,
no native speaker sitting down with the SFT and GRPO samples and saying which one
they'd rather hear.

This matters more here than it would in most papers, because of the specific
shape of the argument in §5. I was worried about optimizing against a proxy. My
defense was to check two *other* proxies — DNSMOS, which is English-trained, and
TTSDS2, which is distributional rather than perceptual. Neither is a human. So
the strongest honest statement is: **I looked for the signature of reward hacking
in the places I could measure, and didn't find it.** That is genuinely weaker
than "the model didn't degrade," and the gap between those two claims is exactly
the size of the listening test I haven't run.

There's a specific thing a native listener would catch that no metric here would:
whether the retroflex consonants that motivated this entire line of work survived
GRPO intact. CER is computed against a Whisper transcript, and if Whisper's Hindi
is itself weak on the dental/retroflex distinction, then a model that degrades
that contrast could be *rewarded* for it. Nothing in my stack would notice.

### What the reward stack can't see

**Prosody is unmeasured and moved the wrong way.** No component of the reward
scores intonation, phrasing, or emphasis. CER is prosody-blind by construction —
a monotone robot and an expressive speaker with identical transcripts score
identically. TTSDS2's prosody factor drifted down (82.2 → 81.6), which is small
enough to be noise on 50 sentences and directionally exactly what you'd predict
from optimizing a prosody-blind objective. I don't know which it is. Answering
it needs either a prosody-aware reward term or a longer run to see whether the
drift compounds.

**The reward has a ceiling, and it's structural.** GRPO learns from within-group
spread. As the model improves, rollouts converge — four rollouts of the same
prompt all transcribe correctly, all score ~1.0, variance collapses, and the
zero-variance guard from §3 drops the group. The method quietly runs out of
signal as it succeeds. That's not hypothetical; it's a predictable consequence of
the design, and the instrumentation to watch it already exists — the per-window
`skip_ratio` climbing over training would be the tell. What signal comes *after*
CER saturates is an open design question, and I suspect it's where the
interesting work is.

**Speaker similarity was never really tested.** It saturated near 0.995 cosine
almost immediately (§4), so Pipeline 2 never became a learning driver — the term
functioned as a guard against losing the voice, not as a signal that shaped the
policy. Whether GRPO can *improve* cloning is untested; it would need a harder
setting (unseen speakers, cross-lingual transfer) where the base model doesn't
already saturate the metric.

### What one language and one scale can't settle

The `/std` result in §7 is real evidence, and it's also from a single regime:
group size 4, LoRA adapters, a bounded reward, 150–200 step runs. Dr. GRPO's
argument concerns large-scale reasoning RL with large groups and highly variable
problem difficulty. My ablation says removing `/std` destabilized *this* setup in
a way higher learning rates couldn't rescue. It does not say Dr. GRPO is wrong
about theirs.

Similarly, "match the deployment path, not the training layout" (§7) is the most
transferable-feeling finding in this post, and I've demonstrated it exactly once,
on one model. It should generalize to any codec LM whose training and inference
conditioning diverge — which is most of them — but that's a prediction, not a
result.

And everything here is Hindi. The CER-over-WER choice in §4 was argued from
Devanagari orthography specifically; the reasoning would need revisiting for a
language where word-level errors aren't dominated by single-character diacritics.
Whether the whole recipe transfers to other Indic languages, or to languages
without that property, is untested.

### Closing

The honest summary of this post is smaller than the headline number. GRPO moved
CER meaningfully, the improvement survives a rank-based significance test, and
the naturalness checks I could run came back clean or slightly positive. Against
that: a negative result on the first layout I tried, a bug that had the KL term
anchored to the wrong model in the wrong precision for longer than I'd like, a
prosody number pointing gently the wrong way, and no human in the loop anywhere.

What I'm more confident about than any individual number is the shape of the
approach. Post-training a speech model against a computable reward is not exotic
— the policy is an ordinary autoregressive LM, the reward is an ASR call, and
the whole thing fits on a laptop. The hard parts turned out not to be the RL
algorithm at all. They were the mundane, domain-specific things: which
conditioning path Phase A and Phase B agree on, which error metric survives
Devanagari, what your KL is actually anchored to, and whether the metric you're
using to prove you didn't cheat is one you also optimized.

That last one generalizes well beyond speech, and it's the piece I'd carry into
the next system: **hold something out of your objective so it can testify.**

---

## References

**RL and post-training**

- [GRPO / DeepSeekMath — Shao et al., 2024](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 — DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)
- [Dr. GRPO — *Understanding R1-Zero-Like Training*, Liu et al., 2025](https://arxiv.org/abs/2503.20783)
- [DAPO — Yu et al., 2025](https://arxiv.org/abs/2503.14476)
- [PPO — Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- [DPO — Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- [InstructGPT (RLHF) — Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)

**RL for speech**

- [Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models — Ohashi et al., 2026](https://arxiv.org/abs/2606.11167)
- [Moshi — Défossez et al., 2024](https://arxiv.org/abs/2410.00037)

**Evaluation**

- [TTSDS2 — Minixhofer et al., 2025](https://arxiv.org/abs/2506.19441)
- [TTSDS (original) — Minixhofer et al., 2024](https://arxiv.org/abs/2407.12707)
- [DNSMOS P.835 — Reddy et al., 2021](https://arxiv.org/abs/2110.01763)
- [Whisper — Radford et al., 2022](https://arxiv.org/abs/2212.04356)
- [mHuBERT-147 — Zanon Boito et al., 2024](https://arxiv.org/abs/2406.06371)
- [XLS-R — Babu et al., 2021](https://arxiv.org/abs/2111.09296)
- [Allosaurus — Li et al., 2020](https://arxiv.org/abs/2002.11800)

**Models and data**

- [Qwen3-TTS — Alibaba, 2026](https://arxiv.org/abs/2601.15621)
- [IndicVoices-R — Sankar et al., 2024](https://arxiv.org/abs/2409.05356)
- [mlx-audio-train — akashicMarga, 2025](https://github.com/akashicMarga/mlx-audio-train)

---

*Animations generated with [Manim Community](https://www.manim.community/). Render scripts: `sft_plateau.py`, `grpo_step.py`, `reward_hacking.py`, `kl_reference_bug.py` in `/assets/animations/`.*
