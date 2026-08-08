"""
Animation: One GRPO step for TTS.

Phase A (no grad): from one prompt, sample G=4 codec-token rollouts at temp 0.9,
decode each to audio, transcribe with Whisper, compute CER → reward, and turn the
group's rewards into group-relative advantages (no critic — the group mean IS the
baseline).

Phase B (update): advantage-weighted −logπ pushes up the good rollouts and down
the bad ones, with a KL term anchoring to the frozen SFT reference.
"""

from manim import *
import numpy as np


def mini_wave(color, seed, width=0.95, height=0.34):
    rng = np.random.default_rng(seed)
    n = 60
    xs = np.linspace(-width / 2, width / 2, n)
    env = np.exp(-((xs / (width * 0.55)) ** 2))
    ys = (0.5 * np.sin(xs * 22) + 0.5 * rng.standard_normal(n)) * env * (height / 2)
    pts = [np.array([x, y, 0]) for x, y in zip(xs, ys)]
    vm = VMobject(stroke_color=color, stroke_width=2.0)
    vm.set_points_as_corners(pts)
    return vm


def token_strip(colors, cell=0.19):
    row = VGroup()
    for i, c in enumerate(colors):
        sq = Square(side_length=cell, fill_color=c, fill_opacity=0.65,
                    stroke_color=c, stroke_width=1.0)
        sq.move_to([i * (cell + 0.03), 0, 0])
        row.add(sq)
    row.move_to(ORIGIN)
    return row


class GRPOStep(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("One GRPO step: sample a group, score it, update",
                     font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title))

        phaseA = Text("Phase A — rollout (no grad): sample G=4 · decode · transcribe · score",
                      font_size=15, color=GREY_B)
        phaseA.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(phaseA))

        # ── Prompt ────────────────────────────────────────────────────────────
        prompt_box = RoundedRectangle(corner_radius=0.08, width=1.9, height=1.05,
                                      fill_color=BLUE_E, fill_opacity=0.45,
                                      stroke_color=BLUE_B, stroke_width=1.6)
        prompt_box.move_to([-5.9, 0.35, 0])
        prompt_txt = Text("नमस्ते,\nआज मौसम…", font_size=15, color=WHITE, line_spacing=0.9)
        prompt_txt.move_to(prompt_box.get_center())
        prompt_lbl = Text("one prompt\n(text to speak)", font_size=12, color=GREY_A,
                          line_spacing=0.9)
        prompt_lbl.next_to(prompt_box, UP, buff=0.14)
        self.play(FadeIn(prompt_box), FadeIn(prompt_txt), FadeIn(prompt_lbl))

        # ── 4 rollout rows ────────────────────────────────────────────────────
        row_y = [1.65, 0.60, -0.45, -1.50]
        cers = [0.17, 0.24, 0.31, 0.18]
        rewards = [round(1 - c, 3) for c in cers]
        strip_palettes = [
            [TEAL_B, BLUE_B, GREEN_B, TEAL_C, BLUE_C, GREEN_C, TEAL_B, BLUE_B],
            [PURPLE_B, BLUE_B, TEAL_B, PURPLE_C, BLUE_C, TEAL_C, PURPLE_B, BLUE_B],
            [RED_B, ORANGE, MAROON_B, RED_C, ORANGE, MAROON_C, RED_B, ORANGE],
            [GREEN_B, TEAL_B, BLUE_B, GREEN_C, TEAL_C, BLUE_C, GREEN_B, TEAL_B],
        ]

        strips, waves, asr_boxes, cer_txts = [], [], [], []
        col_strip, col_wave, col_asr, col_cer = -3.7, -1.7, 0.15, 1.9

        for i, y in enumerate(row_y):
            strip = token_strip(strip_palettes[i]).move_to([col_strip, y, 0])
            strips.append(strip)

        strip_hdr = Text("cb0 rollout ~ temp 0.9", font_size=12, color=GREY_A)
        strip_hdr.move_to([col_strip, 2.35, 0])
        self.play(FadeIn(strip_hdr))
        # branch arrows prompt → strips
        branches = VGroup(*[
            Arrow(prompt_box.get_right(), s.get_left(), color=GREY_C,
                  stroke_width=1.6, buff=0.12, max_tip_length_to_length_ratio=0.05)
            for s in strips
        ])
        self.play(Create(branches), LaggedStart(*[FadeIn(s) for s in strips],
                                                lag_ratio=0.12), run_time=1.0)

        # decode → waveform
        for i, y in enumerate(row_y):
            w = mini_wave(strip_palettes[i][0], seed=i + 1).move_to([col_wave, y, 0])
            waves.append(w)
        wave_hdr = Text("decode 🔊", font_size=12, color=GREY_A).move_to([col_wave, 2.35, 0])
        dec_arrows = VGroup(*[
            Arrow(strips[i].get_right(), waves[i].get_left(), color=GREY_D,
                  stroke_width=1.4, buff=0.10, max_tip_length_to_length_ratio=0.10)
            for i in range(4)
        ])
        self.play(FadeIn(wave_hdr), Create(dec_arrows),
                  LaggedStart(*[Create(w) for w in waves], lag_ratio=0.12), run_time=1.0)

        # ASR (whisper) → CER
        for i, y in enumerate(row_y):
            b = RoundedRectangle(corner_radius=0.05, width=1.15, height=0.44,
                                 fill_color=GREY_E, fill_opacity=0.5,
                                 stroke_color=GREY_C, stroke_width=1.1)
            b.move_to([col_asr, y, 0])
            t = Text("Whisper", font_size=11, color=GREY_A).move_to(b.get_center())
            asr_boxes.append(VGroup(b, t))
        asr_arrows = VGroup(*[
            Arrow(waves[i].get_right(), asr_boxes[i].get_left(), color=GREY_D,
                  stroke_width=1.4, buff=0.10, max_tip_length_to_length_ratio=0.10)
            for i in range(4)
        ])
        self.play(Create(asr_arrows), LaggedStart(*[FadeIn(b) for b in asr_boxes],
                                                  lag_ratio=0.12), run_time=0.8)

        for i, y in enumerate(row_y):
            good = cers[i] < 0.22
            col = GREEN_B if good else (RED_B if cers[i] > 0.28 else YELLOW_C)
            t = Text(f"CER {cers[i]:.2f}", font_size=14, color=col).move_to([col_cer, y, 0])
            cer_txts.append(t)
        cer_arrows = VGroup(*[
            Arrow(asr_boxes[i].get_right(), cer_txts[i].get_left(), color=GREY_D,
                  stroke_width=1.4, buff=0.10, max_tip_length_to_length_ratio=0.12)
            for i in range(4)
        ])
        self.play(Create(cer_arrows), LaggedStart(*[FadeIn(t) for t in cer_txts],
                                                  lag_ratio=0.12), run_time=0.8)
        self.wait(0.4)

        # ── reward → group mean baseline → advantages ─────────────────────────
        self.play(FadeOut(phaseA))
        phaseA2 = Text("no critic — the group's own mean reward is the baseline",
                       font_size=15, color=GREY_B).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(phaseA2))

        r_txts = []
        col_r = 3.5
        for i, y in enumerate(row_y):
            t = Text(f"r = {rewards[i]:.2f}", font_size=14, color=WHITE).move_to([col_r, y, 0])
            r_txts.append(t)
        r_hdr = Text("reward = 1 − CER", font_size=12, color=GREY_A).move_to([col_r, 2.35, 0])
        conv = VGroup(*[
            Arrow(cer_txts[i].get_right(), r_txts[i].get_left(), color=GREY_D,
                  stroke_width=1.3, buff=0.10, max_tip_length_to_length_ratio=0.14)
            for i in range(4)
        ])
        self.play(FadeIn(r_hdr), Create(conv),
                  LaggedStart(*[FadeIn(t) for t in r_txts], lag_ratio=0.1), run_time=0.8)

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        baseline = DashedLine([col_r - 0.7, 0.05, 0], [col_r + 0.7, 0.05, 0],
                              color=YELLOW_B, stroke_width=1.8)
        base_lbl = Text(f"group mean = {mean_r:.2f}", font_size=13, color=YELLOW_B)
        base_lbl.next_to(baseline, DOWN, buff=0.08)
        self.play(Create(baseline), FadeIn(base_lbl))
        self.wait(0.3)

        # advantages column
        advs = [(rewards[i] - mean_r) / (std_r + 1e-4) for i in range(4)]
        adv_txts = []
        col_a = 5.6
        for i, y in enumerate(row_y):
            pos = advs[i] > 0
            col = GREEN_B if pos else RED_B
            sign = "+" if pos else ""
            t = Text(f"A = {sign}{advs[i]:.1f}", font_size=15, color=col).move_to([col_a, y, 0])
            adv_txts.append(t)
        adv_hdr = Text("advantage\n(r−mean)/std", font_size=12, color=GREY_A,
                       line_spacing=0.9).move_to([col_a, 2.4, 0])
        adv_arrows = VGroup(*[
            Arrow(r_txts[i].get_right(), adv_txts[i].get_left(), color=GREY_D,
                  stroke_width=1.3, buff=0.10, max_tip_length_to_length_ratio=0.14)
            for i in range(4)
        ])
        self.play(FadeIn(adv_hdr), Create(adv_arrows),
                  LaggedStart(*[FadeIn(t) for t in adv_txts], lag_ratio=0.12), run_time=0.9)
        self.wait(0.8)

        # ── Phase B: the update ───────────────────────────────────────────────
        keep = VGroup(prompt_box, prompt_txt, prompt_lbl,
                      *strips, *adv_txts, adv_hdr)
        self.play(*[FadeOut(m) for m in [
            strip_hdr, wave_hdr, r_hdr, adv_hdr, strip_hdr,
            branches, dec_arrows, asr_arrows, cer_arrows, conv, adv_arrows,
            *waves, *asr_boxes, *cer_txts, *r_txts, baseline, base_lbl, phaseA2,
        ]])

        phaseB = Text("Phase B — update: advantage-weighted −logπ, anchored by KL to the SFT reference",
                      font_size=15, color=GREY_B).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(phaseB))

        # policy box on the right
        policy = RoundedRectangle(corner_radius=0.08, width=2.2, height=1.0,
                                  fill_color=GREEN_E, fill_opacity=0.4,
                                  stroke_color=GREEN_B, stroke_width=1.8)
        policy.move_to([4.4, -0.2, 0])
        policy_lbl = Text("policy\n(LoRA on talker)", font_size=13, color=WHITE,
                          line_spacing=0.9).move_to(policy.get_center())
        self.play(FadeIn(policy), FadeIn(policy_lbl))

        # advantage-weighted updates: green pushes up, red pushes down
        up_arrows = VGroup()
        for i, t in enumerate(adv_txts):
            pos = advs[i] > 0
            col = GREEN_B if pos else RED_B
            a = Arrow(t.get_right(), policy.get_left() + [0, 0.35 - i * 0.22, 0],
                      color=col, stroke_width=2.2, buff=0.12,
                      max_tip_length_to_length_ratio=0.10)
            up_arrows.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in up_arrows], lag_ratio=0.12),
                  run_time=1.0)

        # objective (Text, not MathTex — keeps the render LaTeX-free like the
        # other scenes in this repo)
        obj = Text("ℒ  =  − Aᵢ · log πθ   +   β · KL( πθ ‖ π_ref )",
                   font_size=24, color=WHITE)
        obj.move_to([0.2, -2.85, 0])
        self.play(Write(obj))

        # KL anchor
        ref = RoundedRectangle(corner_radius=0.08, width=2.0, height=0.7,
                               fill_color=GREY_E, fill_opacity=0.45,
                               stroke_color=GREY_B, stroke_width=1.4)
        ref.move_to([4.4, 1.7, 0])
        ref_lbl = Text("frozen SFT reference", font_size=12, color=GREY_A).move_to(ref.get_center())
        kl_arrow = DashedLine(ref.get_bottom(), policy.get_top(), color=YELLOW_B,
                              stroke_width=1.8, dash_length=0.09)
        kl_lbl = Text("KL anchor", font_size=12, color=YELLOW_B).next_to(kl_arrow, RIGHT, buff=0.08)
        self.play(FadeIn(ref), FadeIn(ref_lbl))
        self.play(Create(kl_arrow), FadeIn(kl_lbl))
        self.wait(2.6)
