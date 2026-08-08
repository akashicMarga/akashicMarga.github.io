"""
Animation: What reward hacking would sound like.

Two renderings of the same Hindi sentence:
  - natural: normal rhythm, ~2.1 s
  - over-articulated: every syllable isolated with pauses, ~4.3 s

CER cannot tell them apart — the robotic one actually scores slightly BETTER,
because isolated syllables are easier for ASR. Speaking rate and DNSMOS collapse.
That gap is the reward-hacking route, and why CER alone is not a safe objective.

Numbers here are ILLUSTRATIVE of the failure mode, not measurements from the run
(§5 reports that this did not happen).
"""

from manim import *
import numpy as np


def segmented_wave(segments, total_s, width, color, seed=0, height=0.52):
    """Waveform built from (start_s, dur_s) voiced segments over `total_s`."""
    rng = np.random.default_rng(seed)
    n = 900
    t = np.linspace(0, total_s, n)
    amp = np.zeros(n)
    for (s0, dur) in segments:
        m = (t >= s0) & (t <= s0 + dur)
        if not m.any():
            continue
        local = (t[m] - s0) / max(dur, 1e-6)
        # smooth syllable envelope
        amp[m] = np.sin(np.pi * local) ** 0.7
    carrier = 0.55 * np.sin(t * 90) + 0.45 * rng.standard_normal(n)
    y = amp * carrier
    y = y / (np.abs(y).max() + 1e-9) * (height / 2)
    xs = np.linspace(-width / 2, width / 2, n)
    vm = VMobject(stroke_color=color, stroke_width=1.9)
    vm.set_points_as_corners([np.array([x, yy, 0]) for x, yy in zip(xs, y)])
    return vm


class RewardHacking(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("What reward hacking would sound like", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.30)
        sub = Text("same sentence · same words · very different speech",
                   font_size=15, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.12)
        self.play(Write(title), FadeIn(sub))
        self.wait(0.2)

        # Widths are proportional to real duration and share a left edge, so the
        # over-articulated version visibly TAKES LONGER — that lag is the physical
        # signature the speaking-rate guard keys on.
        X_LEFT = -4.40
        SEC_TO_X = 1.55          # scene units per second
        DUR1, DUR2 = 2.1, 4.3
        W1, W2 = DUR1 * SEC_TO_X, DUR2 * SEC_TO_X

        # ── Panel 1: natural ─────────────────────────────────────────────────
        lbl1 = Text("Natural", font_size=19, color=GREEN_B)
        lbl1.move_to([-5.85, 1.72, 0])
        dur1 = Text("2.1 s", font_size=13, color=GREY_A)
        dur1.next_to(lbl1, DOWN, buff=0.10)

        txt1 = Text("आज मौसम अच्छा है।", font_size=24, color=WHITE)
        txt1.move_to([X_LEFT + W1 / 2, 1.92, 0])

        segs1 = [(0.06, 0.38), (0.52, 0.52), (1.12, 0.54), (1.74, 0.30)]
        w1 = segmented_wave(segs1, DUR1, W1, GREEN_B, seed=3)
        w1.move_to([X_LEFT + W1 / 2, 1.10, 0])
        base1 = Line([X_LEFT, 1.10, 0], [X_LEFT + W1, 1.10, 0],
                     color=GREY_E, stroke_width=1.0)

        self.play(FadeIn(lbl1), FadeIn(dur1), FadeIn(txt1))
        self.play(Create(base1), Create(w1), run_time=1.1)

        m1 = VGroup(
            Text("CER  0.04", font_size=16, color=GREEN_B),
            Text("8.1 chars/sec", font_size=15, color=GREY_A),
            Text("DNSMOS  3.27", font_size=15, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        m1.move_to([4.75, 1.35, 0])
        self.play(FadeIn(m1))
        self.wait(0.35)

        sep = DashedLine([-6.9, 0.28, 0], [6.9, 0.28, 0], color=GREY_D, stroke_width=1.0)
        self.play(Create(sep))

        # ── Panel 2: over-articulated ────────────────────────────────────────
        lbl2 = Text("Over-articulated", font_size=19, color=RED_C)
        lbl2.move_to([-5.85, -1.05, 0])
        dur2 = Text("4.3 s", font_size=13, color=GREY_A)
        dur2.next_to(lbl2, DOWN, buff=0.10)

        txt2 = Text("आ····ज····मौ····स····म····अ····च्छा····है", font_size=22, color=WHITE)
        txt2.move_to([X_LEFT + W2 / 2, -0.85, 0])

        # every syllable isolated, long gaps between
        segs2 = [(0.10, 0.26), (0.72, 0.26), (1.34, 0.26), (1.96, 0.26),
                 (2.58, 0.26), (3.20, 0.26), (3.82, 0.26)]
        w2 = segmented_wave(segs2, DUR2, W2, RED_B, seed=9)
        w2.move_to([X_LEFT + W2 / 2, -1.65, 0])
        base2 = Line([X_LEFT, -1.65, 0], [X_LEFT + W2, -1.65, 0],
                     color=GREY_E, stroke_width=1.0)

        self.play(FadeIn(lbl2), FadeIn(dur2), FadeIn(txt2))
        self.play(Create(base2), Create(w2), run_time=1.3)

        m2 = VGroup(
            Text("CER  0.03", font_size=16, color=GREEN_B),
            Text("3.9 chars/sec", font_size=15, color=RED_C),
            Text("DNSMOS  2.41", font_size=15, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        m2.move_to([4.75, -1.40, 0])
        self.play(FadeIn(m2))
        self.wait(0.4)

        # ── The trap: CER prefers the robotic one ────────────────────────────
        cer_box1 = SurroundingRectangle(m1[0], color=YELLOW_B, buff=0.07, stroke_width=1.8)
        cer_box2 = SurroundingRectangle(m2[0], color=YELLOW_B, buff=0.07, stroke_width=1.8)
        self.play(Create(cer_box1), Create(cer_box2))

        trap = Text("the reward prefers\nthis one", font_size=14, color=YELLOW_B,
                    line_spacing=0.9)
        trap.next_to(cer_box2, UP, buff=0.34)
        arrow = Arrow(trap.get_bottom(), cer_box2.get_top(), color=YELLOW_B,
                      stroke_width=2.0, buff=0.06,
                      max_tip_length_to_length_ratio=0.30)
        self.play(FadeIn(trap), GrowArrow(arrow))
        self.wait(0.5)

        punch = Text("CER can't tell these apart — speaking rate and DNSMOS can",
                     font_size=18, color=GREEN_B)
        punch.to_edge(DOWN, buff=0.32)
        self.play(Write(punch))
        self.wait(2.8)
