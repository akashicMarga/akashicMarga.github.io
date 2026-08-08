"""
Animation: Where SFT plateaus — the objective and the metric diverge.

Two curves against training steps:
  - cross-entropy training loss: keeps drifting down (the objective)
  - eval CER: drops early, then flatlines at ~0.205 (the metric we care about)

The gap between "loss still falling" and "CER stalled" is the whole motivation
for post-training with a reward.
"""

from manim import *
import numpy as np


class SFTPlateau(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("Where SFT plateaus: the objective and the metric diverge",
                     font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.30)
        self.play(Write(title))
        self.wait(0.2)

        # ── Axes ──────────────────────────────────────────────────────────────
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 1.0, 0.2],
            x_length=9.2,
            y_length=4.6,
            axis_config={"color": GREY_C, "stroke_width": 2,
                         "include_numbers": False, "include_ticks": True},
            tips=False,
        )
        axes.move_to([0, -0.35, 0])

        x_lbl = Text("training steps →", font_size=16, color=GREY_B)
        x_lbl.next_to(axes.x_axis, DOWN, buff=0.18).align_to(axes.x_axis, RIGHT)
        y_lbl = Text("loss / CER", font_size=16, color=GREY_B)
        y_lbl.rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.10)
        self.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl))

        # ── Curves ────────────────────────────────────────────────────────────
        # cross-entropy: fast initial drop + a slow continued downward drift
        def ce(x):
            return 0.40 + 0.52 * np.exp(-x / 24.0) - 0.0012 * x

        # eval CER: drops, then pinned at 0.205
        def cer(x):
            return 0.205 + 0.32 * np.exp(-x / 8.0)

        ce_curve = axes.plot(ce, x_range=[0, 100], color=BLUE_B, stroke_width=4)
        cer_curve = axes.plot(cer, x_range=[0, 100], color=ORANGE, stroke_width=4)

        ce_tag = Text("cross-entropy (train)", font_size=15, color=BLUE_B)
        ce_tag.next_to(axes.c2p(100, ce(100)), RIGHT, buff=0.12).shift(UP * 0.20)
        cer_tag = Text("eval CER", font_size=15, color=ORANGE)
        cer_tag.next_to(axes.c2p(100, cer(100)), RIGHT, buff=0.12).shift(DOWN * 0.14)

        self.play(Create(ce_curve), run_time=1.6)
        self.play(FadeIn(ce_tag))
        self.play(Create(cer_curve), run_time=1.6)
        self.play(FadeIn(cer_tag))
        self.wait(0.3)

        # ── Plateau region + annotations ──────────────────────────────────────
        region = axes.get_area(cer_curve, x_range=[35, 100],
                               color=ORANGE, opacity=0.10)
        self.play(FadeIn(region))

        # dashed line at the CER floor
        floor = DashedLine(
            axes.c2p(0, 0.205), axes.c2p(100, 0.205),
            color=ORANGE, stroke_width=1.4, dash_length=0.10,
        )
        floor_lbl = Text("CER ≈ 0.205 — stalled", font_size=14, color=ORANGE)
        floor_lbl.next_to(axes.c2p(28, 0.205), UP, buff=0.12)
        self.play(Create(floor), FadeIn(floor_lbl))
        self.wait(0.2)

        # "still falling" note on the CE curve tail (placed in open space above)
        ce_note = Text("still falling — but on the wrong thing",
                       font_size=14, color=BLUE_B)
        ce_note.move_to(axes.c2p(44, 0.84))
        ce_arrow = Arrow(ce_note.get_bottom(), axes.c2p(80, ce(80)),
                         color=BLUE_B, stroke_width=2.0, buff=0.12,
                         max_tip_length_to_length_ratio=0.12)
        self.play(FadeIn(ce_note), GrowArrow(ce_arrow))
        self.wait(0.2)

        # the gap: a double-arrow between the curves where the gap is wide,
        # label floating in the open space above it
        gx = 52
        gap_arrow = DoubleArrow(axes.c2p(gx, cer(gx)), axes.c2p(gx, ce(gx)),
                                color=GREY_A, stroke_width=2.0, buff=0.02,
                                max_tip_length_to_length_ratio=0.14)
        gap_lbl = Text("the gap SFT can't close", font_size=13, color=GREY_A)
        gap_lbl.next_to(axes.c2p(gx, ce(gx)), UP, buff=0.18)
        self.play(GrowFromCenter(gap_arrow), FadeIn(gap_lbl))
        self.wait(0.4)

        punch = Text("cross-entropy optimizes a proxy — not the metric you evaluate on",
                     font_size=17, color=GREEN_B)
        punch.to_edge(DOWN, buff=0.30)
        self.play(Write(punch))
        self.wait(2.6)
