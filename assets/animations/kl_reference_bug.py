"""
Animation: The KL reference bug.

The KL term anchors the policy to a *reference*. The obvious way to get one —
disable the LoRA adapter and call the base model — is subtly wrong: it anchors to
the wrong policy (the un-adapted base, fighting SFT) AND runs in a different dtype
(bf16 vs the fp32 policy path), so KL is nonzero at step 0 and blows up.

The fix: snapshot the SFT adapter itself in the identical fp32 path. Reference ==
policy at init, so KL starts at exactly 0.
"""

from manim import *
import numpy as np


class KLReferenceBug(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("The KL reference bug: what are you anchoring to?",
                     font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.30)
        self.play(Write(title))

        divider = DashedLine([0, 2.2, 0], [0, -3.2, 0], color=GREY_D, stroke_width=1.2)
        self.play(Create(divider))

        # ── LEFT: wrong ───────────────────────────────────────────────────────
        wrong_hdr = Text("Obvious approach", font_size=20, color=RED_C)
        wrong_hdr.move_to([-3.5, 2.4, 0])
        wrong_sub = Text("disable LoRA → reference = base model", font_size=14, color=GREY_A)
        wrong_sub.next_to(wrong_hdr, DOWN, buff=0.12)
        self.play(FadeIn(wrong_hdr), FadeIn(wrong_sub))

        # two mismatched boxes
        pol_w = RoundedRectangle(corner_radius=0.06, width=2.5, height=0.62,
                                 fill_color=GREEN_E, fill_opacity=0.4,
                                 stroke_color=GREEN_B, stroke_width=1.5)
        pol_w.move_to([-3.5, 1.35, 0])
        pol_w_l = Text("policy: base + SFT LoRA  (fp32)", font_size=12, color=WHITE)
        pol_w_l.move_to(pol_w.get_center())
        ref_w = RoundedRectangle(corner_radius=0.06, width=2.5, height=0.62,
                                 fill_color=MAROON_E, fill_opacity=0.4,
                                 stroke_color=RED_B, stroke_width=1.5)
        ref_w.move_to([-3.5, 0.5, 0])
        ref_w_l = Text("reference: base only  (bf16)", font_size=12, color=WHITE)
        ref_w_l.move_to(ref_w.get_center())
        self.play(FadeIn(pol_w), FadeIn(pol_w_l), FadeIn(ref_w), FadeIn(ref_w_l))

        mismatch = VGroup(
            Text("✗ anchors to the wrong policy (fights SFT)", font_size=12, color=RED_C),
            Text("✗ bf16 ≠ fp32 → ~3 nats drift × 28 layers", font_size=12, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        mismatch.move_to([-3.5, -0.35, 0])
        self.play(FadeIn(mismatch))

        # KL blow-up mini plot
        axes_w = Axes(x_range=[0, 100, 50], y_range=[0, 1.0, 0.5],
                      x_length=2.9, y_length=1.5,
                      axis_config={"color": GREY_C, "stroke_width": 1.5},
                      tips=False).move_to([-3.5, -2.15, 0])
        klw = axes_w.plot(lambda x: min(0.98, 0.06 + 0.0009 * x * x / 8),
                          x_range=[0, 100], color=RED_B, stroke_width=3)
        klw_lbl = Text("KL explodes", font_size=13, color=RED_B).next_to(axes_w, UP, buff=0.06)
        self.play(Create(axes_w), FadeIn(klw_lbl))
        self.play(Create(klw), run_time=1.4)
        self.wait(0.4)

        # ── RIGHT: fix ────────────────────────────────────────────────────────
        fix_hdr = Text("The fix", font_size=20, color=GREEN_C).move_to([3.5, 2.4, 0])
        fix_sub = Text("frozen fp32 snapshot of the SFT adapter", font_size=14, color=GREY_A)
        fix_sub.next_to(fix_hdr, DOWN, buff=0.12)
        self.play(FadeIn(fix_hdr), FadeIn(fix_sub))

        pol_f = RoundedRectangle(corner_radius=0.06, width=2.5, height=0.62,
                                 fill_color=GREEN_E, fill_opacity=0.4,
                                 stroke_color=GREEN_B, stroke_width=1.5)
        pol_f.move_to([3.5, 1.35, 0])
        pol_f_l = Text("policy: base + SFT LoRA  (fp32)", font_size=12, color=WHITE)
        pol_f_l.move_to(pol_f.get_center())
        ref_f = RoundedRectangle(corner_radius=0.06, width=2.5, height=0.62,
                                 fill_color=GREEN_E, fill_opacity=0.4,
                                 stroke_color=GREEN_B, stroke_width=1.5)
        ref_f.move_to([3.5, 0.5, 0])
        ref_f_l = Text("reference: SFT snapshot  (fp32)", font_size=12, color=WHITE)
        ref_f_l.move_to(ref_f.get_center())
        self.play(FadeIn(pol_f), FadeIn(pol_f_l), FadeIn(ref_f), FadeIn(ref_f_l))

        eq = Text("reference == policy at init  →  identical path", font_size=12, color=GREEN_B)
        eq.move_to([3.5, -0.35, 0])
        self.play(FadeIn(eq))

        axes_f = Axes(x_range=[0, 100, 50], y_range=[0, 1.0, 0.5],
                      x_length=2.9, y_length=1.5,
                      axis_config={"color": GREY_C, "stroke_width": 1.5},
                      tips=False).move_to([3.5, -2.15, 0])
        klf = axes_f.plot(lambda x: 0.02 + 0.10 * (1 - np.exp(-x / 30)),
                          x_range=[0, 100], color=GREEN_B, stroke_width=3)
        klf_lbl = Text("KL = 0 at t₀, stays bounded", font_size=13, color=GREEN_B)
        klf_lbl.next_to(axes_f, UP, buff=0.06)
        self.play(Create(axes_f), FadeIn(klf_lbl))
        self.play(Create(klf), run_time=1.4)
        self.wait(2.6)
