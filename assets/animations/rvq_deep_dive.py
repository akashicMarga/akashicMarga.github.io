"""
Animation: RVQ Deep Dive
The most detailed animation — walks through every step of how a single
encoder vector gets quantized through multiple codebook levels:
  1. Encoder outputs vector z
  2. Codebook initialized randomly
  3. Distance calculation to every entry
  4. Argmin -> nearest neighbor -> integer index k
  5. Quantization error = residual
  6. Next level takes residual as input
  7. Straight-through estimator for gradient
  8. EMA update of codebook entries
"""

from manim import *
import numpy as np


class RVQDeepDive(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ─── Title ────────────────────────────────────────────────────────────
        title = Text("RVQ: How It Really Works", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ─── PART 1: The Encoder outputs a vector ─────────────────────────────
        step1 = Text("Step 1: Encoder outputs a continuous vector z", font_size=22, color=BLUE_A)
        step1.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(step1))

        # Draw encoder box
        enc_box = Rectangle(width=1.6, height=1.0, color=BLUE_B, fill_opacity=0.2)
        enc_label = Text("Encoder\n(CNN)", font_size=16, color=BLUE_B)
        enc_label.move_to(enc_box.get_center())
        enc_group = VGroup(enc_box, enc_label).shift(LEFT * 5 + DOWN * 0.5)

        # Audio waveform input
        audio_label = Text("audio", font_size=14, color=GREY_A)
        audio_arrow = Arrow(LEFT * 6.5 + DOWN * 0.5, enc_group.get_left(), color=GREY_A, stroke_width=2, buff=0.05)
        audio_label.next_to(audio_arrow, UP, buff=0.05)

        self.play(FadeIn(enc_group), GrowArrow(audio_arrow), Write(audio_label))

        # Vector z as a colored bar
        z_vals = np.array([0.73, -0.41, 0.88, 0.15, -0.62])
        z_colors = [RED if v < 0 else BLUE_B for v in z_vals]

        z_bars = VGroup()
        for i, (v, c) in enumerate(zip(z_vals, z_colors)):
            bar = Rectangle(
                width=0.35,
                height=abs(v) * 0.8,
                color=c,
                fill_color=c,
                fill_opacity=0.8,
                stroke_width=1,
            )
            bar.shift(DOWN * (0.4 - abs(v) * 0.4) if v >= 0 else UP * (0.4 - abs(v) * 0.4))
            z_bars.add(bar)
        z_bars.arrange(RIGHT, buff=0.06)
        z_bars.shift(LEFT * 2.5 + DOWN * 0.5)

        z_label = Text("z  (d-dim vector)", font_size=15, color=BLUE_A)
        z_label.next_to(z_bars, UP, buff=0.1)

        z_arrow = Arrow(enc_group.get_right(), z_bars.get_left() + LEFT * 0.1, color=BLUE_A, stroke_width=2, buff=0.05)
        self.play(GrowArrow(z_arrow))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in z_bars], lag_ratio=0.1), Write(z_label))
        self.wait(0.6)

        # ─── PART 2: The Codebook ─────────────────────────────────────────────
        self.play(FadeOut(step1))
        step2 = Text("Step 2: Codebook — N randomly initialized vectors", font_size=22, color=GREEN_A)
        step2.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(step2))

        # Draw codebook as a matrix of rows
        n_shown = 6  # show 6 entries
        cb_entries = VGroup()
        cb_labels = VGroup()
        cb_colors_list = [GREEN_B, TEAL_B, YELLOW_B, ORANGE, RED_B, PURPLE_B]
        rng = np.random.default_rng(99)

        for i in range(n_shown):
            row_vals = rng.normal(size=5) * 0.5
            row_bars = VGroup()
            for v in row_vals:
                b = Rectangle(width=0.28, height=max(abs(v) * 0.6, 0.05),
                               color=cb_colors_list[i], fill_color=cb_colors_list[i],
                               fill_opacity=0.7, stroke_width=0.8)
                row_bars.add(b)
            row_bars.arrange(RIGHT, buff=0.04)
            lbl = Text(f"e{i}", font_size=13, color=cb_colors_list[i])
            cb_entries.add(row_bars)
            cb_labels.add(lbl)

        cb_entries.arrange(DOWN, buff=0.15)
        cb_entries.shift(RIGHT * 1.5 + DOWN * 0.5)
        for i, (entry, lbl) in enumerate(zip(cb_entries, cb_labels)):
            lbl.next_to(entry, LEFT, buff=0.1)

        cb_box = SurroundingRectangle(VGroup(cb_entries, cb_labels), color=GREEN_B, buff=0.15, corner_radius=0.1)
        cb_title = Text("Codebook\n(N=1024 entries)", font_size=15, color=GREEN_A)
        cb_title.next_to(cb_box, UP, buff=0.1)

        self.play(
            Create(cb_box),
            Write(cb_title),
            LaggedStart(*[FadeIn(VGroup(e, l)) for e, l in zip(cb_entries, cb_labels)], lag_ratio=0.1),
            run_time=1.2,
        )
        self.wait(0.5)

        # ─── PART 3: Distance calculation ─────────────────────────────────────
        self.play(FadeOut(step2))
        step3 = Text("Step 3: Compute distance from z to every codebook entry", font_size=22, color=YELLOW)
        step3.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(step3))

        dist_arrows = VGroup()
        dist_labels = VGroup()
        fake_dists = [2.3, 1.1, 3.7, 0.4, 2.9, 1.8]  # fake L2 distances

        for i, (entry, dist) in enumerate(zip(cb_entries, fake_dists)):
            arr = DashedLine(
                z_bars.get_right() + RIGHT * 0.1,
                entry.get_left() + LEFT * 0.1,
                color=GREY_C,
                stroke_width=1,
                dash_length=0.1,
            )
            dist_lbl = Text(f"d={dist:.1f}", font_size=11, color=GREY_B)
            dist_lbl.next_to(arr, UP if i % 2 == 0 else DOWN, buff=0.03)
            dist_arrows.add(arr)
            dist_labels.add(dist_lbl)

        self.play(
            LaggedStart(*[Create(a) for a in dist_arrows], lag_ratio=0.1),
            LaggedStart(*[Write(l) for l in dist_labels], lag_ratio=0.1),
            run_time=1.2,
        )
        self.wait(0.5)

        # ─── PART 4: Argmin → nearest neighbor ────────────────────────────────
        self.play(FadeOut(step3))
        step4 = Text("Step 4: argmin -> nearest entry becomes the code k", font_size=22, color=ORANGE)
        step4.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(step4))

        # Highlight the nearest entry (index 3, dist=0.4)
        nearest_idx = 3
        nearest_entry = cb_entries[nearest_idx]
        nearest_rect = SurroundingRectangle(nearest_entry, color=ORANGE, stroke_width=3, buff=0.05)
        nearest_label = Text("k = 3  (nearest!  d=0.4)", font_size=16, color=ORANGE)
        nearest_label.next_to(nearest_rect, RIGHT, buff=0.15)

        self.play(Create(nearest_rect), Write(nearest_label))
        self.play(
            *[FadeOut(a) for a in dist_arrows],
            *[FadeOut(l) for l in dist_labels],
        )

        # Arrow from codebook entry back to z_q
        zq_bars = z_bars.copy().set_color(ORANGE).shift(RIGHT * 0.2 + DOWN * 1.4)
        zq_label = Text("z_q = codebook[k=3]", font_size=15, color=ORANGE)
        zq_label.next_to(zq_bars, UP, buff=0.1)
        zq_arrow = Arrow(nearest_entry.get_bottom(), zq_bars.get_top(), color=ORANGE, stroke_width=2, buff=0.1)

        self.play(GrowArrow(zq_arrow))
        self.play(FadeIn(zq_bars), Write(zq_label))
        self.wait(0.6)

        # ─── PART 5: Residual ─────────────────────────────────────────────────
        self.play(FadeOut(step4))
        step5 = Text("Step 5: residual = z - z_q  (the error left over)", font_size=22, color=RED_B)
        step5.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(step5))

        # Show residual bars (smaller)
        res_vals = z_vals - np.array([0.73 - 0.4, -0.41 + 0.2, 0.88 - 0.5, 0.15 - 0.1, -0.62 + 0.3])
        res_bars = VGroup()
        for v in res_vals:
            c = RED_B if v < 0 else PINK
            b = Rectangle(width=0.35, height=max(abs(v) * 0.8, 0.03), color=c, fill_color=c,
                           fill_opacity=0.8, stroke_width=1)
            res_bars.add(b)
        res_bars.arrange(RIGHT, buff=0.06)
        res_bars.next_to(zq_bars, DOWN, buff=0.5)

        minus_lbl = Text("z - z_q =", font_size=15, color=RED_B)
        minus_lbl.next_to(res_bars, LEFT, buff=0.1)
        res_lbl = Text("r1  (residual)", font_size=15, color=RED_B)
        res_lbl.next_to(res_bars, RIGHT, buff=0.1)

        res_arrow = Arrow(zq_bars.get_bottom(), res_bars.get_top(), color=RED_B, stroke_width=2, buff=0.1)
        self.play(GrowArrow(res_arrow))
        self.play(FadeIn(res_bars), Write(minus_lbl), Write(res_lbl))
        self.wait(0.4)

        lvl2_label = Text("Level 2 takes r1 as input -> repeats same process", font_size=17, color=PINK)
        lvl2_label.to_edge(DOWN, buff=0.6)
        self.play(Write(lvl2_label))
        self.wait(1.0)

        # ─── PART 6: Straight-Through Estimator ──────────────────────────────
        self.play(
            FadeOut(step5), FadeOut(lvl2_label),
            FadeOut(res_bars), FadeOut(res_arrow), FadeOut(minus_lbl), FadeOut(res_lbl),
            FadeOut(zq_bars), FadeOut(zq_label), FadeOut(zq_arrow),
            FadeOut(nearest_rect), FadeOut(nearest_label),
            FadeOut(dist_arrows), FadeOut(dist_labels),
            FadeOut(cb_box), FadeOut(cb_title),
            FadeOut(cb_entries), FadeOut(cb_labels),
            FadeOut(z_bars), FadeOut(z_label),
            FadeOut(z_arrow), FadeOut(enc_group),
            FadeOut(audio_arrow), FadeOut(audio_label),
        )

        step6 = Text("Step 6: Gradient problem — argmin has zero gradient", font_size=22, color=RED)
        step6.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(step6))

        # Forward pass diagram
        boxes = ["Encoder", "argmin\n(no grad!)", "Decoder", "Loss"]
        box_colors = [BLUE_B, RED, GREEN_B, YELLOW_B]
        box_groups = VGroup()
        for label, color in zip(boxes, box_colors):
            box = Rectangle(width=1.6, height=0.8, color=color, fill_opacity=0.15)
            txt = Text(label, font_size=14, color=color)
            txt.move_to(box.get_center())
            box_groups.add(VGroup(box, txt))
        box_groups.arrange(RIGHT, buff=0.5).shift(DOWN * 0.3)

        fwd_arrows = VGroup()
        for i in range(len(box_groups) - 1):
            arr = Arrow(box_groups[i].get_right(), box_groups[i+1].get_left(), color=WHITE, stroke_width=2, buff=0.05)
            fwd_arrows.add(arr)

        fwd_lbl = Text("Forward pass (correct)", font_size=16, color=GREY_A)
        fwd_lbl.next_to(box_groups, UP, buff=0.2)

        self.play(FadeIn(box_groups), FadeIn(fwd_arrows), Write(fwd_lbl))

        # Backward pass: gradient stops at argmin
        bwd_arrows = VGroup()
        for i in range(len(box_groups) - 1, 0, -1):
            color = RED if i == 1 else GREEN_C
            arr = Arrow(
                box_groups[i].get_left() + DOWN * 0.3,
                box_groups[i-1].get_right() + DOWN * 0.3,
                color=color,
                stroke_width=2,
                buff=0.05,
            )
            if i == 1:
                cross = Cross(arr, color=RED, stroke_width=3)
                bwd_arrows.add(arr, cross)
            else:
                bwd_arrows.add(arr)

        bwd_lbl = Text("Backward pass: gradient BLOCKED at argmin", font_size=16, color=RED)
        bwd_lbl.next_to(box_groups, DOWN, buff=0.5)

        self.play(FadeIn(bwd_arrows), Write(bwd_lbl))
        self.wait(0.8)

        # STE fix
        ste_lbl = Text(
            "Fix: Straight-Through Estimator\nz_q_st = z + (z_q - z).detach()\n-> gradient pretends quantization never happened",
            font_size=16,
            color=GREEN_A,
            line_spacing=1.3,
        )
        ste_lbl.to_edge(DOWN, buff=0.2)
        self.play(Write(ste_lbl))
        self.wait(2.5)
