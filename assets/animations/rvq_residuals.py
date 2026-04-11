"""
Animation 3: Residual Vector Quantization (RVQ)
Shows how stacking quantizers on residuals progressively refines the
approximation — like progressive JPEG passes for audio.
"""

from manim import *
import numpy as np


class RVQResiduals(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ──────────────────────────────────────────────────────────────
        title = Text("Residual Vector Quantization (RVQ)", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        subtitle = Text(
            "Each level encodes the error left by the previous level",
            font_size=20, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle))
        self.wait(0.5)

        # ── Setup: 1D latent bar showing the true value ────────────────────────
        # We'll show a 1D magnitude bar and a series of progressively better approx
        n_levels = 4
        true_value = 0.73   # the true encoder output (normalized 0-1)

        bar_width = 8.0
        bar_height = 0.45
        bar_origin = ORIGIN + UP * 0.5

        def make_bar(value, color, label_str, row_offset):
            """Returns a filled rectangle representing a quantized approximation."""
            track = Rectangle(
                width=bar_width,
                height=bar_height,
                color=GREY_D,
                fill_color=GREY_E,
                fill_opacity=0.3,
                stroke_width=1,
            ).move_to(bar_origin + DOWN * row_offset)

            fill = Rectangle(
                width=bar_width * value,
                height=bar_height,
                fill_color=color,
                fill_opacity=0.85,
                stroke_width=0,
            )
            fill.align_to(track, LEFT)
            fill.move_to(
                track.get_left() + RIGHT * (bar_width * value / 2),
            )
            fill.set_y(track.get_y())

            lbl = Text(label_str, font_size=18, color=GREY_A)
            lbl.next_to(track, LEFT, buff=0.2)

            val_lbl = Text(f"{value:.3f}", font_size=16, color=color)
            val_lbl.next_to(track, RIGHT, buff=0.2)

            return VGroup(track, fill, lbl, val_lbl)

        # True value bar
        true_bar = make_bar(true_value, WHITE, "z  (encoder output)", 0)
        self.play(FadeIn(true_bar), run_time=0.8)

        true_line = DashedLine(
            start=true_bar[0].get_left() + RIGHT * bar_width * true_value + UP * 0.6,
            end=true_bar[0].get_left() + RIGHT * bar_width * true_value + DOWN * 3.5,
            color=WHITE,
            stroke_width=1.5,
            dash_length=0.12,
        )
        self.play(Create(true_line), run_time=0.5)
        self.wait(0.3)

        # ── Level-by-level RVQ ────────────────────────────────────────────────
        # Simulate RVQ: each level has a codebook entry that is the nearest
        # quantized value at that level's granularity
        quant_steps = [0.5, 0.25, 0.125, 0.0625]  # each level halves the step
        level_colors = [RED_B, ORANGE, YELLOW_B, GREEN_B]

        approx = 0.0
        residual = true_value
        level_bars = []
        residual_bars = []

        step_label = Text("", font_size=20)

        for lvl in range(n_levels):
            row = lvl + 1
            step = quant_steps[lvl]
            # Quantize residual to nearest step
            k = round(residual / step)
            q_val = k * step
            q_val = np.clip(q_val, 0, 1)

            new_approx = approx + q_val
            new_residual = true_value - new_approx

            # Show "residual at this level"
            res_bar = make_bar(
                residual,
                GREY_C,
                f"residual r{lvl}",
                row * 1.0,
            )
            self.play(FadeIn(res_bar), run_time=0.4)

            # Show quantized value
            q_bar = make_bar(
                new_approx,
                level_colors[lvl],
                f"approx (L1–{lvl+1})",
                row * 1.0 + 0.5,
            )

            # Annotation
            ann = Text(
                f"Level {lvl+1}: quantize r{lvl} → q{lvl}={q_val:.3f}  |  approx={new_approx:.3f}  |  residual={new_residual:.3f}",
                font_size=14,
                color=level_colors[lvl],
            )
            ann.to_edge(DOWN, buff=0.25)

            self.play(
                FadeIn(q_bar),
                FadeIn(ann),
                run_time=0.6,
            )
            self.wait(0.7)
            self.play(FadeOut(ann), run_time=0.2)

            level_bars.append(q_bar)
            residual_bars.append(res_bar)

            approx = new_approx
            residual = new_residual

        # ── Final comparison ───────────────────────────────────────────────────
        final_text = Text(
            f"After {n_levels} levels: approx ≈ {approx:.3f}  |  true = {true_value:.3f}  |  error = {abs(true_value - approx):.4f}",
            font_size=18,
            color=GREEN,
        )
        final_text.to_edge(DOWN, buff=0.25)
        self.play(Write(final_text), run_time=0.8)
        self.wait(1.2)

        # ── Collapse to token stream ───────────────────────────────────────────
        self.play(
            FadeOut(final_text),
            *[FadeOut(b) for b in residual_bars],
            *[FadeOut(b) for b in level_bars],
            FadeOut(true_bar),
            FadeOut(true_line),
            FadeOut(subtitle),
            run_time=0.8,
        )

        # Show: one frame → Q integers
        token_title = Text("One audio frame → Q integer indices", font_size=26, color=WHITE)
        token_title.move_to(ORIGIN + UP * 1.5)
        self.play(Write(token_title))

        tokens = VGroup()
        for i in range(n_levels):
            box = Square(side_length=0.9, color=level_colors[i], fill_color=level_colors[i], fill_opacity=0.25)
            label = Text(f"k{i+1}", font_size=26, color=level_colors[i])
            label.move_to(box.get_center())
            grp = VGroup(box, label)
            tokens.add(grp)
        tokens.arrange(RIGHT, buff=0.3)
        tokens.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[GrowFromCenter(t) for t in tokens], lag_ratio=0.15), run_time=1.2)

        brace = Brace(tokens, DOWN, color=GREY_A)
        brace_text = Text("4 integers per frame  (N=1024 each = 40 bits)", font_size=16, color=GREY_A)
        brace_text.next_to(brace, DOWN, buff=0.1)
        self.play(GrowFromCenter(brace), Write(brace_text))
        self.wait(0.5)

        coarse_fine = Text(
            "Coarse (L1): speaker identity + broad phoneme\n"
            "Fine (L4):   acoustic texture + room details",
            font_size=18,
            color=GREY_A,
            line_spacing=1.3,
        )
        coarse_fine.to_edge(DOWN, buff=0.4)
        self.play(Write(coarse_fine))
        self.wait(2.5)
