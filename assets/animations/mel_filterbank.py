"""
Animation: Mel Filterbank
Shows triangular filters on linear frequency axis, spaced uniformly on mel scale,
then compresses to show why log spacing matches human perception.
"""

from manim import *
import numpy as np


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


class MelFilterbank(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("The Mel Filterbank", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ── Step 1: Linear frequency axis ─────────────────────────────────────
        subtitle = Text("Problem: humans hear pitch logarithmically, not linearly", font_size=20, color=GREY_A)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle))

        axes = Axes(
            x_range=[0, 8000, 1000],
            y_range=[0, 1.2, 0.5],
            x_length=10,
            y_length=3.0,
            axis_config={"color": GREY_C, "stroke_width": 1.5},
            tips=False,
        ).shift(DOWN * 0.8)

        x_label = Text("Frequency (Hz)", font_size=16, color=GREY_A).next_to(axes, DOWN, buff=0.1)
        y_label = Text("Magnitude", font_size=16, color=GREY_A).next_to(axes, LEFT, buff=0.1)

        self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.8)

        # Draw 10 evenly spaced triangular filters on linear scale
        n_filters = 10
        f_min, f_max = 0, 8000
        centers_linear = np.linspace(f_min, f_max, n_filters + 2)
        colors = color_gradient([BLUE_B, GREEN_B, YELLOW_B, RED_B], n_filters)

        linear_filters = VGroup()
        for i in range(n_filters):
            left  = centers_linear[i]
            center = centers_linear[i + 1]
            right  = centers_linear[i + 2]
            points = [
                axes.c2p(left, 0),
                axes.c2p(center, 1.0),
                axes.c2p(right, 0),
            ]
            tri = Polygon(*points, color=colors[i], fill_color=colors[i], fill_opacity=0.35, stroke_width=1.5)
            linear_filters.add(tri)

        lin_note = Text("Linearly spaced filters: too many in high freqs, too few in low", font_size=16, color=YELLOW)
        lin_note.to_edge(DOWN, buff=0.3)

        self.play(
            LaggedStart(*[FadeIn(f) for f in linear_filters], lag_ratio=0.07),
            run_time=1.2,
        )
        self.play(Write(lin_note))
        self.wait(0.8)

        # ── Step 2: Morph to mel-spaced filters ──────────────────────────────
        self.play(FadeOut(lin_note), FadeOut(subtitle))
        mel_subtitle = Text("Solution: space filters uniformly on the mel scale", font_size=20, color=GREEN)
        mel_subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(mel_subtitle))

        mel_min = hz_to_mel(80)
        mel_max = hz_to_mel(8000)
        centers_mel_scale = np.linspace(mel_min, mel_max, n_filters + 2)
        centers_hz = mel_to_hz(centers_mel_scale)

        mel_filters = VGroup()
        for i in range(n_filters):
            left   = centers_hz[i]
            center = centers_hz[i + 1]
            right  = centers_hz[i + 2]
            points = [
                axes.c2p(left, 0),
                axes.c2p(center, 1.0),
                axes.c2p(right, 0),
            ]
            tri = Polygon(*points, color=colors[i], fill_color=colors[i], fill_opacity=0.35, stroke_width=1.5)
            mel_filters.add(tri)

        self.play(
            Transform(linear_filters, mel_filters),
            run_time=1.5,
        )

        mel_note = Text("Mel-spaced: dense at low freqs (speech fundamentals), sparse at high", font_size=16, color=GREEN)
        mel_note.to_edge(DOWN, buff=0.3)
        self.play(Write(mel_note))
        self.wait(0.8)

        # ── Step 3: Show the mel formula ─────────────────────────────────────
        self.play(FadeOut(mel_note), FadeOut(mel_subtitle))

        formula_line1 = Text("mel(f) = 2595 x log10(1 + f / 700)", font_size=22, color=BLUE_A)
        formula_line2 = Text("100 Hz -> 150 mel    |    1000 Hz -> 999 mel    |    8000 Hz -> 2840 mel", font_size=17, color=GREY_A)
        formula_line1.next_to(title, DOWN, buff=0.2)
        formula_line2.next_to(formula_line1, DOWN, buff=0.2)

        self.play(Write(formula_line1), run_time=0.8)
        self.play(Write(formula_line2), run_time=0.8)

        # highlight low-freq dense region
        low_freq_rect = Rectangle(
            width=axes.x_axis.unit_size * 1000,
            height=axes.y_axis.unit_size * 1.2,
            color=YELLOW,
            stroke_width=2,
            fill_opacity=0.08,
        ).move_to(axes.c2p(500, 0.6))

        dense_label = Text("Dense here\n(vowels, F0)", font_size=14, color=YELLOW)
        dense_label.next_to(low_freq_rect, UP, buff=0.1)

        self.play(Create(low_freq_rect), Write(dense_label))
        self.wait(0.5)

        # ── Step 4: Apply filterbank = matrix multiply ────────────────────────
        self.play(
            FadeOut(low_freq_rect), FadeOut(dense_label),
            FadeOut(formula_line1), FadeOut(formula_line2),
            FadeOut(linear_filters),
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
        )

        # Show the matrix multiply visually
        fft_box = Rectangle(width=1.5, height=3.5, color=BLUE_B, fill_opacity=0.2)
        fft_label = Text("FFT\nmagnitude\n[513 freqs]", font_size=16, color=BLUE_B)
        fft_label.move_to(fft_box.get_center())
        fft_group = VGroup(fft_box, fft_label).shift(LEFT * 4)

        times_label = Text("@", font_size=36, color=WHITE)

        mel_matrix_box = Rectangle(width=3.0, height=1.5, color=GREEN_B, fill_opacity=0.2)
        mel_matrix_label = Text("Mel filterbank\nmatrix\n[80 x 513]", font_size=16, color=GREEN_B)
        mel_matrix_label.move_to(mel_matrix_box.get_center())
        mel_group = VGroup(mel_matrix_box, mel_matrix_label)

        eq_label = Text("=", font_size=36, color=WHITE)

        mel_out_box = Rectangle(width=1.5, height=1.5, color=YELLOW_B, fill_opacity=0.2)
        mel_out_label = Text("Mel spec\n[80 mels]", font_size=16, color=YELLOW_B)
        mel_out_label.move_to(mel_out_box.get_center())
        mel_out_group = VGroup(mel_out_box, mel_out_label).shift(RIGHT * 4.5)

        mat_group = VGroup(fft_group, times_label, mel_group, eq_label, mel_out_group).arrange(RIGHT, buff=0.4)

        self.play(FadeIn(mat_group), run_time=1)

        log_step = Text("Then: log(mel_spec + 1e-8)  ->  compress dynamic range", font_size=18, color=ORANGE)
        log_step.to_edge(DOWN, buff=0.5)
        self.play(Write(log_step))
        self.wait(2)
