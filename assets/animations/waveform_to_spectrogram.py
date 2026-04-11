"""
Animation 1: Waveform → STFT → Spectrogram
Shows how a 1D time-domain waveform is sliced into overlapping windows,
each window FFT'd, and stacked into a 2D spectrogram image.
"""

from manim import *
import numpy as np


class WaveformToSpectrogram(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── 1. Title ──────────────────────────────────────────────────────────
        title = Text("From Waveform to Spectrogram", font_size=38, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)

        # ── 2. Draw the raw waveform ──────────────────────────────────────────
        t = np.linspace(0, 1, 500)
        # Simulate a speech-like waveform: mix of harmonics
        y = (
            0.5 * np.sin(2 * PI * 5 * t)
            + 0.3 * np.sin(2 * PI * 12 * t + 0.8)
            + 0.15 * np.sin(2 * PI * 25 * t + 1.2)
            + 0.05 * np.random.default_rng(42).normal(size=len(t))
        )
        y = y / np.max(np.abs(y)) * 0.9

        axes_wave = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.1, 1.1, 0.5],
            x_length=10,
            y_length=2.0,
            axis_config={"color": GREY_B, "stroke_width": 1.5},
            tips=False,
        ).shift(DOWN * 0.5)

        wave_label = Text("Raw Waveform (pressure over time)", font_size=20, color=GREY_A)
        wave_label.next_to(axes_wave, DOWN, buff=0.15)

        wave_graph = axes_wave.plot_line_graph(
            x_values=t,
            y_values=y,
            line_color=BLUE_B,
            stroke_width=2,
            add_vertex_dots=False,
        )

        self.play(Create(axes_wave), Write(wave_label), run_time=0.8)
        self.play(Create(wave_graph), run_time=1.5)
        self.wait(0.5)

        # ── 3. Highlight overlapping windows ─────────────────────────────────
        window_width = 0.25   # 25% of signal = ~25ms window
        hop = 0.12
        colors = [YELLOW, GREEN, RED, PURPLE, ORANGE]
        window_rects = []
        n_windows = 5

        window_text = Text(
            "Slide overlapping windows (25 ms, 10 ms hop)",
            font_size=20,
            color=YELLOW,
        )
        window_text.next_to(axes_wave, DOWN, buff=0.15)
        self.play(FadeOut(wave_label), FadeIn(window_text))

        for i in range(n_windows):
            x_start = i * hop
            x_end = x_start + window_width
            rect = Rectangle(
                width=axes_wave.x_axis.unit_size * window_width,
                height=axes_wave.y_axis.unit_size * 2.2,
                color=colors[i % len(colors)],
                fill_color=colors[i % len(colors)],
                fill_opacity=0.15,
                stroke_width=2,
            )
            # position relative to axes
            rect.move_to(
                axes_wave.c2p((x_start + x_end) / 2, 0)
            )
            window_rects.append(rect)
            self.play(FadeIn(rect), run_time=0.25)

        self.wait(0.8)

        # ── 4. Morph: slide everything up, show spectrogram below ────────────
        self.play(
            FadeOut(window_text),
            *[FadeOut(r) for r in window_rects],
            wave_graph.animate.shift(UP * 1.8).scale(0.5),
            axes_wave.animate.shift(UP * 1.8).scale(0.5),
            run_time=1,
        )

        # ── 5. Show the FFT of one window ─────────────────────────────────────
        fft_label = Text("FFT each window → frequency spectrum", font_size=22, color=GREEN)
        fft_label.move_to(ORIGIN + UP * 0.8)
        self.play(Write(fft_label), run_time=0.6)

        fft_axes = Axes(
            x_range=[0, 50, 10],
            y_range=[0, 1.1, 0.5],
            x_length=5,
            y_length=2.0,
            axis_config={"color": GREY_B, "stroke_width": 1.5},
            x_axis_config={"label_direction": DOWN},
            tips=False,
        ).shift(DOWN * 0.5 + LEFT * 2.5)

        freq_x = np.linspace(0, 50, 300)
        # Gaussian peaks at the signal harmonics
        freq_y = (
            0.9 * np.exp(-((freq_x - 5) ** 2) / 2)
            + 0.6 * np.exp(-((freq_x - 12) ** 2) / 3)
            + 0.35 * np.exp(-((freq_x - 25) ** 2) / 5)
        )
        freq_y = np.clip(freq_y, 0, 1)

        fft_graph = fft_axes.plot_line_graph(
            x_values=freq_x,
            y_values=freq_y,
            line_color=GREEN_B,
            stroke_width=2.5,
            add_vertex_dots=False,
        )
        x_lbl = Text("Frequency (Hz)", font_size=16, color=GREY_A).next_to(fft_axes, DOWN, buff=0.1)
        y_lbl = Text("Magnitude", font_size=16, color=GREY_A).next_to(fft_axes, LEFT, buff=0.1).rotate(PI / 2)

        self.play(Create(fft_axes), Write(x_lbl), run_time=0.6)
        self.play(Create(fft_graph), run_time=0.8)
        self.wait(0.3)

        # ── 6. Arrow pointing to spectrogram ──────────────────────────────────
        arrow = Arrow(
            start=fft_axes.get_right() + RIGHT * 0.1,
            end=fft_axes.get_right() + RIGHT * 1.5,
            color=WHITE,
            stroke_width=3,
        )
        stack_label = Text("Stack columns\n→ Spectrogram", font_size=18, color=WHITE)
        stack_label.next_to(arrow, RIGHT, buff=0.1)
        self.play(GrowArrow(arrow), Write(stack_label), run_time=0.8)

        # ── 7. Build spectrogram column by column ─────────────────────────────
        spec_width = 3.0
        spec_height = 2.0
        spec_origin = fft_axes.get_right() + RIGHT * 2.1 + DOWN * 0.0

        # Generate fake spectrogram data
        rng = np.random.default_rng(7)
        n_cols = 30
        n_rows = 40
        spec_data = np.zeros((n_rows, n_cols))
        for col in range(n_cols):
            phase = col / n_cols * 2 * PI
            for i, (freq_center, strength) in enumerate([(0.1, 0.9), (0.3, 0.5), (0.6, 0.25)]):
                sigma = 0.08
                rows = np.linspace(0, 1, n_rows)
                spec_data[:, col] += strength * (1 + 0.4 * np.sin(phase + i)) * np.exp(
                    -((rows - freq_center) ** 2) / (2 * sigma ** 2)
                )
        spec_data = spec_data / spec_data.max()

        # Build pixel rects column by column for animation
        col_width = spec_width / n_cols
        row_height = spec_height / n_rows

        # Draw all at once with imshow-style grid
        spec_group = VGroup()
        for col in range(n_cols):
            for row in range(n_rows):
                val = spec_data[n_rows - 1 - row, col]  # flip y
                color = interpolate_color(BLACK, YELLOW, val ** 0.5)
                rect = Rectangle(
                    width=col_width,
                    height=row_height,
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0,
                )
                rect.move_to(
                    spec_origin
                    + RIGHT * (col * col_width - spec_width / 2 + col_width / 2)
                    + UP * (row * row_height - spec_height / 2 + row_height / 2)
                )
                spec_group.add(rect)

        spec_border = Rectangle(
            width=spec_width,
            height=spec_height,
            color=WHITE,
            stroke_width=1.5,
            fill_opacity=0,
        ).move_to(spec_origin)

        spec_title = Text("Spectrogram\n(time × frequency)", font_size=16, color=GREY_A)
        spec_title.next_to(spec_border, DOWN, buff=0.15)

        x_axis_lbl = Text("Time →", font_size=14, color=GREY_B)
        x_axis_lbl.next_to(spec_border, DOWN + RIGHT, buff=0.05)
        y_axis_lbl = Text("Freq ↑", font_size=14, color=GREY_B)
        y_axis_lbl.next_to(spec_border, LEFT, buff=0.05)

        self.play(
            FadeOut(fft_graph),
            FadeOut(fft_axes),
            FadeOut(fft_label),
            FadeOut(x_lbl),
            FadeOut(arrow),
            FadeOut(stack_label),
        )
        self.play(
            LaggedStart(*[FadeIn(r, shift=LEFT * 0.05) for r in spec_group], lag_ratio=0.01),
            run_time=2,
        )
        self.play(Create(spec_border), Write(spec_title), Write(x_axis_lbl), Write(y_axis_lbl))
        self.wait(1.5)

        # ── 8. Final label ────────────────────────────────────────────────────
        conclusion = Text(
            "Audio → 2D image. Every image tool now applies.",
            font_size=22,
            color=BLUE_A,
        )
        conclusion.to_edge(DOWN, buff=0.3)
        self.play(Write(conclusion), run_time=1)
        self.wait(2)
