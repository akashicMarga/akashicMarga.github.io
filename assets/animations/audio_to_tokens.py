"""
Social media animation: Audio → Codec → Token Grid
Narrative:
  1. Show 1 second of raw waveform (24,000 samples)
  2. Zoom into one 13ms frame window
  3. Show the codec compressing that frame → 8 integers (one RVQ level per integer)
  4. Pull back — repeat for all 75 frames → full 8×75 token grid
  5. Side-by-side comparison: raw samples vs token grid
"""

from manim import *
import numpy as np


class AudioToTokens(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"
        rng = np.random.default_rng(42)

        # ── Shared waveform data ───────────────────────────────────────────────
        n_samples = 600   # visual resolution
        t = np.linspace(0, 1, n_samples)
        y = (
            0.55 * np.sin(2 * PI * 4 * t)
            + 0.28 * np.sin(2 * PI * 11 * t + 0.6)
            + 0.14 * np.sin(2 * PI * 23 * t + 1.1)
            + 0.06 * np.sin(2 * PI * 41 * t + 0.3)
            + 0.03 * rng.normal(size=n_samples)
        )
        y = y / np.max(np.abs(y)) * 0.88

        # ══════════════════════════════════════════════════════════════════════
        # SCENE 1 — 1 second of raw audio
        # ══════════════════════════════════════════════════════════════════════
        title1 = Text("1 second of speech audio", font_size=36, color=WHITE)
        title1.to_edge(UP, buff=0.4)
        self.play(Write(title1), run_time=0.7)

        axes1 = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1.2, 1.2, 0.5],
            x_length=11,
            y_length=3.5,
            axis_config={"color": GREY_C, "stroke_width": 1.2},
            tips=False,
        ).shift(DOWN * 0.5)

        x_lbl = Text("time (seconds)", font_size=16, color=GREY_B)
        x_lbl.next_to(axes1, DOWN, buff=0.15)

        wave1 = axes1.plot_line_graph(
            x_values=t, y_values=y,
            line_color=BLUE_B, stroke_width=1.6, add_vertex_dots=False,
        )

        self.play(Create(axes1), Write(x_lbl), run_time=0.6)
        self.play(Create(wave1), run_time=1.4)

        # Sample count label
        sample_lbl = Text("24,000 float32 samples  ·  96 KB raw", font_size=18, color=GREY_A)
        sample_lbl.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(sample_lbl), run_time=0.5)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════════
        # SCENE 2 — zoom into one 13ms frame
        # ══════════════════════════════════════════════════════════════════════
        zoom_label = Text("Zoom in — one codec frame (13 ms)", font_size=28, color=YELLOW)
        zoom_label.to_edge(UP, buff=0.4)

        # Frame window: first 13ms = first ~1/75 of the signal
        frame_start = 0.0
        frame_end   = 1 / 75          # ≈ 0.0133 s

        # Highlight rectangle over the frame
        x_left  = axes1.c2p(frame_start, -1.15)
        x_right = axes1.c2p(frame_end,    1.15)
        frame_rect = Rectangle(
            width=x_right[0] - x_left[0],
            height=axes1.y_axis.unit_size * 2.3,
            color=YELLOW,
            fill_color=YELLOW,
            fill_opacity=0.15,
            stroke_width=2,
        ).move_to(axes1.c2p((frame_start + frame_end) / 2, 0))

        self.play(
            FadeOut(title1), FadeOut(sample_lbl),
            FadeIn(zoom_label),
            Create(frame_rect),
            run_time=0.7,
        )
        self.wait(0.4)

        # Animate zoom: shrink full wave, grow zoomed view
        # Build zoomed axes
        frame_mask = (t >= frame_start) & (t <= frame_end + 0.002)
        t_zoom = t[frame_mask]
        y_zoom = y[frame_mask]

        axes_zoom = Axes(
            x_range=[0, frame_end, frame_end / 4],
            y_range=[-1.2, 1.2, 0.5],
            x_length=6,
            y_length=3.5,
            axis_config={"color": GREY_C, "stroke_width": 1.2},
            tips=False,
        ).shift(LEFT * 2.8 + DOWN * 0.4)

        ms_lbl = Text("0 – 13 ms", font_size=16, color=YELLOW)
        ms_lbl.next_to(axes_zoom, DOWN, buff=0.15)

        wave_zoom = axes_zoom.plot_line_graph(
            x_values=t_zoom, y_values=y_zoom,
            line_color=YELLOW, stroke_width=2.2, add_vertex_dots=False,
        )

        self.play(
            FadeOut(wave1), FadeOut(axes1), FadeOut(x_lbl), FadeOut(frame_rect),
            run_time=0.5,
        )
        self.play(
            Create(axes_zoom), Write(ms_lbl), run_time=0.6,
        )
        self.play(Create(wave_zoom), run_time=0.8)

        frame_sample_lbl = Text("320 samples", font_size=16, color=GREY_A)
        frame_sample_lbl.next_to(axes_zoom, UP, buff=0.15)
        self.play(FadeIn(frame_sample_lbl), run_time=0.4)
        self.wait(0.5)

        # ══════════════════════════════════════════════════════════════════════
        # SCENE 3 — codec compresses this frame → 8 integers
        # ══════════════════════════════════════════════════════════════════════
        codec_title = Text("Codec: 320 samples  →  8 integers", font_size=26, color=WHITE)
        codec_title.to_edge(UP, buff=0.4)
        self.play(FadeOut(zoom_label), Write(codec_title), run_time=0.6)

        # Arrow encode
        arrow_enc = Arrow(
            start=axes_zoom.get_right() + RIGHT * 0.1,
            end=axes_zoom.get_right() + RIGHT * 1.6,
            color=WHITE, stroke_width=2.5, buff=0.05,
            max_tip_length_to_length_ratio=0.2,
        )
        enc_txt = Text("RVQ\nencode", font_size=15, color=GREY_A)
        enc_txt.next_to(arrow_enc, UP, buff=0.08)
        self.play(GrowArrow(arrow_enc), Write(enc_txt), run_time=0.5)

        # 8 token boxes (one per RVQ level)
        level_colors = [BLUE_B, GREEN_B, YELLOW_B, RED_B, TEAL_B, ORANGE, PURPLE_B, PINK]
        level_vals   = [412, 87, 631, 204, 519, 93, 377, 841]
        level_labels = [f"L{i+1}" for i in range(8)]

        token_col = VGroup()
        for i, (val, col, lbl) in enumerate(zip(level_vals, level_colors, level_labels)):
            row = VGroup()
            level_tag = Text(lbl, font_size=13, color=col)
            box = RoundedRectangle(
                width=1.1, height=0.52, corner_radius=0.08,
                color=col, fill_color=col, fill_opacity=0.18, stroke_width=1.8,
            )
            num = Text(str(val), font_size=15, color=col)
            num.move_to(box.get_center())
            level_tag.next_to(box, LEFT, buff=0.12)
            row.add(level_tag, box, num)
            token_col.add(row)

        token_col.arrange(DOWN, buff=0.1)
        token_col.shift(RIGHT * 3.8 + DOWN * 0.2)

        self.play(
            LaggedStart(*[FadeIn(row, shift=LEFT * 0.2) for row in token_col], lag_ratio=0.1),
            run_time=1.2,
        )

        frame_note = Text("1 frame = 8 integers\n(one per codebook level)", font_size=16, color=GREY_A)
        frame_note.next_to(token_col, DOWN, buff=0.25)
        self.play(Write(frame_note), run_time=0.5)
        self.wait(0.8)

        # ══════════════════════════════════════════════════════════════════════
        # SCENE 4 — pull back: 75 frames → full 8×75 grid
        # ══════════════════════════════════════════════════════════════════════
        pullback_title = Text("Do this for all 75 frames in 1 second", font_size=28, color=WHITE)
        pullback_title.to_edge(UP, buff=0.4)

        self.play(
            FadeOut(codec_title),
            FadeOut(axes_zoom), FadeOut(wave_zoom), FadeOut(ms_lbl),
            FadeOut(frame_sample_lbl), FadeOut(arrow_enc), FadeOut(enc_txt),
            FadeOut(token_col), FadeOut(frame_note),
            Write(pullback_title),
            run_time=0.7,
        )

        # Build 8×75 grid
        n_levels = 8
        n_frames = 75
        cell_w = 0.13
        cell_h = 0.42
        grid_w  = n_frames * cell_w
        grid_h  = n_levels * cell_h

        grid_origin = ORIGIN + DOWN * 0.3

        grid = VGroup()
        for lvl in range(n_levels):
            for frm in range(n_frames):
                val = rng.integers(0, 1024)
                intensity = val / 1023
                color = interpolate_color(
                    level_colors[lvl],
                    BLACK,
                    0.7 - 0.5 * intensity,
                )
                cell = Rectangle(
                    width=cell_w - 0.01,
                    height=cell_h - 0.02,
                    fill_color=color,
                    fill_opacity=0.9,
                    stroke_width=0,
                )
                cell.move_to(
                    grid_origin
                    + RIGHT * (frm * cell_w - grid_w / 2 + cell_w / 2)
                    + UP    * ((n_levels - 1 - lvl) * cell_h - grid_h / 2 + cell_h / 2)
                )
                grid.add(cell)

        # Level axis labels on left
        level_axis = VGroup()
        for i in range(n_levels):
            lbl = Text(f"L{i+1}", font_size=11, color=level_colors[i])
            lbl.move_to(
                grid_origin
                + LEFT * (grid_w / 2 + 0.35)
                + UP   * ((n_levels - 1 - i) * cell_h - grid_h / 2 + cell_h / 2)
            )
            level_axis.add(lbl)

        # Time axis label
        time_axis_lbl = Text("75 frames  (1 second  ·  13 ms each)", font_size=16, color=GREY_A)
        time_axis_lbl.next_to(grid_origin + DOWN * grid_h / 2, DOWN, buff=0.2)

        # Left label
        left_axis_lbl = Text("8 RVQ\nlevels", font_size=14, color=GREY_A)
        left_axis_lbl.next_to(grid_origin + LEFT * grid_w / 2, LEFT, buff=0.5)

        self.play(
            LaggedStart(*[FadeIn(cell, scale=0.8) for cell in grid], lag_ratio=0.003),
            run_time=2.0,
        )
        self.play(
            FadeIn(level_axis), Write(time_axis_lbl), Write(left_axis_lbl),
            run_time=0.6,
        )
        self.wait(0.5)

        # ══════════════════════════════════════════════════════════════════════
        # SCENE 5 — final comparison
        # ══════════════════════════════════════════════════════════════════════
        raw_stat  = Text("Raw:    24,000 floats  (96 KB)", font_size=19, color=BLUE_B)
        tok_stat  = Text("Codec:      600 integers  (< 1 KB)", font_size=19, color=GREEN_B)
        tok_stat2 = Text("8 levels × 75 frames = 600 tokens", font_size=16, color=GREY_A)

        stats = VGroup(raw_stat, tok_stat, tok_stat2).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        stats_box = SurroundingRectangle(stats, color=GREY_D, buff=0.25, corner_radius=0.12)

        stats_group = VGroup(stats_box, stats)
        stats_group.to_edge(DOWN, buff=0.22)

        self.play(
            FadeOut(pullback_title),
            FadeIn(stats_group),
            run_time=0.8,
        )

        punchline = Text(
            "This is how a language model generates speech.",
            font_size=22, color=WHITE,
        )
        punchline.to_edge(UP, buff=0.4)
        self.play(Write(punchline), run_time=0.9)
        self.wait(3.0)
