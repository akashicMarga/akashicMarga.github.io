"""
Animation 2: Vector Quantization
Shows a 2D latent space with continuous encoder outputs,
then snaps each point to its nearest codebook entry —
the continuous becomes discrete (integer indices).
"""

from manim import *
import numpy as np


class VectorQuantization(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"
        rng = np.random.default_rng(42)

        # ── Title ──────────────────────────────────────────────────────────────
        title = Text("Vector Quantization: Continuous → Discrete", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)

        # ── Axes (latent space) ────────────────────────────────────────────────
        axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=7,
            y_length=7,
            axis_config={"color": GREY_C, "stroke_width": 1},
            tips=False,
        ).shift(DOWN * 0.3)

        axis_label = Text("Encoder latent space (2D projection)", font_size=18, color=GREY_A)
        axis_label.next_to(axes, DOWN, buff=0.15)
        self.play(Create(axes), Write(axis_label), run_time=0.8)

        # ── Codebook entries (fixed, large colored dots) ───────────────────────
        codebook_coords = np.array([
            [-2.0,  1.5],
            [ 0.2,  2.5],
            [ 2.2,  1.8],
            [-2.3, -0.5],
            [ 0.0,  0.0],
            [ 2.5, -0.8],
            [-1.5, -2.5],
            [ 1.0, -2.0],
        ])
        N = len(codebook_coords)
        cb_colors = [RED, ORANGE, YELLOW, GREEN, TEAL, BLUE, PURPLE, PINK]

        cb_dots = VGroup()
        cb_labels = VGroup()
        for i, (cx, cy) in enumerate(codebook_coords):
            dot = Dot(axes.c2p(cx, cy), radius=0.18, color=cb_colors[i])
            dot.set_fill(cb_colors[i], opacity=0.9)
            label = Text(f"e{i}", font_size=14, color=cb_colors[i])
            label.next_to(dot, UP + RIGHT, buff=0.05)
            cb_dots.add(dot)
            cb_labels.add(label)

        codebook_legend = Text("◆ Codebook entries", font_size=16, color=GREY_A)
        codebook_legend.to_corner(UR, buff=0.5)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in cb_dots], lag_ratio=0.1),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[Write(l) for l in cb_labels], lag_ratio=0.05),
            Write(codebook_legend),
            run_time=0.8,
        )
        self.wait(0.4)

        # ── Encoder outputs: random continuous points ──────────────────────────
        n_points = 18
        # Cluster the points loosely around codebook entries
        enc_coords = []
        assignments = []
        for i in range(n_points):
            cb_idx = rng.integers(0, N)
            noise = rng.normal(scale=0.55, size=2)
            pt = codebook_coords[cb_idx] + noise
            pt = np.clip(pt, -3.2, 3.2)
            enc_coords.append(pt)
            assignments.append(cb_idx)
        enc_coords = np.array(enc_coords)

        enc_legend = Text("● Encoder outputs (continuous)", font_size=16, color=WHITE)
        enc_legend.next_to(codebook_legend, DOWN, buff=0.2)

        enc_dots = VGroup()
        for pt in enc_coords:
            dot = Dot(axes.c2p(pt[0], pt[1]), radius=0.09, color=WHITE)
            dot.set_fill(WHITE, opacity=0.85)
            enc_dots.add(dot)

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in enc_dots], lag_ratio=0.04),
            Write(enc_legend),
            run_time=1.2,
        )
        self.wait(0.5)

        # ── Step label ────────────────────────────────────────────────────────
        step_label = Text(
            "Find nearest codebook entry for each point",
            font_size=20, color=YELLOW,
        )
        step_label.to_edge(DOWN, buff=0.8)
        self.play(Write(step_label))

        # ── Draw arrows from enc point to nearest codebook entry ──────────────
        arrows = VGroup()
        for i, (pt, idx) in enumerate(zip(enc_coords, assignments)):
            cb = codebook_coords[idx]
            arr = Arrow(
                start=axes.c2p(pt[0], pt[1]),
                end=axes.c2p(cb[0], cb[1]),
                color=cb_colors[idx],
                stroke_width=1.5,
                buff=0.15,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(arr)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05),
            run_time=1.8,
        )
        self.wait(0.5)

        # ── Snap enc dots to codebook positions and color them ─────────────────
        snap_label = Text(
            "Snap! Each point becomes an integer index k",
            font_size=20, color=GREEN,
        )
        snap_label.to_edge(DOWN, buff=0.8)

        snap_animations = []
        for i, (enc_dot, idx) in enumerate(zip(enc_dots, assignments)):
            cb = codebook_coords[idx]
            snap_animations.append(
                enc_dot.animate
                .move_to(axes.c2p(cb[0], cb[1]))
                .set_color(cb_colors[idx])
                .scale(1.4)
            )

        self.play(FadeOut(step_label), FadeIn(snap_label))
        self.play(
            LaggedStart(*snap_animations, lag_ratio=0.03),
            FadeOut(arrows),
            run_time=1.8,
        )
        self.wait(0.5)

        # ── Show integer indices appearing ────────────────────────────────────
        idx_label = Text(
            "Continuous encoder output  →  integer {k}",
            font_size=20, color=BLUE_A,
        )
        idx_label.to_edge(DOWN, buff=0.8)

        # Show a single example with index label
        example_idx = 2
        cb_x, cb_y = codebook_coords[example_idx]
        idx_bubble = Text(f"k = {example_idx}", font_size=22, color=YELLOW)
        idx_bubble.next_to(axes.c2p(cb_x, cb_y), RIGHT + UP * 0.5, buff=0.15)
        brace_rect = SurroundingRectangle(idx_bubble, color=YELLOW, buff=0.1, corner_radius=0.1)

        self.play(FadeOut(snap_label), FadeIn(idx_label))
        self.play(Write(idx_bubble), Create(brace_rect))
        self.wait(1.0)

        # ── Fade to summary ────────────────────────────────────────────────────
        summary = Text(
            "N codebook entries  →  log₂(N) bits per frame",
            font_size=20, color=GREY_A,
        )
        summary.to_edge(DOWN, buff=0.3)
        self.play(
            FadeOut(idx_label),
            FadeOut(idx_bubble),
            FadeOut(brace_rect),
            Write(summary),
            run_time=0.8,
        )
        self.wait(2)
