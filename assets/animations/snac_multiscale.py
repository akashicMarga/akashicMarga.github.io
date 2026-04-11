"""
Animation: SNAC Multi-Scale Token Streams
Shows how coarse/mid/fine tokens align to audio time,
and how 1 coarse = 2 mid = 4 fine tokens.
"""

from manim import *
import numpy as np


class SNACMultiscale(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("SNAC: Multi-Scale Residual Quantization", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        sub = Text("Each codebook level runs at a different frame rate", font_size=20, color=GREY_A)
        sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub))

        # ── Draw audio waveform at top ─────────────────────────────────────────
        t = np.linspace(0, 1, 300)
        y = 0.5 * np.sin(2 * PI * 4 * t) + 0.3 * np.sin(2 * PI * 11 * t)
        y = y / np.max(np.abs(y)) * 0.4

        wave_axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-0.5, 0.5, 0.5],
            x_length=10,
            y_length=0.9,
            axis_config={"color": GREY_D, "stroke_width": 1},
            tips=False,
        ).shift(UP * 1.5)

        wave = wave_axes.plot_line_graph(
            x_values=t, y_values=y, line_color=GREY_B, stroke_width=1.5, add_vertex_dots=False
        )
        wave_lbl = Text("Audio (240ms segment)", font_size=14, color=GREY_A)
        wave_lbl.next_to(wave_axes, LEFT, buff=0.1)

        self.play(Create(wave_axes), Create(wave), Write(wave_lbl))

        # ── Draw three token rows ──────────────────────────────────────────────
        row_y = [0.4, -0.5, -1.5]
        row_labels = ["Coarse  (12.5fps)", "Mid     (25fps)", "Fine    (50fps)"]
        row_colors = [RED_B, ORANGE, YELLOW_B]
        row_counts = [3, 6, 12]

        token_groups = []
        for row_i, (ry, rlabel, rcolor, rcount) in enumerate(zip(row_y, row_labels, row_colors, row_counts)):
            grp = VGroup()
            for j in range(rcount):
                box = Rectangle(
                    width=10 / rcount - 0.08,
                    height=0.55,
                    color=rcolor,
                    fill_color=rcolor,
                    fill_opacity=0.25,
                    stroke_width=1.5,
                )
                # number inside
                num = Text(f"k{j}", font_size=11, color=rcolor)
                num.move_to(box.get_center())
                grp.add(VGroup(box, num))
            grp.arrange(RIGHT, buff=0.08)
            grp.move_to(np.array([0, ry, 0]))

            lbl = Text(rlabel, font_size=15, color=rcolor)
            lbl.next_to(grp, LEFT, buff=0.15)
            token_groups.append((grp, lbl))

        for grp, lbl in token_groups:
            self.play(
                LaggedStart(*[FadeIn(t, scale=0.8) for t in grp], lag_ratio=0.05),
                Write(lbl),
                run_time=0.6,
            )

        self.wait(0.4)

        # ── Show correspondence: 1 coarse = 2 mid = 4 fine ───────────────────
        coarse_grp = token_groups[0][0]
        mid_grp    = token_groups[1][0]
        fine_grp   = token_groups[2][0]

        # Highlight first coarse token and its corresponding mid+fine
        c0 = coarse_grp[0]
        m0, m1 = mid_grp[0], mid_grp[1]
        f0, f1, f2, f3 = fine_grp[0], fine_grp[1], fine_grp[2], fine_grp[3]

        highlight_c = SurroundingRectangle(c0, color=WHITE, stroke_width=2.5, buff=0.05)
        highlight_m = SurroundingRectangle(VGroup(m0, m1), color=WHITE, stroke_width=2, buff=0.05)
        highlight_f = SurroundingRectangle(VGroup(f0, f1, f2, f3), color=WHITE, stroke_width=2, buff=0.05)

        corr_lbl = Text("1 coarse token = 2 mid tokens = 4 fine tokens", font_size=18, color=WHITE)
        corr_lbl.to_edge(DOWN, buff=0.8)

        self.play(
            Create(highlight_c), Create(highlight_m), Create(highlight_f),
            Write(corr_lbl),
        )
        self.wait(0.8)

        # ── Show generation order ─────────────────────────────────────────────
        gen_lbl = Text(
            "Generation:\n"
            "Large AR model generates coarse (cheap, semantic)\n"
            "Small NAR model fills mid+fine per coarse token (fast)",
            font_size=16,
            color=GREY_A,
            line_spacing=1.3,
        )
        gen_lbl.to_edge(DOWN, buff=0.2)
        self.play(FadeOut(corr_lbl), Write(gen_lbl))
        self.wait(0.8)

        # ── Streaming problem ─────────────────────────────────────────────────
        self.play(FadeOut(gen_lbl))
        stream_lbl = Text(
            "Streaming problem: must wait for all 7 tokens\n"
            "(1 coarse + 2 mid + 4 fine) before decoding 80ms of audio",
            font_size=17,
            color=RED_B,
            line_spacing=1.3,
        )
        stream_lbl.to_edge(DOWN, buff=0.25)
        self.play(Write(stream_lbl))
        self.wait(2.5)
