"""
Animation: Three Token Arrangement Strategies

Side-by-side comparison of how a [Q, T] codec token matrix gets flattened
into a 1D sequence for a transformer, and what each choice costs:

  • Flat interleaved  (VALL-E)       — Q × T sequential AR steps
  • Delay pattern     (MusicGen)     — T sequential steps, Q parallel heads
  • AR + NAR                         — T AR steps for coarse, 1 NAR step for fine
"""

from manim import *
import numpy as np


class TokenArrangements(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        Q = 4   # levels
        T = 5   # frames
        CW, CH = 0.52, 0.38
        GAP = 0.07

        LEVEL_COLORS = [RED_B, ORANGE, YELLOW_B, GREEN_B]
        COARSE_COLOR = RED_B
        FINE_COLORS  = [ORANGE, YELLOW_B, GREEN_B]

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("Three Ways to Arrange Codec Tokens", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title))
        self.wait(0.25)

        # ── Helper ────────────────────────────────────────────────────────────
        def cell(label, color, opacity=0.70):
            r = RoundedRectangle(corner_radius=0.05, width=CW, height=CH,
                                 fill_color=color, fill_opacity=opacity,
                                 stroke_color=color, stroke_width=1.1)
            t = Text(label, font_size=9, color=WHITE)
            t.move_to(r.get_center())
            return VGroup(r, t)

        def empty_cell(color):
            r = RoundedRectangle(corner_radius=0.05, width=CW, height=CH,
                                 fill_color=color, fill_opacity=0.14,
                                 stroke_color=color, stroke_width=0.8,
                                 stroke_opacity=0.40)
            return r

        def grid(rows_data, ox, oy):
            """rows_data: list of (label_prefix, color) per level."""
            grp = VGroup()
            for q, (prefix, color) in enumerate(rows_data):
                for t in range(T):
                    c = cell(f"{prefix}{t + 1}", color)
                    c.move_to([ox + t * (CW + GAP), oy - q * (CH + GAP), 0])
                    grp.add(c)
            return grp

        def stat_box(text_str, color):
            b = RoundedRectangle(corner_radius=0.08, width=3.20, height=0.54,
                                 fill_color=GREY_E, fill_opacity=0.55,
                                 stroke_color=color, stroke_width=1.3)
            t = Text(text_str, font_size=12, color=color)
            t.move_to(b.get_center())
            return VGroup(b, t)

        # Column centres
        col_xs = [-4.30, 0.00, 4.30]

        # ── Panel headers ──────────────────────────────────────────────────────
        headers = [
            ("Flat Interleaved", RED_C),
            ("Delay Pattern", YELLOW_B),
            ("AR + NAR", GREEN_B),
        ]
        header_grps = VGroup()
        for (hdr, color), cx in zip(headers, col_xs):
            h = Text(hdr, font_size=18, color=color)
            h.move_to([cx, 2.80, 0])
            header_grps.add(h)
        self.play(LaggedStart(*[Write(h) for h in header_grps], lag_ratio=0.20))

        # ── PANEL 1: Flat Interleaved ──────────────────────────────────────────
        # Sequence: [k1_1, k2_1, k3_1, k4_1, k1_2, k2_2, …]
        # Display as a single long column of Q×T cells arranged in a grid
        p1_oy = 1.80
        p1_grp = VGroup()
        for q in range(Q):
            for t in range(T):
                c = cell(f"k{q + 1}_{t + 1}", LEVEL_COLORS[q])
                c.move_to([col_xs[0] + t * (CW + GAP), p1_oy - q * (CH + GAP), 0])
                p1_grp.add(c)

        # Sequential step arrow below
        seq_arrow = Arrow(
            [col_xs[0] - 1.20, p1_oy - (Q - 0.5) * (CH + GAP) - 0.18, 0],
            [col_xs[0] + T * (CW + GAP) * 0.78, p1_oy - (Q - 0.5) * (CH + GAP) - 0.18, 0],
            color=RED_C, stroke_width=1.8, max_tip_length_to_length_ratio=0.10,
        )
        seq_lbl = Text(f"Q × T = {Q * T} AR steps", font_size=12, color=RED_C)
        seq_lbl.next_to(seq_arrow, DOWN, buff=0.12)

        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in p1_grp], lag_ratio=0.03),
            run_time=0.90,
        )
        self.play(GrowArrow(seq_arrow), FadeIn(seq_lbl), run_time=0.50)

        p1_stat = stat_box(f"AR steps: Q×T = {Q * T}  ·  Slow", RED_C)
        p1_stat.move_to([col_xs[0], -2.50, 0])
        self.play(FadeIn(p1_stat))
        self.wait(0.50)

        # ── PANEL 2: Delay Pattern ─────────────────────────────────────────────
        p2_oy = 1.80
        p2_raw = {}
        p2_grp = VGroup()
        for q in range(Q):
            for t in range(T):
                c = cell(f"k{q + 1}_{t + 1}", LEVEL_COLORS[q])
                # Shift each row right by q positions
                sx = col_xs[1] - (T - 1) * (CW + GAP) / 2 + (t + q) * (CW + GAP)
                c.move_to([sx, p2_oy - q * (CH + GAP), 0])
                p2_raw[(q, t)] = c
                p2_grp.add(c)

        # Empty placeholders
        p2_empty = VGroup()
        for q in range(1, Q):
            for t in range(q):
                sx = col_xs[1] - (T - 1) * (CW + GAP) / 2 + t * (CW + GAP)
                ec = empty_cell(LEVEL_COLORS[q])
                ec.move_to([sx, p2_oy - q * (CH + GAP), 0])
                p2_empty.add(ec)

        # Highlight one "diagonal" column (step s=3)
        s = 3
        diag_boxes = VGroup()
        for q in range(Q):
            audio_t = s - q
            if 0 <= audio_t < T:
                hb = SurroundingRectangle(
                    p2_raw[(q, audio_t)], color=WHITE, buff=0.05, stroke_width=1.8,
                )
                diag_boxes.add(hb)

        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in p2_grp], lag_ratio=0.03),
            FadeIn(p2_empty),
            run_time=0.90,
        )
        self.play(Create(diag_boxes), run_time=0.45)

        diag_lbl = Text("Q heads, 1 step", font_size=11, color=WHITE)
        diag_lbl.next_to(diag_boxes, RIGHT, buff=0.12)
        self.play(FadeIn(diag_lbl))

        p2_stat = stat_box(f"AR steps: T = {T}  ·  Q heads parallel", YELLOW_B)
        p2_stat.move_to([col_xs[1], -2.50, 0])
        self.play(FadeIn(p2_stat))
        self.wait(0.50)

        # ── PANEL 3: AR + NAR ─────────────────────────────────────────────────
        p3_oy = 1.80
        p3_cx = col_xs[2]
        row_w = (T - 1) * (CW + GAP)

        # Coarse row (AR)
        coarse_cells = VGroup()
        for t in range(T):
            c = cell(f"k₁_{t + 1}", COARSE_COLOR)
            c.move_to([p3_cx - row_w / 2 + t * (CW + GAP), p3_oy, 0])
            coarse_cells.add(c)

        ar_lbl = Text("AR  (sequential)", font_size=12, color=COARSE_COLOR)
        ar_lbl.next_to(coarse_cells, LEFT, buff=0.14)

        # Fine rows (NAR)
        fine_rows = VGroup()
        for q in range(Q - 1):
            row = VGroup()
            for t in range(T):
                c = cell(f"k{q + 2}_{t + 1}", FINE_COLORS[q])
                c.move_to([p3_cx - row_w / 2 + t * (CW + GAP),
                            p3_oy - (q + 1) * (CH + GAP) - 0.10, 0])
                row.add(c)
            fine_rows.add(row)

        nar_lbl = Text("NAR  (one pass)", font_size=12, color=YELLOW_B)
        nar_lbl.next_to(fine_rows, LEFT, buff=0.14)

        # Divider
        div_y = p3_oy - (CH + GAP) * 0.65
        divider = DashedLine(
            [p3_cx - row_w / 2 - 0.20, div_y, 0],
            [p3_cx + row_w / 2 + 0.20, div_y, 0],
            color=GREY_D, stroke_width=1.0, dash_length=0.12,
        )

        # Sequential arrow for AR row
        ar_arrow = Arrow(
            coarse_cells[0].get_left() + LEFT * 0.05 + DOWN * 0.25,
            coarse_cells[-1].get_right() + RIGHT * 0.05 + DOWN * 0.25,
            color=COARSE_COLOR, stroke_width=1.6,
            max_tip_length_to_length_ratio=0.10,
        )

        # Single wide arrow for NAR
        nar_arrow = Arrow(
            [p3_cx, div_y - 0.06, 0],
            [p3_cx, fine_rows.get_top()[1] + 0.06, 0],
            color=YELLOW_B, stroke_width=1.8,
            max_tip_length_to_length_ratio=0.12, buff=0.00,
        )

        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in coarse_cells], lag_ratio=0.08),
            FadeIn(ar_lbl), run_time=0.70,
        )
        self.play(GrowArrow(ar_arrow), run_time=0.45)
        self.play(Create(divider), run_time=0.30)
        self.play(GrowArrow(nar_arrow), run_time=0.35)
        self.play(
            LaggedStart(*[GrowFromCenter(c) for row in fine_rows for c in row],
                        lag_ratio=0.02),
            FadeIn(nar_lbl), run_time=0.70,
        )

        p3_stat = stat_box(f"AR steps: T = {T}  ·  NAR: 1 step  ·  Fastest", GREEN_B)
        p3_stat.move_to([col_xs[2], -2.50, 0])
        self.play(FadeIn(p3_stat))
        self.wait(3.00)
