"""
Animation: AR + NAR Split

Shows how the two-stage pipeline works:
  Stage 1 — Autoregressive (AR): coarse (level-1) tokens generated one-by-one,
             because speaker identity and prosody need sequential context.
  Stage 2 — Non-Autoregressive (NAR): fine tokens (levels 2–Q) predicted in a
             single parallel forward pass conditioned on the coarse sequence.

The visual makes the speed asymmetry obvious: the AR stage is a slow
sequential chain; the NAR stage is one wide parallel shot.
"""

from manim import *
import numpy as np


class ARNARSplit(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        T = 6           # number of coarse frames
        Q_FINE = 3      # fine levels to show (simplified from 7)
        CW, CH = 0.70, 0.46
        GAP = 0.09

        COARSE_COLOR = RED_B
        FINE_COLORS  = [ORANGE, YELLOW_B, GREEN_B]
        TEXT_COLOR   = BLUE_B

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("AR + NAR: Where to Spend Your Compute", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title))
        subtitle = Text(
            "Sequential AR for coarse structure  ·  Parallel NAR for fine detail",
            font_size=17, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(subtitle))
        self.wait(0.35)

        # ── Helper ────────────────────────────────────────────────────────────
        def make_cell(label, color, opacity=0.70):
            rect = RoundedRectangle(
                corner_radius=0.06, width=CW, height=CH,
                fill_color=color, fill_opacity=opacity,
                stroke_color=color, stroke_width=1.3,
            )
            txt = Text(label, font_size=11, color=WHITE)
            txt.move_to(rect.get_center())
            return VGroup(rect, txt)

        def make_empty_cell(color):
            rect = RoundedRectangle(
                corner_radius=0.06, width=CW, height=CH,
                fill_color=color, fill_opacity=0.12,
                stroke_color=color, stroke_width=1.0,
                stroke_opacity=0.40,
            )
            q = Text("?", font_size=14, color=color, fill_opacity=0.40)
            q.move_to(rect.get_center())
            return VGroup(rect, q)

        # ── STAGE 1: AR — sequential coarse generation ─────────────────────────
        stage1_lbl = Text("Stage 1  ·  AR model", font_size=18, color=COARSE_COLOR)
        stage1_lbl.shift(LEFT * 3.60 + UP * 1.00)
        self.play(Write(stage1_lbl))

        # Text input (static)
        text_box = RoundedRectangle(
            corner_radius=0.08, width=2.20, height=0.56,
            fill_color=BLUE_E, fill_opacity=0.60,
            stroke_color=TEXT_COLOR, stroke_width=1.5,
        )
        text_box.shift(LEFT * 3.60 + UP * 0.00)
        text_lbl = Text("text tokens", font_size=13, color=TEXT_COLOR)
        text_lbl.move_to(text_box.get_center())
        text_grp = VGroup(text_box, text_lbl)
        self.play(FadeIn(text_grp))

        # Coarse token row (starts empty)
        coarse_row_origin = text_box.get_right() + RIGHT * 0.50 + DOWN * 0.0
        coarse_cells = []
        coarse_grp = VGroup()
        for t in range(T):
            c = make_empty_cell(COARSE_COLOR)
            c.move_to(
                coarse_row_origin + RIGHT * t * (CW + GAP) + RIGHT * CW / 2
            )
            coarse_cells.append(c)
            coarse_grp.add(c)

        coarse_label = Text("coarse (level 1)", font_size=12, color=COARSE_COLOR)
        coarse_label.next_to(coarse_grp, DOWN, buff=0.12)
        self.play(FadeIn(coarse_grp), FadeIn(coarse_label))

        # Sequential AR generation: fill one cell at a time with a "step" arrow
        for t in range(T):
            filled = make_cell(f"k₁_{t + 1}", COARSE_COLOR)
            filled.move_to(coarse_cells[t].get_center())

            if t == 0:
                arr = Arrow(
                    text_box.get_right(), coarse_cells[0].get_left(),
                    color=COARSE_COLOR, stroke_width=1.8,
                    max_tip_length_to_length_ratio=0.14, buff=0.06,
                )
            else:
                arr = Arrow(
                    coarse_cells[t - 1].get_right(), coarse_cells[t].get_left(),
                    color=COARSE_COLOR, stroke_width=1.8,
                    max_tip_length_to_length_ratio=0.14, buff=0.06,
                )
            self.play(GrowArrow(arr), run_time=0.28)
            self.play(FadeOut(coarse_grp[t]), FadeIn(filled), run_time=0.25)
            coarse_grp.remove(coarse_grp[t])
            coarse_grp.add(filled)
            coarse_cells[t] = filled

        seq_note = Text(
            f"{T} sequential steps — expensive, but necessary for prosody & speaker identity",
            font_size=12, color=GREY_B,
        )
        seq_note.next_to(coarse_grp, UP, buff=0.22)
        self.play(Write(seq_note))
        self.wait(0.60)

        # ── STAGE 2: NAR — parallel fine generation ────────────────────────────
        stage2_lbl = Text("Stage 2  ·  NAR model", font_size=18, color=YELLOW_B)
        stage2_lbl.shift(LEFT * 3.60 + DOWN * 1.20)
        self.play(Write(stage2_lbl))

        # Fine token grid (all empty to start)
        fine_origin = text_box.get_right() + RIGHT * 0.50 + DOWN * 1.80
        fine_cells = {}
        fine_grp = VGroup()
        for q in range(Q_FINE):
            for t in range(T):
                c = make_empty_cell(FINE_COLORS[q])
                c.move_to(
                    fine_origin + RIGHT * t * (CW + GAP) + RIGHT * CW / 2
                              + DOWN * q * (CH + GAP)
                )
                fine_cells[(q, t)] = c
                fine_grp.add(c)

        fine_label = Text("fine levels 2–Q", font_size=12, color=YELLOW_B)
        fine_label.next_to(fine_grp, DOWN, buff=0.12)
        self.play(FadeIn(fine_grp), FadeIn(fine_label))
        self.wait(0.30)

        # One big parallel arrow from entire coarse row to fine grid
        par_arrow = Arrow(
            coarse_grp.get_bottom() + DOWN * 0.08,
            fine_grp.get_top()   + UP * 0.08,
            color=YELLOW_B, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.10, buff=0.06,
        )
        par_note_top = Text("one forward pass", font_size=13, color=YELLOW_C)
        par_note_top.next_to(par_arrow, RIGHT, buff=0.14)

        self.play(GrowArrow(par_arrow), Write(par_note_top), run_time=0.60)

        # Fill all fine cells simultaneously
        filled_fine = VGroup()
        for q in range(Q_FINE):
            for t in range(T):
                f = make_cell(f"k{q + 2}_{t + 1}", FINE_COLORS[q])
                f.move_to(fine_cells[(q, t)].get_center())
                filled_fine.add(f)

        self.play(
            FadeOut(fine_grp),
            LaggedStart(*[GrowFromCenter(f) for f in filled_fine], lag_ratio=0.02),
            run_time=0.80,
        )

        par_note2 = Text(
            f"Q–1 levels × {T} frames filled simultaneously — one step",
            font_size=12, color=GREY_B,
        )
        par_note2.next_to(filled_fine, DOWN, buff=0.16)
        self.play(Write(par_note2))
        self.wait(0.70)

        # ── Speed comparison callout ───────────────────────────────────────────
        self.play(FadeOut(seq_note), FadeOut(par_note2), FadeOut(par_note_top))

        speed_box = RoundedRectangle(
            corner_radius=0.10, width=5.80, height=0.80,
            fill_color=GREY_E, fill_opacity=0.50,
            stroke_color=GREY_C, stroke_width=1.2,
        )
        speed_box.to_edge(DOWN, buff=0.25)
        speed_txt = Text(
            f"AR: {T} sequential steps  ·  NAR: 1 parallel step  ·  "
            f"Total ≈ {T + 1} steps  (vs {T * (Q_FINE + 1)} for full AR)",
            font_size=13, color=GREEN_C,
        )
        speed_txt.move_to(speed_box.get_center())
        self.play(FadeIn(speed_box), Write(speed_txt))
        self.wait(3.00)
