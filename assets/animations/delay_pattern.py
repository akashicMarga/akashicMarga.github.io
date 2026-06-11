"""
Animation: The Delay Pattern (MusicGen)

Shows how staggering each codec level by one frame position lets a single
transformer step predict all Q codebook levels in parallel via Q output heads —
reducing AR decode steps from Q×T to T while keeping cross-level conditioning.
"""

from manim import *
import numpy as np


class DelayPattern(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        Q = 4   # codec levels (simplified from 8)
        T = 5   # frames
        CW, CH = 0.82, 0.50
        GAP = 0.10
        LEVEL_COLORS = [RED_B, ORANGE, YELLOW_B, GREEN_B]

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("The Delay Pattern", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.30)
        self.play(Write(title))
        subtitle = Text(
            "Stagger each level by one frame → Q heads predict in parallel",
            font_size=18, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(subtitle))
        self.wait(0.40)

        # ── Helper: coloured cell ──────────────────────────────────────────────
        def make_cell(row, col, label, color, opacity=0.72, ox=0.0, oy=0.0):
            x = ox + col * (CW + GAP)
            y = oy - row * (CH + GAP)
            rect = RoundedRectangle(
                corner_radius=0.06, width=CW, height=CH,
                fill_color=color, fill_opacity=opacity,
                stroke_color=color, stroke_width=1.3,
            )
            rect.move_to([x, y, 0])
            txt = Text(label, font_size=11, color=WHITE)
            txt.move_to(rect.get_center())
            return VGroup(rect, txt)

        def make_empty(row, col, ox=0.0, oy=0.0):
            x = ox + col * (CW + GAP)
            y = oy - row * (CH + GAP)
            rect = RoundedRectangle(
                corner_radius=0.06, width=CW, height=CH,
                fill_color=GREY_E, fill_opacity=0.25,
                stroke_color=GREY_D, stroke_width=0.9,
            )
            rect.move_to([x, y, 0])
            dash = Text("—", font_size=16, color=GREY_D)
            dash.move_to(rect.get_center())
            return VGroup(rect, dash)

        # Grid centred on screen
        total_w = T * (CW + GAP) - GAP
        total_h = Q * (CH + GAP) - GAP
        ox = -total_w / 2 + CW / 2
        oy = total_h / 2 - CH / 2 - 0.30

        # ── STEP 1: Unshifted grid ─────────────────────────────────────────────
        step1 = Text("① Unshifted: every level starts at frame 1",
                     font_size=16, color=GREY_B)
        step1.to_edge(DOWN, buff=1.30)
        self.play(Write(step1))

        # Level labels (left)
        level_lbls = VGroup()
        for q in range(Q):
            lbl = Text(f"L{q + 1}", font_size=15, color=LEVEL_COLORS[q])
            lbl.move_to([ox - CW * 1.05, oy - q * (CH + GAP), 0])
            level_lbls.add(lbl)

        # Frame labels (top)
        frame_lbls = VGroup()
        for t in range(T):
            lbl = Text(f"t{t + 1}", font_size=13, color=GREY_C)
            lbl.move_to([ox + t * (CW + GAP), oy + CH * 0.92, 0])
            frame_lbls.add(lbl)

        # Raw cells — store in dict for later manipulation
        raw = {}
        all_raw = VGroup()
        for q in range(Q):
            for t in range(T):
                c = make_cell(q, t, f"k{q + 1}_{t + 1}", LEVEL_COLORS[q], ox=ox, oy=oy)
                raw[(q, t)] = c
                all_raw.add(c)

        self.play(FadeIn(level_lbls), FadeIn(frame_lbls))
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in all_raw], lag_ratio=0.04),
            run_time=1.20,
        )
        self.wait(0.70)

        # ── STEP 2: Apply the delay ────────────────────────────────────────────
        self.play(FadeOut(step1))
        step2 = Text("② Shift level q right by q–1 positions (delay)",
                     font_size=16, color=YELLOW)
        step2.to_edge(DOWN, buff=1.30)
        self.play(Write(step2))

        shift_anims = []
        for q in range(Q):
            for t in range(T):
                shift_anims.append(
                    raw[(q, t)].animate.shift(RIGHT * q * (CW + GAP))
                )
        self.play(*shift_anims, run_time=1.10)

        # Empty placeholder cells for the vacated left positions
        placeholders = VGroup()
        for q in range(1, Q):
            for t in range(q):
                placeholders.add(make_empty(q, t, ox=ox, oy=oy))
        self.play(FadeIn(placeholders), run_time=0.50)
        self.wait(0.60)

        # ── STEP 3: One step — all Q heads fire in parallel ────────────────────
        self.play(FadeOut(step2))
        step3 = Text("③ At step s — one hidden state drives Q heads in parallel",
                     font_size=16, color=GREEN_B)
        step3.to_edge(DOWN, buff=1.30)
        self.play(Write(step3))

        # Choose step s=3 (0-indexed). The cell at (q, s-q) is now at screen col s.
        s = 3
        # Screen x of column s
        col_x = ox + s * (CW + GAP)

        # Highlight the diagonal cells at screen column s
        diag_boxes = VGroup()
        for q in range(Q):
            audio_t = s - q       # audio frame index (0-indexed)
            if 0 <= audio_t < T:
                hbox = SurroundingRectangle(
                    raw[(q, audio_t)], color=WHITE, buff=0.06, stroke_width=2.2,
                )
                diag_boxes.add(hbox)
        self.play(Create(diag_boxes), run_time=0.50)

        # Hidden state box below the grid
        hs_y = oy - (Q - 0.5) * (CH + GAP) - 0.55
        hs = RoundedRectangle(
            corner_radius=0.07, width=1.80, height=0.44,
            fill_color=BLUE_E, fill_opacity=0.72,
            stroke_color=BLUE_B, stroke_width=1.6,
        )
        hs.move_to([col_x, hs_y, 0])
        hs_lbl = Text("hₛ", font_size=15, color=BLUE_B)
        hs_lbl.move_to(hs.get_center())
        hidden = VGroup(hs, hs_lbl)
        self.play(FadeIn(hidden))

        # One arrow per level, each coloured by its level
        head_arrows = VGroup()
        for q in range(Q):
            audio_t = s - q
            if 0 <= audio_t < T:
                target_y = oy - q * (CH + GAP) + CH / 2
                target_x = ox + (audio_t + q) * (CW + GAP)   # screen x after shift
                arr = Arrow(
                    start=hs.get_top(),
                    end=[target_x, target_y, 0],
                    color=LEVEL_COLORS[q],
                    stroke_width=2.0,
                    max_tip_length_to_length_ratio=0.13,
                    buff=0.06,
                )
                head_arrows.add(arr)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in head_arrows], lag_ratio=0.12),
            run_time=0.90,
        )

        # Annotation
        heads_note = Text("Q heads — one per level, all from same hidden state",
                          font_size=14, color=GREY_A)
        heads_note.to_edge(DOWN, buff=0.30)
        self.play(Write(heads_note))
        self.wait(0.60)

        # Key result callout
        self.play(FadeOut(heads_note), FadeOut(step3))
        result = Text(
            "AR steps = T  (not Q × T)  ·  Sequence length = Q × T  ·  Quality unchanged",
            font_size=14, color=GREEN_C,
        )
        result.to_edge(DOWN, buff=0.30)
        self.play(Write(result))
        self.wait(2.80)
