"""
Animation: Moshi's Inner Monologue

Shows three ideas in sequence:

1. Full-duplex: user stream and system stream run simultaneously — no turn-taking.
2. Per-timestep ordering: at each 80ms frame, the system predicts a text token
   FIRST (the inner monologue) and THEN 8 audio tokens conditioned on it.
3. Why this matters: text token grounds the audio tokens linguistically and
   doubles as implicit streaming ASR with no extra model.
"""

from manim import *
import numpy as np


class MoshiInnerMonologue(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        N_STEPS = 5     # timesteps to animate
        N_AUDIO  = 4    # audio token levels per step (simplified from 8)

        TEXT_COLOR   = BLUE_B
        SYS_COLOR    = RED_B
        USER_COLOR   = GREY_C
        INNER_COLOR  = TEAL_B

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("Moshi: The Inner Monologue", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title))
        subtitle = Text(
            "Text token first → audio tokens conditioned on it",
            font_size=18, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(subtitle))
        self.wait(0.35)

        # ── PART 1: Two parallel streams (full-duplex overview) ────────────────
        part1_lbl = Text("① Full-duplex: user and system run simultaneously",
                         font_size=16, color=GREY_B)
        part1_lbl.to_edge(DOWN, buff=1.30)
        self.play(Write(part1_lbl))

        step_w, step_h = 1.00, 0.46
        step_gap = 0.14
        row_gap  = 0.90
        origin_x = -((N_STEPS * (step_w + step_gap) - step_gap) / 2 - step_w / 2)
        user_y   = +0.40
        sys_y    = user_y - row_gap

        # Row labels
        user_lbl = Text("user audio", font_size=14, color=USER_COLOR)
        user_lbl.move_to([origin_x - 1.50, user_y, 0])
        sys_lbl  = Text("system audio", font_size=14, color=SYS_COLOR)
        sys_lbl.move_to([origin_x - 1.50, sys_y, 0])
        self.play(FadeIn(user_lbl), FadeIn(sys_lbl))

        # Animate steps appearing simultaneously in both rows
        step_labels = VGroup()
        for s in range(N_STEPS):
            x = origin_x + s * (step_w + step_gap)
            t_lbl = Text(f"t{s + 1}", font_size=12, color=GREY_D)
            t_lbl.move_to([x, user_y + step_h * 0.85, 0])
            step_labels.add(t_lbl)

        self.play(FadeIn(step_labels))

        user_cells = VGroup()
        sys_cells  = VGroup()
        for s in range(N_STEPS):
            x = origin_x + s * (step_w + step_gap)
            # User block
            ub = RoundedRectangle(corner_radius=0.05, width=step_w, height=step_h,
                                  fill_color=USER_COLOR, fill_opacity=0.20,
                                  stroke_color=USER_COLOR, stroke_width=1.2)
            ub.move_to([x, user_y, 0])
            # System block
            sb = RoundedRectangle(corner_radius=0.05, width=step_w, height=step_h,
                                  fill_color=SYS_COLOR, fill_opacity=0.20,
                                  stroke_color=SYS_COLOR, stroke_width=1.2)
            sb.move_to([x, sys_y, 0])
            user_cells.add(ub)
            sys_cells.add(sb)
            self.play(FadeIn(ub), FadeIn(sb), run_time=0.22)

        no_turntake = Text("No VAD. No turn-taking. Both streams at every 80ms frame.",
                           font_size=13, color=GREY_C)
        no_turntake.to_edge(DOWN, buff=0.30)
        self.play(Write(no_turntake))
        self.wait(1.20)

        # ── PART 2: Zoom into one system timestep — text first, audio second ───
        self.play(
            FadeOut(part1_lbl), FadeOut(no_turntake),
            FadeOut(user_cells), FadeOut(sys_cells),
            FadeOut(step_labels), FadeOut(user_lbl), FadeOut(sys_lbl),
        )

        part2_lbl = Text("② Per-timestep ordering: what actually happens inside one step",
                         font_size=15, color=GREY_B)
        part2_lbl.to_edge(DOWN, buff=1.30)
        self.play(Write(part2_lbl))

        # Timeline row for one step, expanded
        step_count = 4
        box_w, box_h = 1.30, 0.58
        box_gap = 0.18
        row_y = 0.30
        start_x = -((step_count * (box_w + box_gap) - box_gap) / 2 - box_w / 2)

        # Step 0: inner text token
        inner_rect = RoundedRectangle(
            corner_radius=0.08, width=box_w, height=box_h,
            fill_color=INNER_COLOR, fill_opacity=0.72,
            stroke_color=INNER_COLOR, stroke_width=1.6,
        )
        inner_rect.move_to([start_x, row_y, 0])
        inner_txt = Text("inner\ntext", font_size=13, color=WHITE)
        inner_txt.move_to(inner_rect.get_center())
        inner_lbl = Text("① inner monologue\n(text token)", font_size=12, color=INNER_COLOR)
        inner_lbl.next_to(inner_rect, DOWN, buff=0.16)

        self.play(GrowFromCenter(VGroup(inner_rect, inner_txt)))
        self.play(FadeIn(inner_lbl))
        self.wait(0.40)

        # Arrow to audio tokens
        audio_boxes = VGroup()
        audio_labels = VGroup()
        audio_colors = [RED_B, ORANGE, YELLOW_B, GREEN_B]
        for i in range(step_count):
            x = start_x + (i + 1) * (box_w + box_gap)
            ab = RoundedRectangle(
                corner_radius=0.07, width=box_w * 0.82, height=box_h,
                fill_color=audio_colors[i], fill_opacity=0.65,
                stroke_color=audio_colors[i], stroke_width=1.3,
            )
            ab.move_to([x, row_y, 0])
            al = Text(f"a_{i + 1}", font_size=13, color=WHITE)
            al.move_to(ab.get_center())
            audio_boxes.add(VGroup(ab, al))

        audio_row_lbl = Text("② system audio tokens (levels 1–8)", font_size=12, color=SYS_COLOR)
        audio_row_lbl.next_to(audio_boxes, DOWN, buff=0.16)

        cond_arrow = Arrow(
            inner_rect.get_right(),
            audio_boxes[0].get_left() + LEFT * 0.08,
            color=WHITE, stroke_width=2.0,
            max_tip_length_to_length_ratio=0.13, buff=0.08,
        )
        cond_note = Text("conditions", font_size=11, color=GREY_B)
        cond_note.next_to(cond_arrow, UP, buff=0.06)

        self.play(GrowArrow(cond_arrow), Write(cond_note))
        self.play(
            LaggedStart(*[GrowFromCenter(b) for b in audio_boxes], lag_ratio=0.12),
            run_time=0.80,
        )
        self.play(FadeIn(audio_row_lbl))
        self.wait(0.80)

        # ── PART 3: Three benefits of the inner monologue ─────────────────────
        self.play(FadeOut(part2_lbl))
        self.play(FadeOut(inner_lbl), FadeOut(audio_row_lbl), FadeOut(cond_note))

        benefits = [
            ("① Linguistic grounding",
             "forces hidden state to know WHAT to say\nbefore committing to acoustics",
             TEAL_B),
            ("② Implicit ASR",
             "inner text tokens = transcription\nno separate ASR model needed",
             BLUE_B),
            ("③ Streaming control",
             "text token conditions audio tokens\nwithin the same 80ms frame",
             GREEN_B),
        ]

        ben_grp = VGroup()
        for i, (head, body, color) in enumerate(benefits):
            h = Text(head, font_size=14, color=color)
            b = Text(body, font_size=12, color=GREY_A, line_spacing=1.25)
            b.next_to(h, DOWN, buff=0.10)
            grp = VGroup(h, b)
            ben_grp.add(grp)

        ben_grp.arrange(RIGHT, buff=0.55, aligned_edge=UP)
        ben_grp.to_edge(DOWN, buff=0.30)

        self.play(
            LaggedStart(*[FadeIn(g, shift=UP * 0.15) for g in ben_grp], lag_ratio=0.25),
            run_time=1.10,
        )
        self.wait(3.00)
