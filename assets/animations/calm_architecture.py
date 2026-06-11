"""
Animation: CALM vs Moshi — Architecture Comparison

Shows the key architectural difference between Moshi and CALM:

Moshi: LM hidden state → RQ-Transformer (8 sequential AR steps over codebook levels)
       → 8 discrete codebook indices → Mimi decoder → waveform

CALM:  LM hidden state → Consistency MLP (~10M params, one shot)
       → continuous VAE latent → VAE decoder → waveform

The 70× parameter reduction in the audio generation head is the edge deployment story.
"""

from manim import *
import numpy as np


class CALMArchitecture(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("CALM vs Moshi: The Audio Generation Head", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title))
        subtitle = Text(
            "Replacing a 701M RQ-Transformer with a 10M consistency MLP",
            font_size=17, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(subtitle))
        self.wait(0.30)

        # ── Helper: rounded box ────────────────────────────────────────────────
        def box(label, color, w=2.40, h=0.62, sub=None):
            r = RoundedRectangle(corner_radius=0.08, width=w, height=h,
                                 fill_color=color, fill_opacity=0.22,
                                 stroke_color=color, stroke_width=1.6)
            t = Text(label, font_size=14, color=color)
            t.move_to(r.get_center())
            grp = VGroup(r, t)
            if sub:
                s = Text(sub, font_size=10, color=GREY_B)
                s.next_to(r, DOWN, buff=0.06)
                grp.add(s)
            return grp

        def arrow(start, end, color=WHITE):
            return Arrow(start, end, color=color, stroke_width=2.0,
                         max_tip_length_to_length_ratio=0.13, buff=0.07)

        # Column centres
        LX, RX = -3.30, 3.30
        col_lbl_y = 1.80

        # ── Column headers ─────────────────────────────────────────────────────
        moshi_hdr = Text("Moshi", font_size=22, color=RED_B)
        moshi_hdr.move_to([LX, col_lbl_y, 0])
        calm_hdr = Text("CALM", font_size=22, color=BLUE_B)
        calm_hdr.move_to([RX, col_lbl_y, 0])
        self.play(Write(moshi_hdr), Write(calm_hdr))

        # ── Shared: LM hidden state ────────────────────────────────────────────
        hstate_lbl = Text("LM hidden state  hₜ", font_size=16, color=GREY_A)
        hstate_lbl.move_to([0, col_lbl_y - 0.55, 0])
        hstate_line = Line([-5.5, col_lbl_y - 0.55, 0], [5.5, col_lbl_y - 0.55, 0],
                           color=GREY_D, stroke_width=0.8)
        self.play(Write(hstate_lbl), Create(hstate_line))

        # Arrow down to each column
        a_moshi = arrow([LX, col_lbl_y - 0.80, 0], [LX, col_lbl_y - 1.15, 0], RED_B)
        a_calm  = arrow([RX, col_lbl_y - 0.80, 0], [RX, col_lbl_y - 1.15, 0], BLUE_B)
        self.play(GrowArrow(a_moshi), GrowArrow(a_calm))

        # ── MOSHI path ─────────────────────────────────────────────────────────
        # RQ-Transformer: 8 sequential steps
        rq_y = col_lbl_y - 1.70
        rq_box = box("RQ-Transformer", RED_B, w=2.50, h=0.62, sub="701M parameters")
        rq_box.move_to([LX, rq_y, 0])
        self.play(FadeIn(rq_box))

        # Animated sequential steps inside the RQ-Transformer
        step_colors = [RED_B, ORANGE, YELLOW_B, GREEN_B, TEAL_B, BLUE_B, PURPLE_B, PINK]
        n_levels = 4   # show 4 for clarity (labelled 1-8)
        step_w, step_h = 0.45, 0.34
        steps_origin_x = LX - (n_levels - 1) * (step_w + 0.06) / 2
        steps_y = rq_y - 0.92

        step_boxes = VGroup()
        for i in range(n_levels):
            sx = steps_origin_x + i * (step_w + 0.06)
            sr = RoundedRectangle(corner_radius=0.04, width=step_w, height=step_h,
                                  fill_color=step_colors[i], fill_opacity=0.15,
                                  stroke_color=step_colors[i], stroke_width=1.1)
            sr.move_to([sx, steps_y, 0])
            st = Text(f"k{i + 1}", font_size=11, color=step_colors[i])
            st.move_to(sr.get_center())
            step_boxes.add(VGroup(sr, st))

        seq_lbl = Text("8 sequential steps", font_size=11, color=GREY_C)
        seq_lbl.next_to(step_boxes, DOWN, buff=0.08)
        ellipsis = Text("...", font_size=14, color=GREY_D)
        ellipsis.next_to(step_boxes, RIGHT, buff=0.10)

        step_arrow_down = arrow([LX, rq_y - 0.34, 0], [LX, steps_y + step_h / 2 + 0.06, 0], RED_B)
        self.play(GrowArrow(step_arrow_down))

        for sb in step_boxes:
            self.play(FadeIn(sb), run_time=0.22)
        self.play(FadeIn(seq_lbl), FadeIn(ellipsis))

        # Arrow down to Mimi decoder
        mimi_y = steps_y - 0.90
        mimi_arr = arrow([LX, steps_y - step_h / 2 - 0.08, 0], [LX, mimi_y + 0.33, 0], RED_B)
        mimi_box = box("Mimi decoder", RED_B, w=2.20, h=0.60)
        mimi_box.move_to([LX, mimi_y, 0])
        self.play(GrowArrow(mimi_arr), FadeIn(mimi_box))

        # ── CALM path ──────────────────────────────────────────────────────────
        mlp_y = rq_y
        mlp_box = box("Consistency MLP", BLUE_B, w=2.50, h=0.62, sub="~10M parameters")
        mlp_box.move_to([RX, mlp_y, 0])
        self.play(FadeIn(mlp_box))

        # One shot arrow
        latent_y = steps_y
        mlp_down_arr = arrow([RX, mlp_y - 0.34, 0], [RX, latent_y + 0.22, 0], BLUE_B)
        latent_box = box("VAE latent", BLUE_B, w=2.00, h=0.52, sub="continuous · 1 step")
        latent_box.move_to([RX, latent_y, 0])
        one_shot_lbl = Text("1 forward pass", font_size=11, color=BLUE_C)
        one_shot_lbl.next_to(latent_box, DOWN, buff=0.08)
        self.play(GrowArrow(mlp_down_arr), FadeIn(latent_box), FadeIn(one_shot_lbl))

        vae_y = mimi_y
        vae_arr = arrow([RX, latent_y - 0.28, 0], [RX, vae_y + 0.33, 0], BLUE_B)
        vae_box = box("VAE decoder", BLUE_B, w=2.20, h=0.60)
        vae_box.move_to([RX, vae_y, 0])
        self.play(GrowArrow(vae_arr), FadeIn(vae_box))

        # ── Both → waveform ────────────────────────────────────────────────────
        wave_y = mimi_y - 0.88
        mimi_wave_arr = arrow([LX, mimi_y - 0.32, 0], [LX, wave_y + 0.22, 0], GREY_A)
        vae_wave_arr  = arrow([RX, vae_y  - 0.32, 0], [RX, wave_y + 0.22, 0], GREY_A)
        wave_box_l = box("waveform", GREY_B, w=1.80, h=0.48)
        wave_box_r = box("waveform", GREY_B, w=1.80, h=0.48)
        wave_box_l.move_to([LX, wave_y, 0])
        wave_box_r.move_to([RX, wave_y, 0])
        self.play(
            GrowArrow(mimi_wave_arr), GrowArrow(vae_wave_arr),
            FadeIn(wave_box_l), FadeIn(wave_box_r),
        )
        self.wait(0.50)

        # ── Key callout ────────────────────────────────────────────────────────
        reduction = Text("70× fewer parameters in the audio generation head",
                         font_size=15, color=GREEN_C)
        reduction.to_edge(DOWN, buff=0.30)
        self.play(Write(reduction))

        # Bracket highlighting the difference
        left_grp  = VGroup(rq_box, step_boxes, seq_lbl, ellipsis)
        right_grp = VGroup(mlp_box, latent_box, one_shot_lbl)

        brace_l = Brace(left_grp, LEFT, color=RED_C)
        param_l = Text("701M", font_size=14, color=RED_C)
        param_l.next_to(brace_l, LEFT, buff=0.12)
        brace_r = Brace(right_grp, RIGHT, color=BLUE_C)
        param_r = Text("~10M", font_size=14, color=BLUE_C)
        param_r.next_to(brace_r, RIGHT, buff=0.12)

        self.play(
            GrowFromCenter(brace_l), Write(param_l),
            GrowFromCenter(brace_r), Write(param_r),
        )
        self.wait(3.00)
