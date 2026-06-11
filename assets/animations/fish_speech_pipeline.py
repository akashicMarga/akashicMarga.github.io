"""
Animation: Fish Speech Two-Pass AR Pipeline

Shows three ideas:
1. Text → Semantic LM → semantic tokens (speaker-agnostic, 25fps)
2. Semantic tokens → Acoustic LM → RVQ codec tokens (speaker-aware, 75fps)
   with the 3× temporal upsampling between the two token streams
3. Why semantic tokens are speaker-agnostic: two different speakers saying
   the same word produce identical semantic tokens but different codec tokens
"""

from manim import *
import numpy as np


class FishSpeechPipeline(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("Fish Speech: Two-Pass Autoregressive Pipeline", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title))
        subtitle = Text(
            "Text → semantic tokens (language)  →  acoustic tokens (voice)",
            font_size=17, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(subtitle))
        self.wait(0.30)

        # ── PART 1: Two-pass pipeline ──────────────────────────────────────────
        def box(label, color, w=2.0, h=0.58, sub=None):
            r = RoundedRectangle(corner_radius=0.07, width=w, height=h,
                                 fill_color=color, fill_opacity=0.22,
                                 stroke_color=color, stroke_width=1.5)
            t = Text(label, font_size=13, color=color)
            t.move_to(r.get_center())
            g = VGroup(r, t)
            if sub:
                s = Text(sub, font_size=10, color=GREY_C)
                s.next_to(r, DOWN, buff=0.06)
                g.add(s)
            return g

        def arr(start, end, color=WHITE):
            return Arrow(start, end, color=color, stroke_width=2.0,
                         max_tip_length_to_length_ratio=0.13, buff=0.07)

        row_y = 0.80

        # 1. Text input
        text_box = box("text tokens", BLUE_B, w=1.80, h=0.54)
        text_box.move_to([-5.20, row_y, 0])
        self.play(FadeIn(text_box))

        # 2. Semantic LM
        a1 = arr(text_box.get_right(), [-3.60, row_y, 0], BLUE_B)
        sem_lm_box = box("Semantic LM", GREEN_B, w=2.00, h=0.58, sub="text → sem tokens")
        sem_lm_box.move_to([-2.60, row_y, 0])
        self.play(GrowArrow(a1), FadeIn(sem_lm_box))

        # 3. Semantic tokens (25fps)
        sem_tok_w, sem_tok_h = 0.46, 0.40
        sem_gap = 0.08
        n_sem = 4
        a2 = arr(sem_lm_box.get_right(), [-1.10, row_y, 0], GREEN_B)
        sem_tokens = VGroup()
        for i in range(n_sem):
            r = RoundedRectangle(corner_radius=0.05, width=sem_tok_w, height=sem_tok_h,
                                 fill_color=GREEN_B, fill_opacity=0.65,
                                 stroke_color=GREEN_B, stroke_width=1.2)
            r.move_to([-0.70 + i * (sem_tok_w + sem_gap), row_y, 0])
            t = Text(f"s{i+1}", font_size=11, color=WHITE)
            t.move_to(r.get_center())
            sem_tokens.add(VGroup(r, t))
        sem_fps_lbl = Text("25 fps", font_size=12, color=GREEN_C)
        sem_fps_lbl.next_to(sem_tokens, UP, buff=0.12)
        sem_lbl = Text("semantic tokens\n(speaker-agnostic)", font_size=11, color=GREEN_C,
                       line_spacing=1.2)
        sem_lbl.next_to(sem_tokens, DOWN, buff=0.12)
        self.play(GrowArrow(a2))
        self.play(
            LaggedStart(*[GrowFromCenter(t) for t in sem_tokens], lag_ratio=0.12),
            run_time=0.70,
        )
        self.play(FadeIn(sem_fps_lbl), FadeIn(sem_lbl))
        self.wait(0.40)

        # 4. Acoustic LM
        a3 = arr(sem_tokens.get_right() + RIGHT * 0.10, [1.70, row_y, 0], GREEN_B)
        acou_lm_box = box("Acoustic LM", RED_B, w=2.00, h=0.58, sub="sem → RVQ tokens")
        acou_lm_box.move_to([2.70, row_y, 0])
        self.play(GrowArrow(a3), FadeIn(acou_lm_box))

        # Also show speaker embedding going into Acoustic LM
        spk_box = RoundedRectangle(corner_radius=0.06, width=1.50, height=0.44,
                                   fill_color=ORANGE, fill_opacity=0.22,
                                   stroke_color=ORANGE, stroke_width=1.3)
        spk_box.move_to([2.70, row_y + 1.10, 0])
        spk_lbl = Text("speaker emb", font_size=12, color=ORANGE)
        spk_lbl.move_to(spk_box.get_center())
        spk_arr = arr(spk_box.get_bottom(), acou_lm_box.get_top(), ORANGE)
        self.play(FadeIn(VGroup(spk_box, spk_lbl)), GrowArrow(spk_arr))

        # 5. RVQ tokens (75fps — 3× more than semantic)
        n_rvq = 12   # 3× n_sem
        rvq_tok_w = 0.27
        rvq_gap  = 0.04
        a4 = arr(acou_lm_box.get_right(), [3.90, row_y, 0], RED_B)
        rvq_tokens = VGroup()
        rvq_colors = [RED_B, ORANGE, YELLOW_B, GREEN_B] * 3
        rvq_origin_x = 4.10
        for i in range(n_rvq):
            r = RoundedRectangle(corner_radius=0.04, width=rvq_tok_w, height=sem_tok_h,
                                 fill_color=rvq_colors[i], fill_opacity=0.60,
                                 stroke_color=rvq_colors[i], stroke_width=1.0)
            r.move_to([rvq_origin_x + i * (rvq_tok_w + rvq_gap), row_y, 0])
            rvq_tokens.add(r)
        rvq_fps_lbl = Text("75 fps  (3×)", font_size=12, color=RED_C)
        rvq_fps_lbl.next_to(rvq_tokens, UP, buff=0.12)
        rvq_lbl = Text("RVQ codec tokens\n(speaker-aware)", font_size=11, color=RED_C,
                       line_spacing=1.2)
        rvq_lbl.next_to(rvq_tokens, DOWN, buff=0.12)
        self.play(GrowArrow(a4))
        self.play(
            LaggedStart(*[GrowFromCenter(t) for t in rvq_tokens], lag_ratio=0.04),
            run_time=0.60,
        )
        self.play(FadeIn(rvq_fps_lbl), FadeIn(rvq_lbl))
        self.wait(0.50)

        # ── PART 2: 3× upsampling bracket ─────────────────────────────────────
        # Show that 1 semantic frame → 3 acoustic frames
        upsample_note = Text(
            "1 semantic frame  →  3 acoustic frames  (learned by acoustic LM)",
            font_size=13, color=GREY_B,
        )
        upsample_note.to_edge(DOWN, buff=1.50)
        self.play(Write(upsample_note))

        # Draw bracket connecting one semantic token to three RVQ tokens
        sem1 = sem_tokens[1]
        rvq3 = VGroup(*rvq_tokens[3:6])
        brace = Brace(rvq3, DOWN, color=YELLOW_B)
        brace_lbl = Text("3 frames", font_size=11, color=YELLOW_B)
        brace_lbl.next_to(brace, DOWN, buff=0.06)
        sem_pointer = DashedLine(
            sem1.get_bottom() + DOWN * 0.05,
            brace.get_left() + LEFT * 0.05,
            color=YELLOW_B, stroke_width=1.0, dash_length=0.10,
        )
        self.play(Create(sem_pointer), GrowFromCenter(brace), Write(brace_lbl))
        self.wait(0.60)

        # ── PART 3: Speaker-agnostic semantic tokens ───────────────────────────
        self.play(
            FadeOut(upsample_note), FadeOut(sem_pointer),
            FadeOut(brace), FadeOut(brace_lbl),
        )

        spk_note = Text(
            "Same text, two speakers → identical semantic tokens, different codec tokens",
            font_size=13, color=GREEN_B,
        )
        spk_note.to_edge(DOWN, buff=1.50)
        self.play(Write(spk_note))

        # Two speaker icons below the pipeline
        def speaker_icon(cx, cy, color, label):
            circle = Circle(radius=0.22, fill_color=color, fill_opacity=0.50,
                           stroke_color=color, stroke_width=1.5)
            circle.move_to([cx, cy, 0])
            body = RoundedRectangle(corner_radius=0.08, width=0.40, height=0.30,
                                    fill_color=color, fill_opacity=0.40,
                                    stroke_color=color, stroke_width=1.2)
            body.next_to(circle, DOWN, buff=0.04)
            lbl = Text(label, font_size=11, color=color)
            lbl.next_to(body, DOWN, buff=0.06)
            return VGroup(circle, body, lbl)

        spk_a = speaker_icon(-1.50, -1.90, ORANGE, "speaker A")
        spk_b = speaker_icon(-0.10, -1.90, TEAL_B, "speaker B")
        self.play(FadeIn(spk_a), FadeIn(spk_b))

        same_sem = Text('both → "s₁ s₂ s₃ s₄" (identical)', font_size=12, color=GREEN_C)
        same_sem.move_to([1.80, -1.90, 0])
        self.play(Write(same_sem))

        diff_codec = Text("but different codec tokens at level 1 (different timbre)",
                          font_size=11, color=RED_C)
        diff_codec.next_to(same_sem, DOWN, buff=0.18)
        self.play(Write(diff_codec))
        self.wait(2.80)
