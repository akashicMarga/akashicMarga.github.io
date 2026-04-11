"""
Animation: GAN Decoder & Discriminator
Shows how the generator (decoder) and multi-scale discriminators
play a minimax game to produce perceptually sharp audio.
"""

from manim import *
import numpy as np


class GANDecoder(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("GAN Decoder: How the Audio Sounds Good", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        sub = Text("Generator vs Discriminator — a minimax game", font_size=20, color=GREY_A)
        sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub))
        self.wait(0.3)

        # ── Main diagram ──────────────────────────────────────────────────────
        # [codec tokens] -> [Generator/Decoder] -> [fake audio] -> [Discriminator] -> real/fake
        #                                           [real audio] -> [Discriminator]

        def make_box(label, color, w=2.0, h=1.0):
            box = Rectangle(width=w, height=h, color=color, fill_opacity=0.18)
            txt = Text(label, font_size=15, color=color)
            txt.move_to(box.get_center())
            return VGroup(box, txt)

        tokens_box  = make_box("RVQ tokens\n[k1,k2,...,kQ]", BLUE_B, w=2.2, h=1.0)
        gen_box     = make_box("Generator\n(CNN Decoder)", GREEN_B, w=2.2, h=1.0)
        fake_box    = make_box("Fake audio", RED_B, w=1.8, h=0.8)
        real_box    = make_box("Real audio", TEAL_B, w=1.8, h=0.8)
        disc_box    = make_box("Discriminator\n(multi-scale)", ORANGE, w=2.2, h=1.0)
        output_box  = make_box("real / fake\n(logit)", YELLOW_B, w=1.8, h=0.8)

        tokens_box.shift(LEFT * 5 + UP * 0.3)
        gen_box.shift(LEFT * 2.2 + UP * 0.3)
        fake_box.shift(RIGHT * 0.5 + UP * 0.8)
        real_box.shift(RIGHT * 0.5 + DOWN * 0.4)
        disc_box.shift(RIGHT * 3.0 + UP * 0.2)
        output_box.shift(RIGHT * 5.5 + UP * 0.2)

        self.play(
            FadeIn(tokens_box), FadeIn(gen_box),
            FadeIn(fake_box), FadeIn(real_box),
            FadeIn(disc_box), FadeIn(output_box),
            run_time=0.8,
        )

        # Arrows
        arr_tok_gen  = Arrow(tokens_box.get_right(), gen_box.get_left(), color=BLUE_A, stroke_width=2, buff=0.05)
        arr_gen_fake = Arrow(gen_box.get_right(), fake_box.get_left(), color=GREEN_A, stroke_width=2, buff=0.05)
        arr_fake_disc= Arrow(fake_box.get_right(), disc_box.get_left() + UP*0.2, color=RED_B, stroke_width=2, buff=0.05)
        arr_real_disc= Arrow(real_box.get_right(), disc_box.get_left() + DOWN*0.2, color=TEAL_B, stroke_width=2, buff=0.05)
        arr_disc_out = Arrow(disc_box.get_right(), output_box.get_left(), color=ORANGE, stroke_width=2, buff=0.05)

        self.play(
            GrowArrow(arr_tok_gen), GrowArrow(arr_gen_fake),
            GrowArrow(arr_fake_disc), GrowArrow(arr_real_disc),
            GrowArrow(arr_disc_out),
        )
        self.wait(0.5)

        # ── Generator loss: fool discriminator ───────────────────────────────
        self.play(FadeOut(sub))
        loss_sub = Text("Generator loss: make discriminator say 'real' for fake audio", font_size=18, color=GREEN_A)
        loss_sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(loss_sub))

        gen_loss_arrow = Arrow(
            output_box.get_bottom(),
            gen_box.get_bottom() + DOWN * 0.5,
            color=GREEN_A,
            stroke_width=2,
            path_arc=-PI/3,
        )
        gen_loss_lbl = Text("L_adv (fool disc)", font_size=13, color=GREEN_A)
        gen_loss_lbl.next_to(gen_loss_arrow, DOWN, buff=0.1)
        self.play(Create(gen_loss_arrow), Write(gen_loss_lbl))
        self.wait(0.7)

        # ── Discriminator loss ────────────────────────────────────────────────
        self.play(FadeOut(loss_sub))
        disc_sub = Text("Discriminator loss: correctly label real vs fake", font_size=18, color=ORANGE)
        disc_sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(disc_sub))
        self.wait(0.7)

        # ── Multi-scale discriminators ────────────────────────────────────────
        self.play(
            FadeOut(disc_sub), FadeOut(gen_loss_arrow), FadeOut(gen_loss_lbl),
            FadeOut(tokens_box), FadeOut(gen_box), FadeOut(fake_box),
            FadeOut(real_box), FadeOut(disc_box), FadeOut(output_box),
            FadeOut(arr_tok_gen), FadeOut(arr_gen_fake),
            FadeOut(arr_fake_disc), FadeOut(arr_real_disc), FadeOut(arr_disc_out),
        )

        msd_title = Text("Multi-Scale Discriminator (MSD) + Multi-Period (MPD)", font_size=22, color=ORANGE)
        msd_title.next_to(title, DOWN, buff=0.2)
        self.play(Write(msd_title))

        scales = ["Original\n(full res)", "Avg pool x2\n(half res)", "Avg pool x4\n(quarter res)"]
        scale_colors = [RED_B, ORANGE, YELLOW_B]
        scale_boxes = VGroup()
        for s, c in zip(scales, scale_colors):
            b = make_box(s, c, w=2.4, h=1.1)
            scale_boxes.add(b)
        scale_boxes.arrange(RIGHT, buff=0.5).shift(UP * 0.3)

        msd_label = Text("MSD: same audio analyzed at 3 different time resolutions", font_size=16, color=GREY_A)
        msd_label.next_to(scale_boxes, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in scale_boxes], lag_ratio=0.2), Write(msd_label))

        mpd_label = Text(
            "MPD: audio reshaped into 2D grids at periods [2,3,5,7,11]\n"
            "-> captures periodic structure of speech harmonics",
            font_size=15,
            color=GREY_A,
            line_spacing=1.3,
        )
        mpd_label.to_edge(DOWN, buff=0.4)
        self.play(Write(mpd_label))

        insight = Text(
            "Together: generator must fool 8 discriminators simultaneously\n"
            "-> forces perceptually sharp output at all time scales",
            font_size=16,
            color=GREEN_A,
            line_spacing=1.3,
        )
        insight.to_edge(DOWN, buff=0.2)
        self.play(FadeOut(mpd_label), Write(insight))
        self.wait(2.5)
