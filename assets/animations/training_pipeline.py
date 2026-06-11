"""
Animation: Training the Codec LM — Data Pipeline

Shows three stages:
1. Raw audio files (varying lengths) → codec encoder → .codec.npy arrays
2. Naive batching (lots of padding waste) vs length-bucketed batching (tight packing)
3. The dual-channel input format: text channel + codec channel, packed for training
"""

from manim import *
import numpy as np


class TrainingPipeline(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("Training the Codec LM: Data Pipeline", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title))
        self.wait(0.25)

        # ── PART 1: Audio → codec tokens ──────────────────────────────────────
        part1_lbl = Text("① Audio → codec encoder → integer arrays saved offline",
                         font_size=16, color=GREY_B)
        part1_lbl.to_edge(DOWN, buff=1.50)
        self.play(Write(part1_lbl))

        # Audio bars (varying lengths, simulating audio clips)
        audio_lengths = [5.0, 3.2, 7.8, 4.1, 6.5]
        audio_colors  = [BLUE_B, TEAL_B, GREEN_B, ORANGE, RED_B]
        bar_h = 0.38
        bar_gap = 0.22
        bar_origin_y = 1.10

        audio_bars = VGroup()
        for i, (length, color) in enumerate(zip(audio_lengths, audio_colors)):
            bar = Rectangle(width=length * 0.45, height=bar_h,
                            fill_color=color, fill_opacity=0.55,
                            stroke_color=color, stroke_width=1.2)
            bar.move_to([-4.00 + length * 0.45 / 2, bar_origin_y - i * (bar_h + bar_gap), 0])
            lbl = Text(f"{length:.1f}s", font_size=10, color=color)
            lbl.next_to(bar, RIGHT, buff=0.12)
            audio_bars.add(VGroup(bar, lbl))

        audio_header = Text("raw audio (wav)", font_size=13, color=GREY_A)
        audio_header.move_to([-3.20, bar_origin_y + 0.52, 0])
        self.play(FadeIn(audio_header))
        self.play(LaggedStart(*[FadeIn(b) for b in audio_bars], lag_ratio=0.12), run_time=0.80)

        # Codec encoder arrow
        enc_arrow = Arrow([-1.20, bar_origin_y - 1.00, 0], [0.40, bar_origin_y - 1.00, 0],
                          color=WHITE, stroke_width=2.0, buff=0.08)
        enc_box = RoundedRectangle(corner_radius=0.07, width=2.20, height=0.52,
                                   fill_color=GREY_E, fill_opacity=0.50,
                                   stroke_color=GREY_C, stroke_width=1.3)
        enc_box.move_to([-0.40, bar_origin_y - 1.00, 0])
        enc_lbl = Text("codec encoder\n(run once, offline)", font_size=11, color=GREY_A,
                       line_spacing=1.2)
        enc_lbl.move_to(enc_box.get_center())
        self.play(GrowArrow(enc_arrow), FadeIn(enc_box), FadeIn(enc_lbl))

        # .codec.npy arrays on the right
        npy_boxes = VGroup()
        npy_origin_x = 2.20
        for i, (length, color) in enumerate(zip(audio_lengths, audio_colors)):
            n_frames = int(length * 12.5)
            npy = Rectangle(width=length * 0.28, height=bar_h * 0.80,
                            fill_color=color, fill_opacity=0.35,
                            stroke_color=color, stroke_width=1.0)
            npy.move_to([npy_origin_x + length * 0.14, bar_origin_y - i * (bar_h + bar_gap), 0])
            nlbl = Text(f"[{n_frames}] int32", font_size=9, color=color)
            nlbl.next_to(npy, RIGHT, buff=0.08)
            npy_boxes.add(VGroup(npy, nlbl))

        npy_header = Text(".codec.npy", font_size=13, color=GREY_A)
        npy_header.move_to([npy_origin_x + 1.40, bar_origin_y + 0.52, 0])

        out_arrow = Arrow([0.55, bar_origin_y - 1.00, 0], [1.70, bar_origin_y - 1.00, 0],
                          color=WHITE, stroke_width=2.0, buff=0.08)
        self.play(GrowArrow(out_arrow), FadeIn(npy_header))
        self.play(LaggedStart(*[FadeIn(b) for b in npy_boxes], lag_ratio=0.10), run_time=0.70)
        self.wait(0.60)

        # ── PART 2: Length bucketing ───────────────────────────────────────────
        self.play(FadeOut(part1_lbl))
        part2_lbl = Text("② Length bucketing: group similar lengths, minimize padding",
                         font_size=16, color=GREY_B)
        part2_lbl.to_edge(DOWN, buff=1.50)
        self.play(Write(part2_lbl))

        self.play(
            FadeOut(audio_bars), FadeOut(audio_header),
            FadeOut(enc_arrow), FadeOut(enc_box), FadeOut(enc_lbl),
            FadeOut(out_arrow), FadeOut(npy_header), FadeOut(npy_boxes),
        )

        # Without bucketing: random order → lots of padding
        wo_title = Text("Without bucketing", font_size=15, color=RED_C)
        wo_title.move_to([-3.20, 1.50, 0])
        self.play(Write(wo_title))

        unsorted_lengths = [5.0, 3.2, 7.8, 4.1]
        batch_h = 0.36
        batch_y_start = 0.90
        wo_bars = VGroup()
        max_len = max(unsorted_lengths)
        for i, (length, color) in enumerate(zip(unsorted_lengths, audio_colors)):
            # actual data
            data_bar = Rectangle(width=length * 0.45, height=batch_h,
                                 fill_color=color, fill_opacity=0.55,
                                 stroke_color=color, stroke_width=1.2)
            # padding
            pad_w = (max_len - length) * 0.45
            if pad_w > 0:
                pad_bar = Rectangle(width=pad_w, height=batch_h,
                                    fill_color=GREY_E, fill_opacity=0.30,
                                    stroke_color=GREY_D, stroke_width=0.8)
                pad_bar.next_to(data_bar, RIGHT, buff=0)
                pad_lbl = Text("PAD", font_size=8, color=GREY_D)
                pad_lbl.move_to(pad_bar.get_center())
                grp = VGroup(data_bar, pad_bar, pad_lbl)
            else:
                grp = VGroup(data_bar)
            grp.move_to([-4.50 + grp.width / 2, batch_y_start - i * (batch_h + 0.14), 0])
            wo_bars.add(grp)

        self.play(LaggedStart(*[FadeIn(b) for b in wo_bars], lag_ratio=0.10), run_time=0.70)
        waste_pct = Text("~35% wasted compute (padding)", font_size=12, color=RED_C)
        waste_pct.next_to(wo_bars, DOWN, buff=0.16)
        self.play(Write(waste_pct))
        self.wait(0.50)

        # With bucketing: sorted → minimal padding
        w_title = Text("With bucketing (sort by length)", font_size=15, color=GREEN_C)
        w_title.move_to([2.50, 1.50, 0])
        self.play(Write(w_title))

        sorted_lengths = sorted(unsorted_lengths)
        sorted_colors  = [audio_colors[unsorted_lengths.index(l)] for l in sorted_lengths]
        w_bars = VGroup()
        s_max = max(sorted_lengths)
        for i, (length, color) in enumerate(zip(sorted_lengths, sorted_colors)):
            data_bar = Rectangle(width=length * 0.45, height=batch_h,
                                 fill_color=color, fill_opacity=0.55,
                                 stroke_color=color, stroke_width=1.2)
            pad_w = (s_max - length) * 0.45
            if pad_w > 0.05:
                pad_bar = Rectangle(width=pad_w, height=batch_h,
                                    fill_color=GREY_E, fill_opacity=0.25,
                                    stroke_color=GREY_D, stroke_width=0.7)
                pad_bar.next_to(data_bar, RIGHT, buff=0)
                pad_lbl = Text("PAD", font_size=8, color=GREY_D)
                pad_lbl.move_to(pad_bar.get_center())
                grp = VGroup(data_bar, pad_bar, pad_lbl)
            else:
                grp = VGroup(data_bar)
            grp.move_to([0.50 + grp.width / 2, batch_y_start - i * (batch_h + 0.14), 0])
            w_bars.add(grp)

        self.play(LaggedStart(*[FadeIn(b) for b in w_bars], lag_ratio=0.10), run_time=0.70)
        efficient_pct = Text("~8% padding — 3–5× throughput gain", font_size=12, color=GREEN_C)
        efficient_pct.next_to(w_bars, DOWN, buff=0.16)
        self.play(Write(efficient_pct))
        self.wait(0.60)

        # ── PART 3: Dual-channel input format ─────────────────────────────────
        self.play(
            FadeOut(part2_lbl), FadeOut(wo_title), FadeOut(wo_bars), FadeOut(waste_pct),
            FadeOut(w_title), FadeOut(w_bars), FadeOut(efficient_pct),
        )
        part3_lbl = Text("③ Dual-channel input: text tokens stacked with codec tokens",
                         font_size=16, color=GREY_B)
        part3_lbl.to_edge(DOWN, buff=1.50)
        self.play(Write(part3_lbl))

        def channel_row(tokens, color, label, y):
            row = VGroup()
            for i, tok in enumerate(tokens):
                w = 0.55 if len(tok) <= 3 else 0.72
                r = RoundedRectangle(corner_radius=0.05, width=w, height=0.42,
                                     fill_color=color, fill_opacity=0.55 if tok != "PAD" else 0.15,
                                     stroke_color=color if tok != "PAD" else GREY_D,
                                     stroke_width=1.2 if tok != "PAD" else 0.8)
                t = Text(tok, font_size=9, color=WHITE if tok != "PAD" else GREY_D)
                r.move_to([-4.80 + i * 0.80, y, 0])
                t.move_to(r.get_center())
                row.add(VGroup(r, t))
            lbl = Text(label, font_size=12, color=color)
            lbl.move_to([-5.80, y, 0])
            return VGroup(row, lbl)

        text_tokens  = ["PAD", "PAD", "PAD", "BOS", "हे", "ल्लो", "EOS", "PAD", "PAD"]
        codec_tokens = ["PAD", "PAD", "PAD", "BOS", "k₁", "k₁", "k₁", "k₁", "EOS"]

        text_row  = channel_row(text_tokens,  BLUE_B,  "text",  0.60)
        codec_row = channel_row(codec_tokens, RED_B,   "codec", -0.20)

        self.play(FadeIn(text_row), FadeIn(codec_row))

        # Loss mask: only codec positions after BOS count
        loss_lbl = Text("loss computed on codec tokens only (after BOS)",
                        font_size=12, color=GREEN_B)
        loss_lbl.to_edge(DOWN, buff=0.55)
        # Highlight the 4 codec tokens that count
        loss_boxes = VGroup()
        for i in range(4, 8):
            lb = SurroundingRectangle(codec_row[0][i], color=GREEN_B, buff=0.04,
                                      stroke_width=1.8)
            loss_boxes.add(lb)
        self.play(Write(loss_lbl))
        self.play(Create(loss_boxes), run_time=0.50)
        self.wait(2.80)
