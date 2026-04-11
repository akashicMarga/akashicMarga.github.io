"""
Animation: Mimi vs SNAC — Flat Structure & Streaming
Shows:
1. Mimi's flat token structure (all 8 levels at same 12.5fps)
2. Why flat = streamable: frames decode one by one
3. CNN + Transformer encoder: local + global context
4. Larger codebook (2048) compensates for fewer frames
5. Side-by-side comparison with SNAC's multi-rate structure
"""

from manim import *
import numpy as np


class MimiStreaming(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("Mimi: Flat RVQ + Streaming", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── PART 1: Side-by-side token grids ─────────────────────────────────
        sub = Text("Why Mimi streams but SNAC cannot", font_size=20, color=GREY_A)
        sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub))

        # ── SNAC side (left) ──────────────────────────────────────────────────
        snac_title = Text("SNAC  (multi-rate)", font_size=18, color=RED_B)
        snac_title.shift(LEFT * 3.8 + UP * 1.8)
        self.play(Write(snac_title))

        snac_rows = [
            ("Coarse 12.5fps", 3, RED_B),
            ("Mid    25fps",   6, ORANGE),
            ("Fine   50fps",  12, YELLOW_B),
        ]
        snac_groups = VGroup()
        for ri, (label, count, color) in enumerate(snac_rows):
            row = VGroup()
            for j in range(count):
                box = Rectangle(width=4.5/count - 0.06, height=0.42,
                                 color=color, fill_color=color, fill_opacity=0.22, stroke_width=1.5)
                row.add(box)
            row.arrange(RIGHT, buff=0.06)
            row.move_to(LEFT * 3.8 + UP * (0.9 - ri * 0.6))
            lbl = Text(label, font_size=11, color=color)
            lbl.next_to(row, LEFT, buff=0.1)
            snac_groups.add(VGroup(row, lbl))

        self.play(LaggedStart(*[FadeIn(g) for g in snac_groups], lag_ratio=0.2))

        # ── Mimi side (right) ─────────────────────────────────────────────────
        mimi_title = Text("Mimi  (flat, all 12.5fps)", font_size=18, color=BLUE_B)
        mimi_title.shift(RIGHT * 3.0 + UP * 1.8)
        self.play(Write(mimi_title))

        mimi_colors = [BLUE_B, TEAL_B, GREEN_B, YELLOW_B, ORANGE, RED_B, PINK, PURPLE_B]
        mimi_rows = VGroup()
        n_frames = 3
        for li in range(8):
            row = VGroup()
            for j in range(n_frames):
                box = Rectangle(width=1.3, height=0.38,
                                 color=mimi_colors[li], fill_color=mimi_colors[li],
                                 fill_opacity=0.22, stroke_width=1.5)
                num = Text(f"k{li+1}", font_size=10, color=mimi_colors[li])
                num.move_to(box.get_center())
                row.add(VGroup(box, num))
            row.arrange(RIGHT, buff=0.08)
            row.move_to(RIGHT * 3.0 + UP * (1.35 - li * 0.38))
            lbl = Text(f"L{li+1}", font_size=10, color=mimi_colors[li])
            lbl.next_to(row, LEFT, buff=0.08)
            mimi_rows.add(VGroup(row, lbl))

        self.play(LaggedStart(*[FadeIn(r) for r in mimi_rows], lag_ratio=0.05), run_time=1)

        # Bracket: "1 frame = 8 tokens"
        frame_col = VGroup(*[mimi_rows[li][0][0] for li in range(8)])  # first column, box part
        brace = Brace(VGroup(*[mimi_rows[li][0][0] for li in range(8)]), RIGHT, color=WHITE)
        brace_lbl = Text("1 frame\n= 8 tokens\n= decode now!", font_size=12, color=WHITE)
        brace_lbl.next_to(brace, RIGHT, buff=0.1)
        self.play(GrowFromCenter(brace), Write(brace_lbl))
        self.wait(0.6)

        # ── SNAC problem: must wait for all 7 ────────────────────────────────
        wait_box = SurroundingRectangle(VGroup(*[snac_groups[li] for li in range(3)]),
                                         color=RED, stroke_width=2, buff=0.1)
        wait_lbl = Text("Wait for 1+2+4=7 tokens\nbefore decoding 80ms", font_size=13, color=RED)
        wait_lbl.next_to(wait_box, DOWN, buff=0.1)
        self.play(Create(wait_box), Write(wait_lbl))
        self.wait(0.8)

        # ── PART 2: Streaming animation ───────────────────────────────────────
        self.play(
            FadeOut(sub), FadeOut(snac_groups), FadeOut(snac_title),
            FadeOut(wait_box), FadeOut(wait_lbl),
            FadeOut(brace), FadeOut(brace_lbl),
            FadeOut(mimi_rows), FadeOut(mimi_title),
        )

        stream_sub = Text("Mimi streaming: each frame decoded as tokens arrive", font_size=20, color=BLUE_A)
        stream_sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(stream_sub))

        # Timeline
        n_stream_frames = 6
        frame_w = 1.4
        frame_gap = 0.15
        timeline_y = 0.5

        frame_groups = VGroup()
        for i in range(n_stream_frames):
            col = VGroup()
            for li in range(8):
                box = Rectangle(width=frame_w, height=0.3,
                                 color=mimi_colors[li], fill_color=mimi_colors[li],
                                 fill_opacity=0.2, stroke_width=1)
                col.add(box)
            col.arrange(DOWN, buff=0.04)
            col.move_to(LEFT * 3.5 + RIGHT * i * (frame_w + frame_gap) + UP * timeline_y)
            frame_lbl = Text(f"F{i+1}", font_size=12, color=GREY_A)
            frame_lbl.next_to(col, UP, buff=0.08)
            frame_groups.add(VGroup(col, frame_lbl))

        # Audio output blocks below
        audio_blocks = VGroup()
        for i in range(n_stream_frames):
            blk = Rectangle(width=frame_w, height=0.5,
                              color=GREEN_B, fill_color=GREEN_B, fill_opacity=0.15, stroke_width=1.5)
            lbl = Text("audio\n80ms", font_size=10, color=GREEN_B)
            lbl.move_to(blk.get_center())
            grp = VGroup(blk, lbl)
            grp.move_to(LEFT * 3.5 + RIGHT * i * (frame_w + frame_gap) + DOWN * 1.2)
            audio_blocks.add(grp)

        decode_arrows = VGroup()
        for i in range(n_stream_frames):
            arr = Arrow(frame_groups[i].get_bottom(), audio_blocks[i].get_top(),
                        color=GREEN_A, stroke_width=1.5, buff=0.05)
            decode_arrows.add(arr)

        # Animate frame by frame
        for i in range(n_stream_frames):
            self.play(FadeIn(frame_groups[i]), run_time=0.25)
            self.play(GrowArrow(decode_arrows[i]), FadeIn(audio_blocks[i]), run_time=0.3)

        stream_note = Text(
            "Each frame's 8 tokens arrive together → immediate decode → 80ms of audio\n"
            "No waiting. Latency = 1 frame = 80ms at 12.5fps",
            font_size=15, color=GREY_A, line_spacing=1.3,
        )
        stream_note.to_edge(DOWN, buff=0.35)
        self.play(Write(stream_note))
        self.wait(0.7)

        # ── PART 3: CNN + Transformer encoder ────────────────────────────────
        self.play(
            FadeOut(stream_sub), FadeOut(frame_groups), FadeOut(decode_arrows),
            FadeOut(audio_blocks), FadeOut(stream_note),
        )

        enc_sub = Text("Why quality is high: CNN + Transformer encoder", font_size=20, color=GREEN_A)
        enc_sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(enc_sub))

        def make_box(lbl, color, w=2.0, h=0.9):
            b = Rectangle(width=w, height=h, color=color, fill_opacity=0.18)
            t = Text(lbl, font_size=14, color=color)
            t.move_to(b.get_center())
            return VGroup(b, t)

        audio_in  = make_box("Raw audio\n24kHz", GREY_A, w=1.8)
        cnn_block = make_box("CNN\nDownsampler\n(strided convs)", BLUE_B, w=2.2)
        info1 = Text("Local features\nreceptive field ~40ms", font_size=12, color=BLUE_C)
        xfmr_block= make_box("Transformer\n(self-attention)", TEAL_B, w=2.2)
        info2 = Text("Global context\nfull sequence", font_size=12, color=TEAL_C)
        vq_block  = make_box("RVQ\n8 levels\nN=2048", ORANGE, w=1.8)

        pipeline = VGroup(audio_in, cnn_block, xfmr_block, vq_block)
        pipeline.arrange(RIGHT, buff=0.5).shift(DOWN * 0.2)

        info1.next_to(cnn_block, DOWN, buff=0.15)
        info2.next_to(xfmr_block, DOWN, buff=0.15)

        arrows = VGroup()
        for i in range(len(pipeline) - 1):
            arr = Arrow(pipeline[i].get_right(), pipeline[i+1].get_left(),
                        color=WHITE, stroke_width=2, buff=0.05)
            arrows.add(arr)

        self.play(FadeIn(pipeline), FadeIn(arrows), Write(info1), Write(info2))

        # Highlight 2048 codebook
        cb_note = Text(
            "Codebook N=2048 (vs 1024 in EnCodec)\n"
            "Larger codebook compensates for fewer frames (12.5fps vs 75fps)\n"
            "Each entry covers more acoustic space",
            font_size=15, color=ORANGE, line_spacing=1.3,
        )
        cb_note.to_edge(DOWN, buff=0.3)
        self.play(Write(cb_note))
        self.wait(2.5)
