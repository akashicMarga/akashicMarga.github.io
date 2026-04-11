"""
Animation: Source-Filter Model of Speech
Shows how the glottis (source) produces a harmonic pulse train at F0,
and the vocal tract (filter) shapes it into vowels via formants.
"""

from manim import *
import numpy as np


class SourceFilterModel(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("The Source-Filter Model of Speech", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ── Labels ─────────────────────────────────────────────────────────────
        sub = Text("Speech = Glottis (source) x Vocal Tract (filter)", font_size=20, color=GREY_A)
        sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub))
        self.wait(0.4)

        # ── Three panel layout ─────────────────────────────────────────────────
        # Panel 1: Source (harmonic pulse train)
        # Panel 2: Filter (vocal tract frequency response)
        # Panel 3: Output (vowel spectrum)

        panel_width = 3.5
        panel_height = 2.5
        gap = 0.6

        def make_panel(label_str, color):
            box = Rectangle(width=panel_width, height=panel_height, color=color, fill_opacity=0.08, stroke_width=1.5)
            label = Text(label_str, font_size=18, color=color)
            label.next_to(box, UP, buff=0.15)
            return VGroup(box, label)

        source_panel = make_panel("SOURCE\n(Glottis / F0)", BLUE_B).shift(LEFT * 4.2 + DOWN * 0.6)
        filter_panel = make_panel("FILTER\n(Vocal Tract)", GREEN_B).shift(DOWN * 0.6)
        output_panel = make_panel("OUTPUT\n(Vowel Spectrum)", YELLOW_B).shift(RIGHT * 4.2 + DOWN * 0.6)

        self.play(
            FadeIn(source_panel),
            FadeIn(filter_panel),
            FadeIn(output_panel),
            run_time=0.8,
        )

        # ── Source: harmonic series ────────────────────────────────────────────
        # Vertical bars at F0, 2F0, 3F0, ...
        F0 = 120  # Hz, male voice
        n_harmonics = 8
        source_axes = Axes(
            x_range=[0, 1200, 200],
            y_range=[0, 1.3, 0.5],
            x_length=panel_width - 0.4,
            y_length=panel_height - 0.6,
            axis_config={"color": GREY_D, "stroke_width": 1},
            tips=False,
        ).move_to(source_panel[0].get_center())

        harmonics = VGroup()
        for n in range(1, n_harmonics + 1):
            freq = n * F0
            if freq <= 1100:
                bar = Line(
                    source_axes.c2p(freq, 0),
                    source_axes.c2p(freq, 1.0 / n ** 0.5),
                    color=BLUE_B,
                    stroke_width=3,
                )
                harmonics.add(bar)

        f0_label = Text(f"F0={F0}Hz", font_size=13, color=BLUE_C)
        f0_label.next_to(source_axes.c2p(F0, 1.0), UP, buff=0.05)

        self.play(Create(source_axes), run_time=0.4)
        self.play(
            LaggedStart(*[GrowFromEdge(h, DOWN) for h in harmonics], lag_ratio=0.1),
            run_time=1.0,
        )
        self.play(Write(f0_label))

        # ── Filter: resonance peaks (formants) ────────────────────────────────
        filter_axes = Axes(
            x_range=[0, 1200, 200],
            y_range=[0, 1.3, 0.5],
            x_length=panel_width - 0.4,
            y_length=panel_height - 0.6,
            axis_config={"color": GREY_D, "stroke_width": 1},
            tips=False,
        ).move_to(filter_panel[0].get_center())

        x_freq = np.linspace(0, 1200, 400)
        # /a/ vowel formants: F1~800, F2~1200 (simplified)
        formant_response = (
            0.9 * np.exp(-((x_freq - 300) ** 2) / (2 * 80 ** 2))
            + 0.75 * np.exp(-((x_freq - 850) ** 2) / (2 * 100 ** 2))
            + 0.5 * np.exp(-((x_freq - 1100) ** 2) / (2 * 80 ** 2))
        )
        formant_response = np.clip(formant_response, 0, 1.2)

        filter_curve = filter_axes.plot_line_graph(
            x_values=x_freq,
            y_values=formant_response,
            line_color=GREEN_B,
            stroke_width=2.5,
            add_vertex_dots=False,
        )

        f1_lbl = Text("F1", font_size=13, color=GREEN_C).move_to(filter_axes.c2p(300, 1.05))
        f2_lbl = Text("F2", font_size=13, color=GREEN_C).move_to(filter_axes.c2p(850, 1.05))
        f3_lbl = Text("F3", font_size=13, color=GREEN_C).move_to(filter_axes.c2p(1100, 0.65))

        self.play(Create(filter_axes), run_time=0.4)
        self.play(Create(filter_curve), run_time=0.8)
        self.play(Write(f1_lbl), Write(f2_lbl), Write(f3_lbl))

        # ── Arrows: source * filter -> output ────────────────────────────────
        arr1 = Arrow(source_panel[0].get_right(), filter_panel[0].get_left(), color=WHITE, stroke_width=2.5, buff=0.05)
        arr1_lbl = Text("x", font_size=24, color=WHITE).next_to(arr1, UP, buff=0.05)
        arr2 = Arrow(filter_panel[0].get_right(), output_panel[0].get_left(), color=WHITE, stroke_width=2.5, buff=0.05)
        arr2_lbl = Text("=", font_size=24, color=WHITE).next_to(arr2, UP, buff=0.05)

        self.play(GrowArrow(arr1), Write(arr1_lbl), GrowArrow(arr2), Write(arr2_lbl))

        # ── Output: harmonics shaped by formant envelope ──────────────────────
        output_axes = Axes(
            x_range=[0, 1200, 200],
            y_range=[0, 1.3, 0.5],
            x_length=panel_width - 0.4,
            y_length=panel_height - 0.6,
            axis_config={"color": GREY_D, "stroke_width": 1},
            tips=False,
        ).move_to(output_panel[0].get_center())

        output_bars = VGroup()
        for n in range(1, n_harmonics + 1):
            freq = n * F0
            if freq <= 1100:
                # magnitude = source amplitude * formant envelope at this freq
                src_amp = 1.0 / n ** 0.5
                filt_idx = int(freq / 1200 * 399)
                filt_amp = formant_response[filt_idx]
                final_amp = src_amp * filt_amp
                bar = Line(
                    output_axes.c2p(freq, 0),
                    output_axes.c2p(freq, final_amp),
                    color=YELLOW_B,
                    stroke_width=3,
                )
                output_bars.add(bar)

        self.play(Create(output_axes), run_time=0.4)
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in output_bars], lag_ratio=0.1),
            run_time=1.0,
        )
        self.wait(0.5)

        # ── Key insight box ───────────────────────────────────────────────────
        insight = Text(
            "Change F0  ->  different pitch, same vowel\n"
            "Change formants  ->  different vowel, same pitch\n"
            "Change both  ->  voice conversion",
            font_size=16,
            color=GREY_A,
            line_spacing=1.3,
        )
        insight.to_edge(DOWN, buff=0.25)
        self.play(Write(insight), run_time=1)
        self.wait(2.5)
