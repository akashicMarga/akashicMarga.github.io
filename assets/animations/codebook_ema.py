"""
Animation: Codebook Collapse & EMA Updates
Shows how codebook entries drift toward data clusters via EMA,
and what happens when entries die (collapse) and get reinitialized.
"""

from manim import *
import numpy as np


class CodebookEMA(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        title = Text("Codebook Training: EMA Updates & Collapse", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── Setup: 2D latent space ────────────────────────────────────────────
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=7,
            y_length=7,
            axis_config={"color": GREY_D, "stroke_width": 1},
            tips=False,
        ).shift(DOWN * 0.3)

        self.play(Create(axes), run_time=0.6)

        rng = np.random.default_rng(7)

        # True data clusters (where encoder outputs live)
        cluster_centers = np.array([[-2.0, 2.0], [2.0, 1.5], [-1.5, -2.0], [2.0, -2.0]])
        n_clusters = len(cluster_centers)
        cluster_colors = [BLUE_B, GREEN_B, YELLOW_B, RED_B]

        # Draw data points
        data_dots = VGroup()
        data_assignments = []
        for ci, (cx, cy) in enumerate(cluster_centers):
            for _ in range(12):
                px = cx + rng.normal(scale=0.45)
                py = cy + rng.normal(scale=0.45)
                dot = Dot(axes.c2p(px, py), radius=0.06, color=cluster_colors[ci])
                dot.set_fill(cluster_colors[ci], opacity=0.5)
                data_dots.add(dot)
                data_assignments.append(ci)

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in data_dots], lag_ratio=0.02), run_time=0.8)

        # ── Codebook entries: initially random ───────────────────────────────
        n_entries = 6  # show 6 entries (some will be well-placed, some will die)
        initial_positions = rng.uniform(-3.5, 3.5, (n_entries, 2))
        cb_colors = [RED, ORANGE, YELLOW, GREEN, TEAL, BLUE]

        cb_dots = VGroup()
        cb_labels = VGroup()
        for i, (pos, c) in enumerate(zip(initial_positions, cb_colors)):
            dot = Dot(axes.c2p(pos[0], pos[1]), radius=0.2, color=c)
            dot.set_fill(c, opacity=0.9)
            lbl = Text(f"e{i}", font_size=14, color=c).next_to(dot, UP + RIGHT, buff=0.05)
            cb_dots.add(dot)
            cb_labels.add(lbl)

        init_label = Text("Codebook entries: randomly initialized", font_size=18, color=GREY_A)
        init_label.to_edge(DOWN, buff=0.3)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in cb_dots], lag_ratio=0.1),
            LaggedStart(*[Write(l) for l in cb_labels], lag_ratio=0.1),
            Write(init_label),
            run_time=1,
        )
        self.wait(0.5)

        # ── EMA updates: entries drift toward nearest cluster ─────────────────
        self.play(FadeOut(init_label))
        ema_label = Text("EMA update: each entry moves toward centroid of assigned points", font_size=17, color=GREEN)
        ema_label.to_edge(DOWN, buff=0.3)
        self.play(Write(ema_label))

        # Animate 4 training steps
        # Manually design convergence: 4 entries go to 4 clusters, 2 get no assignments -> collapse
        converged_positions = [
            cluster_centers[0],  # e0 -> cluster 0
            cluster_centers[1],  # e1 -> cluster 1
            cluster_centers[2],  # e2 -> cluster 2
            cluster_centers[3],  # e3 -> cluster 3
            np.array([3.5, 3.5]),  # e4: isolated, no assignments coming
            np.array([-3.5, -3.5]),  # e5: isolated, no assignments coming
        ]

        # Simulate 3 steps of drift
        for step in range(3):
            alpha = (step + 1) / 3
            step_lbl = Text(f"Training step {step+1}", font_size=16, color=GREY_B)
            step_lbl.to_corner(UR, buff=0.5)
            self.play(FadeIn(step_lbl), run_time=0.2)

            anims = []
            for i, (dot, lbl) in enumerate(zip(cb_dots, cb_labels)):
                current = initial_positions[i] + alpha * (converged_positions[i] - initial_positions[i])
                new_pos = axes.c2p(current[0], current[1])
                anims.append(dot.animate.move_to(new_pos))
                anims.append(lbl.animate.next_to(new_pos, UP + RIGHT, buff=0.05))
            self.play(*anims, run_time=0.6)
            self.play(FadeOut(step_lbl), run_time=0.1)

        self.wait(0.4)

        # ── Show collapse: e4 and e5 never get assigned ───────────────────────
        self.play(FadeOut(ema_label))
        collapse_lbl = Text("Codebook collapse: e4 and e5 are never nearest to any point!", font_size=17, color=RED)
        collapse_lbl.to_edge(DOWN, buff=0.4)
        self.play(Write(collapse_lbl))

        # Fade out e4, e5 (they collapse)
        dead_cross_4 = Cross(cb_dots[4], color=RED, stroke_width=4)
        dead_cross_5 = Cross(cb_dots[5], color=RED, stroke_width=4)
        self.play(Create(dead_cross_4), Create(dead_cross_5))
        self.play(
            cb_dots[4].animate.set_opacity(0.15),
            cb_labels[4].animate.set_opacity(0.15),
            cb_dots[5].animate.set_opacity(0.15),
            cb_labels[5].animate.set_opacity(0.15),
        )
        self.wait(0.7)

        # ── Reinitialize dead entries ─────────────────────────────────────────
        self.play(FadeOut(collapse_lbl), FadeOut(dead_cross_4), FadeOut(dead_cross_5))
        reinit_lbl = Text("Fix: reinitialize dead entries to random batch vectors", font_size=17, color=ORANGE)
        reinit_lbl.to_edge(DOWN, buff=0.4)
        self.play(Write(reinit_lbl))

        # Move e4, e5 to random positions near data
        new_pos_4 = axes.c2p(-2.5, 1.2)
        new_pos_5 = axes.c2p(1.2, -1.8)

        self.play(
            cb_dots[4].animate.move_to(new_pos_4).set_opacity(1.0).set_color(TEAL),
            cb_labels[4].animate.move_to(new_pos_4 + UP * 0.25 + RIGHT * 0.25).set_opacity(1.0),
            cb_dots[5].animate.move_to(new_pos_5).set_opacity(1.0).set_color(PURPLE),
            cb_labels[5].animate.move_to(new_pos_5 + UP * 0.25 + RIGHT * 0.25).set_opacity(1.0),
            run_time=0.8,
        )

        restart_lbl = Text("Now they can start learning again!", font_size=16, color=GREEN)
        restart_lbl.next_to(reinit_lbl, UP, buff=0.1)
        self.play(Write(restart_lbl))
        self.wait(2)
