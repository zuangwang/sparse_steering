"""CPU checks for missing measurements and four-seed mask trajectories."""

import unittest

from sparse_steering.analysis.paper_figures import (
    average_curve,
    average_trajectories,
    curve_points,
)


class PaperAveragingTest(unittest.TestCase):
    def test_missing_seed_point_and_nonfinite_alpha(self):
        curves = [
            {0: (10, 1, 0), 0.04: (20, 2, 4)},
            {0: (30, 1, 0)},
            {0: (50, 1, 0), 0.04: (40, 4, None)},
            {0: (70, 1, 0), 0.04: (60, 6, float("nan"))},
        ]
        baseline, point = average_curve(curves)
        self.assertEqual(baseline["score"], 40)
        self.assertEqual(point, dict(budget=0.04, n=3, score=40, nll_ratio=4, alpha=4))

    def test_curve_filters_and_unsteered_origin(self):
        def point(budget, score, **extra):
            return dict(
                edit_norm_ratio_squared=budget, score=score, nll_ratio=2, **extra
            )

        data = dict(
            unsteered=dict(score=12),
            grids=dict(
                ours=[
                    point(0.000025, 100),
                    point(0.04, 30),
                    point(0.09, float("nan")),
                    point(0.16, 80, available=False),
                    point(0.36, 70, status="failed"),
                ]
            ),
        )
        self.assertEqual(
            curve_points(data, "ours"), {0: (12, 1, 0), 0.04: (30, 2, None)}
        )

    def test_step_zero_dropped_before_seed_mean(self):
        def row(step, value, rerun=0):
            return dict(
                model="gemma",
                dataset="truthfulqa",
                rerun=rerun,
                step=step,
                overlap_final=value,
            )

        trajectory = average_trajectories(
            [
                [row(0, 0.9), row(1, 0.2), row(96, 1)],
                [row(1, 0.4), row(96, 1)],
                [row(1, 0.6), row(96, 1)],
                [row(1, 0.8), row(96, 1)],
            ]
        )
        self.assertEqual([r["step"] for r in trajectory], [1, 96])
        self.assertEqual([r["overlap_final"] for r in trajectory], [0.5, 1])
        self.assertEqual(trajectory[0]["restart"], 0)


if __name__ == "__main__":
    unittest.main()
