"""Check quality boundaries and deterministic selection."""

import unittest

from sparse_steering.evaluation.text_quality import best_feasible, select


class TextQualityTests(unittest.TestCase):

    def rows(self):
        return [
            dict(
                edit_norm_ratio_squared=0.0,
                score=0.5,
                actual_squared_edit_norm=0.0,
                completion_nll=1.0,
                repeated_4gram=0.1,
            ),
            dict(
                edit_norm_ratio_squared=0.01,
                score=0.8,
                actual_squared_edit_norm=0.01,
                completion_nll=1.15,
                repeated_4gram=0.5,
            ),
            dict(
                edit_norm_ratio_squared=0.04,
                score=0.9,
                actual_squared_edit_norm=0.04,
                completion_nll=1.2,
                repeated_4gram=0.1,
            ),
        ]

    def test_nll_and_repetition_boundaries(self):
        self.assertEqual(select(self.rows())["score"], 0.8)
        rows = self.rows()
        rows[1]["repeated_4gram"] = 0.50001
        self.assertEqual(select(rows)["edit_norm_ratio_squared"], 0.0)

    def test_no_infeasible_fallback(self):
        rows = self.rows()
        for row in rows:
            row["repeated_4gram"] = 0.9
        self.assertFalse(select(rows)["available"])
        self.assertIsNone(best_feasible([dict(row, feasible=False) for row in rows]))

    def test_ties_prefer_smaller_edit(self):
        rows = self.rows()
        for row in rows:
            row.update(score=0.8, feasible=True)
        self.assertEqual(best_feasible(rows)["edit_norm_ratio_squared"], 0.0)
