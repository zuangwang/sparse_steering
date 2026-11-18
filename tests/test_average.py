"""Saved-measurement aggregation requires no model or dataset downloads."""

import json
from pathlib import Path
import tempfile
import unittest

from reproduce.average import BENCHMARKS, METHODS, MODELS, SEEDS, TASKS, capability, comparison, coordinates


class AverageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for seed in SEEDS:
            for model in MODELS:
                for task in TASKS:
                    folder = self.root / f"seed{seed}" / model / task
                    folder.mkdir(parents=True)
                    # Opposite endpoints win on alternating seeds. A stronger
                    # infeasible point must never enter the average.
                    points = [dict(score=10 if seed % 2 else 30, feasible=True,
                                   actual_squared_edit_norm=1, edit_norm_ratio_squared=.01),
                              dict(score=30 if seed % 2 else 10, feasible=True,
                                   actual_squared_edit_norm=2, edit_norm_ratio_squared=.04),
                              dict(score=99, feasible=False,
                                   actual_squared_edit_norm=3, edit_norm_ratio_squared=.09)]
                    (folder / "test.json").write_text(json.dumps(dict(
                        unsteered=dict(score=5 + seed), grids={m: points for m in METHODS})))
                    if task == "rtp":
                        continue
                    (folder / "coordinate_statistics.json").write_text(json.dumps(dict(
                        k=2, width=10, selected_squared_coefficient_mass=.1 * (seed + 1),
                        top_k_squared_coefficient_mass=.8,
                        overlaps={k: .2 for k in ("top_k_coefficients", "top_k_ie", "gradient", "ie")})))
                    values = {b: dict(status="complete", paired=dict(delta=(-1) ** seed * (seed + 1) / 100)) for b in BENCHMARKS}
                    (folder / "capability.json").write_text(json.dumps(dict(
                        unsteered={b: dict(summary=dict(accuracy=.5, nll=2)) for b in BENCHMARKS},
                        methods={m: values for m in METHODS})))

    def test_selection_before_averaging(self):
        rows = comparison(self.root)
        self.assertEqual(len(rows), 16 * 9)
        self.assertTrue(all(r["score"] == 30 for r in rows if r["method"] != "unsteered"))
        self.assertTrue(all(r["score"] == 6.5 for r in rows if r["method"] == "unsteered"))

    def test_absolute_changes_units_and_global_signed_range(self):
        # Put the extreme in one case only: the range must use measurements,
        # rather than the per-seed means.
        file = self.root / "seed3" / MODELS[0] / TASKS[0] / "capability.json"
        data = json.loads(file.read_text())
        data["methods"]["ours"]["arc_challenge"]["paired"]["delta"] = -.2
        file.write_text(json.dumps(data))
        rows = {(r["method"], r["benchmark"]): r for r in capability(self.root)}
        row = rows["ours", "arc_challenge"]
        self.assertAlmostEqual(row["mean"], 2.5 + 16 / 48)
        self.assertEqual((row["minimum"], row["maximum"]), (-20, 3))
        self.assertAlmostEqual(rows["dense", "wikitext"]["mean"], .025)
        self.assertEqual(rows["unsteered", "arc_challenge"]["mean"], 50)
        self.assertEqual(rows["unsteered", "wikitext"]["mean"], 2)

    def test_coordinate_units_and_missing_seed(self):
        self.assertAlmostEqual(coordinates(self.root)[0]["selected_squared_coefficient_mass"], 25)
        file = self.root / "seed2" / MODELS[0] / TASKS[0] / "test.json"
        file.unlink()
        with self.assertRaisesRegex(ValueError, "Missing measurement"):
            comparison(self.root)


if __name__ == "__main__":
    unittest.main()
