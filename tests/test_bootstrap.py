"""Check paired intervals on aligned capability measurements."""

import unittest

from sparse_steering.evaluation.bootstrap import paired


class BootstrapTests(unittest.TestCase):

    def test_identical_items_have_zero_interval(self):
        rows = [dict(id=str(i), accuracy=float(i % 2)) for i in range(8)]
        value = paired(rows, rows, "arc_challenge", seed=17)
        self.assertEqual(value["delta"], 0.0)
        self.assertEqual(value["ci95"], [0.0, 0.0])

    def test_pairing_uses_item_identity(self):
        base = [dict(item_id=str(i), nll_sum=2.0 + i, scored_tokens=2) for i in range(8)]
        values = [dict(row, nll_sum=row["nll_sum"] + 2) for row in base]
        a = paired(values, base, "wikitext", seed=7)
        b = paired(values, base[::-1], "wikitext", seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["delta"], 1.0)
        self.assertEqual(a["ci95"], [1.0, 1.0])
