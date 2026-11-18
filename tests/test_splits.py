"""Check fixed membership and deterministic split construction."""

import unittest

from sparse_steering.tasks.splits import apply_frozen, build, ids_only, training_seeds


class SplitsTest(unittest.TestCase):

    def test_pair_text_cannot_cross_partitions(self):
        rows = [
            dict(
                id=f"p:{i}", question=f"rewrite {i}", correct=f"neutral {i}", incorrect=f"toxic {i}"
            )
            for i in range(10)
        ]
        rows.append(
            dict(id="duplicate", question="rewrite new", correct="neutral 0", incorrect="toxic new")
        )
        parts = build(rows, "paradetox", 1, counts=dict(train=4, test=4))
        texts = [r[k] for group in parts.values() for r in group for k in ("correct", "incorrect")]
        self.assertEqual(len(texts), len(set(texts)))

    def test_frozen_order_and_seed_identity(self):
        rows = [dict(id=f"q:{i}", question=f"question {i}", correct="yes") for i in range(30)]
        a = build(rows, "halueval", 1, counts=dict(train=8, test=8))
        b = build(rows[::-1], "halueval", 1, counts=dict(train=8, test=8))
        c = build(rows, "halueval", 2, counts=dict(train=8, test=8))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(apply_frozen(rows[::-1], dict(sets=ids_only(a))), a)

    def test_reference_training_rng_formulas(self):
        self.assertEqual(
            training_seeds(0, "truthfulqa", 2, 1),
            dict(initialization=20261011, data_order=20261030),
        )
        self.assertEqual(
            training_seeds(3, "paradetox", 2, 1), dict(initialization=3002, data_order=3021)
        )


if __name__ == "__main__":
    unittest.main()
