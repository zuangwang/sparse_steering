"""Check that constrained normalization changes only selected coordinates."""

import math
import unittest


class SupportTests(unittest.TestCase):

    def test_scaled_operator_uses_only_full_learned_support(self):
        import torch

        from sparse_steering.method.constrained_normalization import (
            realize_constrained_normalization,
        )

        torch.manual_seed(947)
        width = 2048
        k = math.ceil(0.2 * width)
        mask = torch.zeros(width, dtype=torch.bool)
        mask[::5][:k] = True
        self.assertEqual(int(mask.sum()), k)
        direction = torch.zeros(width, dtype=torch.float64)
        direction[mask] = torch.randn(k, dtype=torch.float64)
        direction /= direction.norm()
        states = (torch.randn(3, width) * 2).bfloat16()
        updated, diag = realize_constrained_normalization(
            states, direction, torch.zeros_like(mask), 4.0
        )
        self.assertEqual(diag["normalization_support_count"], k)
        self.assertEqual(diag["steering_coordinate_count"], k)
        self.assertTrue(torch.equal(updated[:, ~mask], states[:, ~mask]))
        ideal = diag["beta"][:, None] * (
            states[:, mask].double() + diag["alpha"][:, None] * direction[mask]
        )
        self.assertTrue(torch.equal(updated[:, mask], ideal.bfloat16()))
        self.assertTrue(bool((diag["norm_relative_drift"] <= 0.001).all()))
        self.assertTrue(bool((diag["realized_squared_edit_norm"] <= 4.0).all()))
        self.assertTrue(bool((diag["realized_squared_edit_norm"] > 0).all()))


if __name__ == "__main__":
    unittest.main()
