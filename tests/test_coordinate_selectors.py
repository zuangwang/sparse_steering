"""Scientific counterexamples to fixed-k, signed, and unnormalized selectors."""

import itertools
import math
import unittest

from sparse_steering.baselines.coordinate_selectors.selectors import (
    cheng_masks,
    fits,
    magnitude_mask,
    masked_direction,
)


class SelectorContracts(unittest.TestCase):

    def test_minimum_cardinality_coverage(self):
        values = [4.0, -3.0, 2.0, 1.0, 0.0]
        for threshold in (0.01, 0.5, 0.7, 0.95, 1.0):
            mask = magnitude_mask(values, threshold)
            target = threshold * sum((v * v for v in values))
            self.assertGreaterEqual(sum((v * v for v, m in zip(values, mask) if m)), target)
            for subset in itertools.combinations(values, sum(mask) - 1):
                self.assertLess(sum((v * v for v in subset)), target)

    def test_magnitude_does_not_force_twenty_percent(self):
        self.assertEqual(sum(magnitude_mask([1.0] * 20)), 19)
        self.assertEqual(sum(magnitude_mask([100.0] + [1.0] * 19)), 1)
        self.assertEqual(magnitude_mask([-1.0, 1.0, 1.0], 2 / 3), [True, True, False])

    def test_cheng_normalized_absolute_ratio_and_matched_ie_count(self):
        s, ie = ([1.0, 2.0, 3.0, 4.0], [-10.0, 6.0, 3.0, 4.0])
        grad, effect = cheng_masks(s, ie)
        self.assertEqual(grad, [True, True, False, False])
        self.assertEqual(effect, [True, True, False, False])
        self.assertEqual(cheng_masks([v * 7 for v in s], [v * 0.03 for v in ie]), (grad, effect))
        self.assertEqual(cheng_masks(s, ie, 0)[0], [True] * 4)

    def test_ie_and_gradient_can_select_different_coordinates(self):
        s, ie = ([10.0, 1.0, 1.0], [5.0, -2.0, 1.0])
        grad, effect = cheng_masks(s, ie)
        self.assertEqual(grad, [False, True, True])
        self.assertEqual(effect, [True, True, False])

    def test_mask_preserves_coefficients_and_zeros(self):
        result = fits([4.0, -3.0, 2.0, 0.0], [4.0, -9.0, 1.0, 0.0])
        for fit in result.values():
            self.assertEqual(
                fit["direction"],
                [v if m else 0 for v, m in zip([4.0, -3.0, 2.0, 0.0], fit["mask"])],
            )
            self.assertFalse(fit["vector_renormalized"])
            self.assertFalse(fit["activation_rescaling"])
        self.assertEqual(masked_direction([4.0, 3.0], [True, False]), [4.0, 0.0])

    def test_invalid_or_empty_support_fails_instead_of_falling_back(self):
        for values in ([], [0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]):
            with self.assertRaises(ValueError):
                magnitude_mask(values)
        with self.assertRaises(ValueError):
            cheng_masks([1.0, 1.0], [1.0, 1.0], 2)
        with self.assertRaises(ValueError):
            cheng_masks([1.0, 0.0], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
