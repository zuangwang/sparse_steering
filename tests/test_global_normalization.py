"""Check global normalization with synthetic activation tensors."""

import unittest
from types import SimpleNamespace

import torch

from sparse_steering.method.constrained_normalization import realize_constrained_normalization
from sparse_steering.runtime.hooks import install_sparse as install


class GlobalRescaleTests(unittest.TestCase):

    def test_closed_form_squared_edit_norm_norm_and_changed_coordinates(self):
        torch.manual_seed(106)
        torch.set_num_threads(2)
        states = torch.randn(8, 1024).bfloat16()
        direction = torch.zeros(1024, dtype=torch.float64)
        direction[:64] = torch.randn(64, dtype=torch.float64)
        unit = direction / direction.norm()
        for edit_norm_ratio_squared in (0.04, 0.16, 0.36):
            squared_edit_norm_limits = edit_norm_ratio_squared * states.double().square().sum(-1)
            out, diag = realize_constrained_normalization(
                states, direction, ~(direction != 0), squared_edit_norm_limits
            )
            expected = (
                diag["beta"][:, None] * (states.double() + diag["alpha"][:, None] * unit)
            ).bfloat16()
            self.assertTrue(torch.equal(out, expected))
            self.assertTrue(bool((diag["changed_coordinates"] > 8 * 64).all()))
            self.assertTrue(bool((diag["norm_relative_drift"] <= 0.001).all()))
            self.assertTrue(
                bool((diag["realized_squared_edit_norm"] <= squared_edit_norm_limits).all())
            )
            self.assertTrue(bool((diag["planned_chord_cosine"] >= 0.9).all()))
            self.assertTrue(
                bool((diag["target_relative_error"][~diag["capacity_limited"]] <= 0.01).all())
            )
            self.assertEqual(diag["normalization_support_count"], states.shape[1])

    def test_capacity_limit_and_zero_edit_norm_ratio_squared_are_explicit(self):
        torch.manual_seed(10)
        states = torch.randn(2, 512).bfloat16()
        direction = torch.zeros(512, dtype=torch.float64)
        direction[:32] = torch.randn(32, dtype=torch.float64)
        out, diag = realize_constrained_normalization(states, direction, direction == 0, 0.0)
        self.assertTrue(torch.equal(out, states))
        self.assertTrue(bool(diag["no_op"].all()))
        out, diag = realize_constrained_normalization(states, direction, direction == 0, 1000000.0)
        self.assertTrue(bool(diag["capacity_limited"].all()))
        self.assertTrue(bool((diag["realized_squared_edit_norm"] <= 1000000.0).all()))
        self.assertTrue(bool((diag["norm_relative_drift"] <= 0.001).all()))

    def test_inference_hook_at_all_32_positions_and_tuple_output(self):
        torch.manual_seed(104)
        runtime = SimpleNamespace(
            cell=dict(
                method="sparsification_global_norm",
                edit_norm_ratio_squared=0.04,
                continuation_squared_norm_multiplier=1.0,
            ),
            scale=16.0,
            _prediction_step=0,
            _contexts=[dict(id="tiny")],
            prediction_trace=[],
        )
        install(runtime, torch)
        direction = torch.zeros(1024, dtype=torch.float64)
        direction[:64] = torch.randn(64, dtype=torch.float64)
        hook, _ = runtime.make_hook(direction)
        for _ in range(32):
            hidden = torch.randn(1, 3, 1024).bfloat16()
            changed, sentinel = hook(None, None, (hidden, "unchanged"))
            self.assertEqual(sentinel, "unchanged")
            self.assertTrue(torch.equal(hidden[:, :-1], changed[:, :-1]))
        self.assertEqual([a["prediction_step"] for a in runtime.prediction_trace], list(range(32)))
        for prediction_step in runtime.prediction_trace:
            self.assertEqual(prediction_step["requested_squared_edit_norm"], 0.04 * 16**2)
            self.assertLessEqual(
                prediction_step["realized_squared_edit_norm"],
                prediction_step["requested_squared_edit_norm"],
            )
            self.assertLessEqual(prediction_step["norm_relative_drift"], 0.001)
            self.assertGreaterEqual(prediction_step["cosine"], 0.9)
            self.assertEqual(prediction_step["normalization_coordinates"], 1024)
            self.assertEqual(prediction_step["steering_coordinate_count"], 64)


if __name__ == "__main__":
    unittest.main()
