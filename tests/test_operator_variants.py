"""Check sparse-edit and constrained-normalization diagnostics."""

import unittest

from sparse_steering.evaluation.sweep_checks import raw_audit


class VariantContracts(unittest.TestCase):

    def records(self):
        fit = dict(masked_direction=[1.0, 0.0, 0.0], extra_normalization_mask=[0, 1, 0], scale=1.0)
        values = [
            dict(
                id="x",
                order=o,
                accuracy=1.0,
                prediction_step_count=1,
                realized_squared_edit_norm=0.01,
                requested_squared_edit_norm=0.01,
                correct_probability=0.9,
                answer_label_probability_mass=1.0,
                correct_answer_label_nll=0.1,
            )
            for o in (0, 1)
        ]
        prediction_steps = [
            dict(
                id="x",
                order=o,
                prediction_step=0,
                realized_squared_edit_norm=0.01,
                requested_squared_edit_norm=0.01,
                cosine=1.0,
                norm_relative_drift=0.5,
                changed_coordinates=1,
                outside_support_squared_edit_norm=0.0,
                support_unchanged=1.0,
                displacement_sign_aligned=1.0,
                quantization_adjustment_steps=0.0,
            )
            for o in (0, 1)
        ]
        return fit, values, prediction_steps

    def test_addition_records_norm_drift_but_gates_support(self):
        fit, values, prediction_steps = self.records()
        config = dict(
            method="sparsification_only",
            edit_norm_ratio_squared=0.01,
            continuation_squared_norm_multiplier=1.0,
        )
        proof = raw_audit(values, prediction_steps, [{"id": "x"}], "binary", config, fit)
        self.assertEqual(proof["maximum_norm_drift"], 0.5)
        prediction_steps[0]["outside_support_squared_edit_norm"] = 1e-08
        with self.assertRaises(AssertionError):
            raw_audit(values, prediction_steps, [{"id": "x"}], "binary", config, fit)

    def test_scaling_allows_capacity_backoff_and_requires_norm(self):
        fit, values, prediction_steps = self.records()
        config = dict(
            method="ours", edit_norm_ratio_squared=0.01, continuation_squared_norm_multiplier=1.0
        )
        for v in values:
            v["realized_squared_edit_norm"] = 0.003
        for a in prediction_steps:
            a.update(
                realized_squared_edit_norm=0.003,
                norm_relative_drift=0.0001,
                alpha=0.3,
                beta=0.9,
                steering_coordinate_count=1,
                normalization_coordinates=2,
                effective_target_squared_edit_norm=0.003,
                no_op=0,
                capacity_limited=1,
                numerical_backoff=0,
            )
        self.assertEqual(
            raw_audit(values, prediction_steps, [{"id": "x"}], "binary", config, fit)[
                "capacity_limited_prediction_steps"
            ],
            2,
        )
        prediction_steps[0]["norm_relative_drift"] = 0.002
        with self.assertRaises(AssertionError):
            raw_audit(values, prediction_steps, [{"id": "x"}], "binary", config, fit)


if __name__ == "__main__":
    unittest.main()
