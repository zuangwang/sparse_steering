"""Tiny tensor contracts; no pretrained model, datasets, or benchmark."""

import math
import types
import unittest

import torch

from sparse_steering.baselines.coordinate_selectors.attribution import logits, sequence_ie


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = types.SimpleNamespace(max_position_embeddings=64)


class ToyRuntime:

    def __init__(self):
        self.model = ToyModel()
        self.layer = torch.nn.Identity()
        self.eos_ids = []
        self.history = []

    def forward(self, ids, mask, use_cache=False, logits_to_keep=None):
        hidden = torch.zeros((*ids.shape, 2), dtype=torch.float32)
        hidden = self.layer(hidden)
        self.history.append(hidden.detach().clone())
        output = torch.stack([torch.zeros_like(hidden[..., 0]), hidden[..., 0] - 1], -1)
        return types.SimpleNamespace(logits=output[:, -logits_to_keep:, :])


class AttributionContracts(unittest.TestCase):

    def test_final_prompt_and_decode_schedule_only(self):
        runtime = ToyRuntime()
        logits(runtime, torch, [1, 2, 3, 4], 1, torch.tensor([2.0, 0.0]))
        expected = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [2 / math.sqrt(32), 0.0], [2 / math.sqrt(32), 0.0]]]
        )
        torch.testing.assert_close(runtime.history[-1], expected)

    def test_ten_midpoints_and_exact_linear_indirect_effect(self):
        runtime = ToyRuntime()
        endpoint = torch.tensor([2.0, 0.0])
        gradient, count = sequence_ie(runtime, torch, [1], 0, endpoint, [0, 1])
        self.assertEqual(count, 1)
        torch.testing.assert_close(gradient, torch.tensor([1.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(
            endpoint * gradient, torch.tensor([2.0, 0.0], dtype=torch.float64)
        )
        sampled = torch.tensor([float(h[0, 0, 0]) for h in runtime.history[2:]])
        torch.testing.assert_close(sampled, torch.arange(0.1, 2.0, 0.2))
        self.assertIsNone(runtime.model.anchor.grad)
        self.assertEqual(len(runtime.layer._forward_hooks), 0)

    def test_equal_greedy_predictions_contribute_zero(self):
        runtime = ToyRuntime()
        gradient, count = sequence_ie(runtime, torch, [1], 0, torch.tensor([0.5, 0.0]), [0, 1])
        self.assertEqual(count, 0)
        self.assertEqual(len(runtime.history), 2)
        self.assertEqual(gradient.tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
