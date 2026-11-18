"""Check inference hooks on a small decoder without pretrained weights."""

import unittest
from types import SimpleNamespace


class EveryTokenTests(unittest.TestCase):

    def test_tiny_model_32_prediction_steps_with_real_bf16_additive_hooks(self):
        import torch

        from sparse_steering.runtime.hooks import install_coordinate as faithful_install
        from sparse_steering.runtime.hooks import install_sparse as variant_install

        torch.manual_seed(17)
        torch.set_num_threads(2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for install in (faithful_install, variant_install):

            class TinyDecoder(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.embedding = torch.nn.Embedding(32, 256)
                    self.layer = torch.nn.Identity()
                    self.head = torch.nn.Linear(256, 32)

                def forward(self, ids):
                    return self.head(self.layer(self.embedding(ids)))

            model = TinyDecoder().to(device=device, dtype=torch.bfloat16).eval()
            runtime = SimpleNamespace(
                cell=dict(
                    method="sparsification_only",
                    edit_norm_ratio_squared=0.04,
                    continuation_squared_norm_multiplier=1.0,
                ),
                scale=8.0,
                _prediction_step=0,
                _contexts=[dict(id="tiny")],
                prediction_trace=[],
            )
            install(runtime, torch)
            direction = torch.randn(256, device=device, dtype=torch.float64)
            direction /= direction.norm()
            hook, _ = runtime.make_hook(direction)
            handle = model.layer.register_forward_hook(hook)
            tokens = torch.tensor([[1, 2, 3]], device=device)
            with torch.no_grad():
                original = model.embedding(tokens).clone()
                for step in range(32):
                    logits = model(tokens)
                    tokens = logits[:, -1].argmax(-1, keepdim=True)
            handle.remove()
            self.assertEqual(
                [a["prediction_step"] for a in runtime.prediction_trace], list(range(32))
            )
            squared_edit_norm_limit = 0.04 * 8**2
            for prediction_step in runtime.prediction_trace:
                self.assertEqual(
                    prediction_step["requested_squared_edit_norm"], squared_edit_norm_limit
                )
                self.assertLessEqual(
                    prediction_step["realized_squared_edit_norm"], squared_edit_norm_limit
                )
                self.assertAlmostEqual(
                    prediction_step["realized_squared_edit_norm"] / squared_edit_norm_limit,
                    1.0,
                    delta=0.01,
                )
            runtime._prediction_step = 0
            changed = hook(None, None, original)
            torch.testing.assert_close(changed[:, :-1], original[:, :-1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
