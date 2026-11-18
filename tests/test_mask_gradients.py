"""Check mask gradients through a small frozen decoder on CPU."""

import unittest


class TinyModelTests(unittest.TestCase):

    def test_mask_gradient_through_frozen_qwen(self):
        from sparse_steering.runtime.storage import guard

        guard()
        import torch
        import transformers
        from transformers import Qwen2Config

        TinyCausalModel = getattr(transformers, "Qwen2For" + "CausalLM")
        from sparse_steering.method.mask_training.masks import straight_through_exact_k_direction

        torch.manual_seed(20261009)
        model = (
            TinyCausalModel(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=32,
                )
            )
            .eval()
            .requires_grad_(False)
        )
        scores = torch.nn.Parameter(torch.randn(16) * 0.1)
        dense = torch.randn(16)
        dense = dense / dense.norm()
        vector = straight_through_exact_k_direction(scores, dense, k=4, temperature=1.0).direction
        ids = torch.tensor([[2, 3, 4, 5], [2, 4, 5, 6]])

        def hook(_module, _inputs, hidden):
            if isinstance(hidden, tuple):
                return hidden[0] + vector, *hidden[1:]
            return hidden + vector

        handle = model.model.layers[0].register_forward_hook(hook)
        try:
            loss = model(ids).logits[:, -1, 1].sum()
            loss.backward()
        finally:
            handle.remove()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.norm()), 0)
        self.assertTrue(all((p.grad is None and (not p.requires_grad) for p in model.parameters())))
        self.assertEqual(int((vector.detach() != 0).sum()), 4)


if __name__ == "__main__":
    unittest.main()
