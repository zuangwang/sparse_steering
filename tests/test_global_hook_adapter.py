"""The binary and generation paths must both reach the frozen global hook."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sparse_steering.runtime.runner import configure


class GlobalHookAdapterTest(unittest.TestCase):

    def test_runtime_call_signatures_preserve_frozen_dispatch(self):
        frozen = Mock(return_value=("hook", "records"))
        module = SimpleNamespace(make_hook=frozen)
        runtime = SimpleNamespace(configure=Mock())
        torch = SimpleNamespace(tensor=lambda values, **kwargs: values, float64="float64")
        fit = dict(scale=3.0, masked_direction=[1.0, 0.0])
        with patch.dict(sys.modules, {"sparse_steering.method.global_normalization": module}):
            config, direction = configure(
                runtime, torch, None, "sparsification_global_norm", 0.01, fit
            )
        runtime.configure.assert_called_once_with(config, 3.0)
        self.assertEqual(runtime.make_hook(direction), ("hook", "records"))
        self.assertEqual(
            runtime.make_hook(direction, 0.0, matched=True, signs=None), ("hook", "records")
        )
        self.assertEqual(frozen.call_count, 2)
        frozen.assert_called_with(runtime, torch, direction)
        with self.assertRaises(AssertionError):
            runtime.make_hook(direction, matched=False)
        with self.assertRaises(AssertionError):
            runtime.make_hook(direction, signs=[1])


if __name__ == "__main__":
    unittest.main()
