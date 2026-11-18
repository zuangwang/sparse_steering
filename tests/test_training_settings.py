"""Check that per-fit configuration preserves defaults and isolates mutations."""

import copy
import unittest

from sparse_steering.method.mask_training.training import training_settings
from sparse_steering.runtime.storage import SETTINGS
from sparse_steering.tasks.splits import seed_value


class TrainingSettings(unittest.TestCase):
    def test_explicit_settings_preserve_defaults_and_isolate_fits(self):
        defaults = copy.deepcopy(SETTINGS)
        for task, objective in (
            ("truthfulqa", "answer_selection"),
            ("paradetox", "open_generation"),
        ):
            expected = copy.deepcopy(defaults["mask_training"][objective])
            expected["seed"] = seed_value(1, task, training=True)
            expected["mask"].update(reruns=[0, 1, 2], update_fraction=0.4)
            settings = training_settings(1, task, (0, 1, 2), 0.4)
            self.assertEqual(settings, expected)
            settings["mask"]["reruns"].append(99)
            other = training_settings(1, task, (0, 1, 2), 0.4)
            self.assertEqual(other, expected)
        self.assertEqual(SETTINGS, defaults)
