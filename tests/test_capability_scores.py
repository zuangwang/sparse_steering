"""Check choice likelihood aggregation and paired capability scoring."""

import unittest

from sparse_steering.evaluation.capability import summary as summarize


class CapabilityTests(unittest.TestCase):

    def test_choice_accuracy_and_wikitext_token_weighting(self):
        rows = [
            dict(subject="a", accuracy=1.0),
            dict(subject="b", accuracy=0.0),
            dict(subject="b", accuracy=0.0),
        ]
        self.assertEqual(summarize("arc_challenge", rows)["accuracy"], 1 / 3)
        rows = [dict(nll_sum=2.0, scored_tokens=1), dict(nll_sum=2.0, scored_tokens=3)]
        self.assertEqual(summarize("wikitext", rows)["nll"], 1.0)
