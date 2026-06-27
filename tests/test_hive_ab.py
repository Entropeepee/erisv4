"""§5 A/B harness — the metric computation is offline-testable (the real run is [machine])."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.experiments.hive_ab import citation_resolution_rate, ab_metrics, compare


class _Res:
    synthesis = "Grounded claim [s:0]. Another [s:1]."
    n_contributors = 3
    n_active = 5
    cycles = 2
    stripped_claims = 1
    thought_id = "abc123"


class TestHiveAB(unittest.TestCase):
    def test_citation_resolution_rate(self):
        self.assertEqual(citation_resolution_rate("a [s:0] b [s:1]", 2), 1.0)   # all resolve
        self.assertEqual(citation_resolution_rate("a [s:0] b [s:9]", 2), 0.5)   # one dangling
        self.assertEqual(citation_resolution_rate("no citations here", 2), 0.0) # control

    def test_ab_metrics(self):
        m = ab_metrics(_Res(), n_sources=2)
        self.assertEqual(m["citation_resolution_rate"], 1.0)
        self.assertEqual(m["domain_diversity"], 3)
        self.assertTrue(m["canonized"])

    def test_compare_treatment_beats_hollow_control(self):
        c = compare(_Res(), n_sources=2)
        self.assertTrue(c["verdict"]["more_grounded"])
        self.assertTrue(c["verdict"]["more_diverse"])
        self.assertTrue(c["verdict"]["produced_synthesis"])
        self.assertEqual(c["control"]["citation_resolution_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
