"""§5 A/B harness — the metric computation is offline-testable (the real run is [machine])."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.experiments.hive_ab import (
    citation_resolution_rate,
    source_alignment,
    metrics_from,
)


class TestHiveAB(unittest.TestCase):
    def test_citation_resolution_rate(self):
        self.assertEqual(citation_resolution_rate("a [s:0] b [s:1]", 2), 1.0)   # all resolve
        self.assertEqual(citation_resolution_rate("a [s:0] b [s:9]", 2), 0.5)   # one dangling
        self.assertEqual(citation_resolution_rate("no citations here", 2), 0.0)  # control

    def test_source_alignment_rewards_drawing_on_sources(self):
        # a synthesis built from source text aligns highly; an invented one does not
        sources = ["The boundary limited exchange governs critical dynamics in the field."]
        drawn = "The boundary limited exchange governs critical dynamics."
        invented = "Quarterly revenue projections exceeded shareholder expectations again."
        self.assertGreater(source_alignment(drawn, sources), source_alignment(invented, sources))
        self.assertEqual(source_alignment("", sources), 0.0)               # nothing to align
        self.assertEqual(source_alignment("ab cd", []), 0.0)               # no sources

    def test_metrics_from_uses_pre_ground_synthesis_for_resolution(self):
        # pre-ground text has a dangling cite the final (stripped) text no longer shows;
        # honest resolution must measure the model's OWN claims (pre-ground)
        summary = {
            "n_sources": 2,
            "synthesis": "Grounded [s:0].",                       # post-strip
            "synthesis_pre_ground": "Grounded [s:0]. Bad [s:9].",  # pre-strip
            "sources": ["Grounded fact from a real source about the topic."],
            "n_contributors": 3,
            "n_active": 5,
            "cycles": 2,
            "stripped_claims": 1,
            "canonized": True,
        }
        m = metrics_from(summary)
        self.assertEqual(m["citation_resolution_pre_ground"], 0.5)   # one of two cites dangled
        self.assertEqual(m["domain_diversity"], 3)
        self.assertEqual(m["cycles"], 2)
        self.assertTrue(m["canonized"])
        self.assertGreater(m["synthesis_len"], 0)

    def test_no_data_run_is_inconclusive_not_a_hive_sweep(self):
        # both arms 0 sources → INCONCLUSIVE, never a 4/4 hive verdict (the 0.0>=0.0 tie bug)
        import asyncio
        from eris.experiments.hive_ab import run_ab

        class _Orch:
            async def hive_research(self, topic, *, max_specialists=5, mode="hive",
                                    scope="memory", document=""):
                return {"topic": topic, "n_sources": 0, "synthesis": "no sources",
                        "synthesis_pre_ground": "no sources", "sources": [],
                        "n_contributors": (5 if mode == "hive" else 0), "cycles": 0,
                        "n_active": (5 if mode == "hive" else 0), "canonized": False}
        out = asyncio.run(run_ab(_Orch(), "obscure"))
        self.assertIsInstance(out["verdict"], str)
        self.assertIn("INCONCLUSIVE", out["verdict"])

    def test_metrics_from_falls_back_when_no_pre_ground(self):
        # single-pass control may not carry a separate pre-ground draft → use final text
        summary = {"n_sources": 1, "synthesis": "A claim [s:0].", "sources": ["A claim source."]}
        m = metrics_from(summary)
        self.assertEqual(m["citation_resolution_pre_ground"], 1.0)
        self.assertEqual(m["cycles"], 0)
        self.assertFalse(m["canonized"])


if __name__ == "__main__":
    unittest.main()
