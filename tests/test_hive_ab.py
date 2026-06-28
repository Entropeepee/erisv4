"""§5 A/B harness — the metric computation is offline-testable (the real run is [machine])."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

import random

from eris.experiments.hive_ab import (
    citation_resolution_rate,
    source_alignment,
    metrics_from,
    make_blind_pair,
)


class TestBlindPair(unittest.TestCase):
    def test_assignment_matches_key_both_branches(self):
        # force each branch via a stub rng; A/B text must match what _key says
        class _R:
            def __init__(self, v): self.v = v
            def random(self): return self.v
        lo = make_blind_pair("TREAT", "CTRL", rng=_R(0.1))   # rng<0.5 → A=treatment
        self.assertEqual(lo["_key"], {"A": "treatment", "B": "control"})
        self.assertEqual(lo["blind_pair"]["A"], "TREAT")
        self.assertEqual(lo["blind_pair"]["B"], "CTRL")
        hi = make_blind_pair("TREAT", "CTRL", rng=_R(0.9))   # rng>=0.5 → A=control
        self.assertEqual(hi["_key"], {"A": "control", "B": "treatment"})
        self.assertEqual(hi["blind_pair"]["A"], "CTRL")

    def test_randomizes_over_runs(self):
        rng = random.Random(0)
        seen = {make_blind_pair("T", "C", rng=rng)["_key"]["A"] for _ in range(30)}
        self.assertEqual(seen, {"treatment", "control"})     # both assignments occur

    def test_a_text_always_consistent_with_key(self):
        rng = random.Random(7)
        for _ in range(20):
            p = make_blind_pair("Tx", "Cx", rng=rng)
            expect = "Tx" if p["_key"]["A"] == "treatment" else "Cx"
            self.assertEqual(p["blind_pair"]["A"], expect)


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

    def test_metrics_surface_outcome_measures(self):
        summary = {"n_sources": 2, "synthesis": "x", "sources": ["s"],
                   "specialist_divergence": 0.62, "gaps_closed": 2, "elos_changed": True,
                   "elos_critique": "weakest claim: X is unsupported"}
        m = metrics_from(summary)
        self.assertEqual(m["specialist_divergence"], 0.62)
        self.assertEqual(m["gaps_closed"], 2)
        self.assertTrue(m["elos_changed"])
        self.assertIn("weakest", m["elos_critique"])

    def test_metrics_scored_on_full_not_truncated_synthesis(self):
        # source_alignment must use synthesis_full, not the display-truncated synthesis
        full = "alpha beta gamma delta epsilon zeta eta theta"
        summary = {"n_sources": 1, "synthesis": "alpha", "synthesis_full": full,
                   "sources": [full]}
        m = metrics_from(summary)
        self.assertEqual(m["synthesis_len"], len(full))     # measured the FULL text
        self.assertEqual(m["source_alignment"], 1.0)        # all full-text words are in source

    def test_verdict_measures_outcomes_not_tautologies(self):
        # treatment runs cycles & writes more but lenses DIDN'T diverge and gaps weren't closed
        # → it must NOT win on diversity/depth; only genuine outcomes count.
        import asyncio
        from eris.experiments.hive_ab import run_ab

        class _Orch:
            async def hive_research(self, topic, *, max_specialists=5, mode="hive",
                                    scope="memory", document=""):
                hive = mode == "hive"
                return {"topic": topic, "n_sources": 2,
                        "synthesis_full": ("the boundary coupling mechanism per the matrix"
                                           if hive else "boundary coupling"),
                        "synthesis": "x", "synthesis_pre_ground_full": "x [s:0]",
                        "sources": ["the boundary coupling mechanism per the matrix document"],
                        "n_contributors": 5 if hive else 0, "n_active": 5 if hive else 0,
                        "cycles": 2 if hive else 0,
                        "specialist_divergence": 0.05 if hive else 0.0,   # lenses echoed
                        "gaps_closed": 0, "elos_changed": False, "canonized": hive}
        out = asyncio.run(run_ab(_Orch(), "topic"))
        v = out["verdict"]
        self.assertFalse(v["hive_lenses_diverged"])        # 0.05 < 0.4 floor
        self.assertFalse(v["hive_closed_gaps"])            # 0 gaps closed
        self.assertFalse(v["hive_elos_bit"])               # no edit
        self.assertNotIn("hive_synthesis_longer", v)       # length demoted out of the verdict
        self.assertIn("blind_pair", out)
        self.assertIn("_key", out)
        self.assertEqual(list(out.keys())[-1], "_key")     # _key printed LAST


if __name__ == "__main__":
    unittest.main()
