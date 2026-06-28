"""Confidence as resonance geometry (cos match + sin/torsion of the unresolved part), per David's
idea that cosine is the MATCH while sine/torsion define the AMOUNT and QUALITY of the unresolved
info. Pure vector math — deterministic, no model needed."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import math
import unittest

import numpy as np

from eris.computation.confidence import resonance_confidence


def v(*x):
    return np.array(x, dtype=np.float32)


class TestResonanceConfidence(unittest.TestCase):
    def test_perfect_alignment_is_high_match_low_unresolved(self):
        c = resonance_confidence(v(1, 0, 0), [v(1, 0, 0), v(1, 0, 0)])
        self.assertAlmostEqual(c["match"], 1.0, places=5)
        self.assertAlmostEqual(c["unresolved"], 0.0, places=5)
        self.assertAlmostEqual(c["torsion"], 0.0, places=5)
        self.assertGreater(c["confidence"], 0.9)

    def test_orthogonal_evidence_is_low_match_high_unresolved(self):
        c = resonance_confidence(v(1, 0, 0), [v(0, 1, 0)])
        self.assertAlmostEqual(c["match"], 0.0, places=5)
        self.assertAlmostEqual(c["unresolved"], 1.0, places=5)
        self.assertAlmostEqual(c["torsion"], math.pi / 2, places=5)
        self.assertAlmostEqual(c["confidence"], 0.0, places=5)

    def test_conservation_law_holds(self):
        # cos^2 + sin^2 = 1 for any claim/evidence
        c = resonance_confidence(v(0.6, 0.8, 0.0), [v(1, 0, 0), v(0, 1, 0)])
        self.assertAlmostEqual(c["match"] ** 2 + c["unresolved"] ** 2, 1.0, places=5)

    def test_coherence_distinguishes_agreeing_from_scattered_evidence(self):
        # same match to centroid is not the point — agreeing supports vs scattered ones differ in
        # the QUALITY of what's unresolved (coherence), exactly David's sin/torsion 'quality' axis
        agree = resonance_confidence(v(1, 0, 0), [v(1, 0.1, 0), v(1, 0.12, 0), v(1, 0.08, 0)])
        scatter = resonance_confidence(v(1, 0, 0), [v(1, 1, 0), v(1, -1, 0), v(1, 0, 1)])
        self.assertGreater(agree["coherence"], scatter["coherence"])
        self.assertGreater(agree["confidence"], scatter["confidence"])   # coherent support → surer

    def test_single_source_has_full_coherence(self):
        c = resonance_confidence(v(1, 0, 0), [v(1, 0, 0)])
        self.assertEqual(c["coherence"], 1.0)

    def test_missing_inputs_degrade_to_zero_confidence(self):
        self.assertEqual(resonance_confidence(None, [v(1, 0)])["confidence"], 0.0)
        self.assertEqual(resonance_confidence(v(1, 0), [])["confidence"], 0.0)
        z = resonance_confidence(v(1, 0), [None])
        self.assertEqual(z["confidence"], 0.0)
        self.assertEqual(z["unresolved"], 1.0)


class TestConfidenceSurfacedInMetrics(unittest.TestCase):
    def test_metrics_from_passes_confidence_through(self):
        from eris.experiments.hive_ab import metrics_from
        summary = {"n_sources": 2, "synthesis_full": "x", "sources": ["s"],
                   "confidence": {"match": 0.7, "unresolved": 0.71, "coherence": 0.6,
                                  "torsion": 0.79, "confidence": 0.56}}
        m = metrics_from(summary)
        self.assertEqual(m["confidence"]["match"], 0.7)
        self.assertEqual(m["confidence"]["coherence"], 0.6)

    def test_metrics_from_tolerates_absent_confidence(self):
        from eris.experiments.hive_ab import metrics_from
        m = metrics_from({"n_sources": 1, "synthesis_full": "x", "sources": ["s"]})
        self.assertEqual(m["confidence"], {})


if __name__ == "__main__":
    unittest.main()
