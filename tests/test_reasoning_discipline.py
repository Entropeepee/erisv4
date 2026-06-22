"""Tests for the deep-reasoning discipline: calibration + topic routing."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.reasoning.calibration import (
    verify_quotes, is_synthesis_task, calibration_system, has_identity_overreach,
)
from eris.metacognition.topic_router import route_topic, _looks_like_chat


class TestQuoteFidelity(unittest.TestCase):
    def test_fabricated_quote_demoted(self):
        # The real LNCS×SGT failure: a phrase attributed as a patent quote that
        # appears nowhere in the sources.
        sources = "The patent describes Path A and Path B with a shared accumulator."
        answer = ('The patent refers to CSBA as a "utility-application embodiment '
                  'of GLNCS", which shows the lineage.')
        cleaned, flags = verify_quotes(answer, sources)
        self.assertEqual(len(flags), 1)
        self.assertNotIn('"utility-application embodiment of GLNCS"', cleaned)
        self.assertIn("paraphrase, not a verbatim quote", cleaned)

    def test_verbatim_quote_kept(self):
        sources = "Path A is a slow always-on integrator that accumulates drift."
        answer = 'She notes "a slow always-on integrator that accumulates drift".'
        cleaned, flags = verify_quotes(answer, sources)
        self.assertEqual(flags, [])
        self.assertIn('"a slow always-on integrator that accumulates drift"', cleaned)

    def test_no_quotes_untouched(self):
        cleaned, flags = verify_quotes("A plain sentence with no quotes.", "src")
        self.assertEqual(flags, [])
        self.assertEqual(cleaned, "A plain sentence with no quotes.")


class TestCalibrationPrompt(unittest.TestCase):
    def test_synthesis_detection(self):
        self.assertTrue(is_synthesis_task("how does the LNCS paper inform my patent?"))
        self.assertTrue(is_synthesis_task("compare X and Y"))
        self.assertTrue(is_synthesis_task("a simple question", named_sources=2))
        self.assertFalse(is_synthesis_task("what time is it?"))

    def test_synthesis_prompt_has_sections_and_discipline(self):
        s = calibration_system(is_synthesis=True, regime="plastic")
        self.assertIn("VERB AUDIT", s)
        self.assertIn("ATTRIBUTION", s)
        self.assertIn("interpretive bridge", s)        # section 4
        self.assertIn("PLASTIC", s)                    # field-state clause
        light = calibration_system(is_synthesis=False)
        self.assertNotIn("Core answer —", light)       # no six-section scaffold

    def test_identity_overreach_signal(self):
        self.assertTrue(has_identity_overreach("the patent is the collapse theorem"))
        self.assertFalse(has_identity_overreach("the patent resembles the theorem"))


class _Rec:
    def __init__(self, emb):
        self.embedding = emb
        self.text = "held material"


class _Mem:
    def __init__(self, hits):
        self._hits = hits
    def retrieve(self, query_embedding=None, top_k=5):
        return self._hits


class TestTopicRouter(unittest.TestCase):
    def test_chat_residue_skips(self):
        self.assertEqual(route_topic("prompt", _Mem([]), lambda q: np.ones(8))["action"], "skip")
        self.assertEqual(route_topic("hey thanks", _Mem([]), lambda q: np.ones(8))["action"], "skip")
        self.assertEqual(route_topic("", _Mem([]), lambda q: np.ones(8))["action"], "skip")

    def test_held_material_introspects(self):
        v = np.ones(8, dtype=np.float32)
        plan = route_topic("Kuramoto coupling", _Mem([_Rec(v)]), lambda q: v)
        self.assertEqual(plan["action"], "introspect")    # high coverage -> own memory

    def test_novel_topic_webs(self):
        v = np.ones(8, dtype=np.float32)
        orth = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
        plan = route_topic("Kuramoto coupling", _Mem([_Rec(orth)]), lambda q: v)
        self.assertEqual(plan["action"], "web")           # low coverage -> crawl

    def test_looks_like_chat(self):
        self.assertTrue(_looks_like_chat("a long sentence with many many many many words here"))
        self.assertTrue(_looks_like_chat("done."))
        self.assertFalse(_looks_like_chat("Kuramoto model"))


if __name__ == "__main__":
    unittest.main()
