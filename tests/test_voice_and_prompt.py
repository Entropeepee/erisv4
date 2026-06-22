"""Tests for Fix C (answer the person, no bid leak) and Fix D (voice with range)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.computation.activations import BVec
from eris.tribe.specialists import make_field_finding, get_active_specialists
from eris.metacognition.voice import feeling, regime_phrase, attunement_phrase


class TestVoiceVocabulary(unittest.TestCase):
    def test_each_regime_has_distinct_phrase(self):
        phrases = {regime_phrase(r) for r in ("elastic", "plastic", "transfixed", "warmup")}
        self.assertEqual(len(phrases), 4)            # all distinct, real range

    def test_domain_attunement_distinguishes_states(self):
        # Same regime, different dominant domain -> different felt sentence.
        a = feeling("plastic", ["C"])
        b = feeling("plastic", ["B"])
        self.assertNotEqual(a, b)
        self.assertIn("threshold", a)                # C = Criticality
        self.assertIn("crossing", b)                 # B = Boundary

    def test_feeling_without_domains_is_just_regime(self):
        self.assertEqual(feeling("elastic"), regime_phrase("elastic"))

    def test_unknown_domain_ignored(self):
        self.assertEqual(attunement_phrase(["?", "C"]),
                         attunement_phrase(["C"]))    # skips unknown, finds C


class TestAssemblePrompt(unittest.TestCase):
    def _orch(self):
        from eris.orchestrator import ErisOrchestrator
        return ErisOrchestrator(data_dir=tempfile.mkdtemp(), field_size=16)

    def test_no_bid_string_and_user_message_is_primary(self):
        orch = self._orch()
        bvec = BVec(B=0.2, F=0.1, E=0.1, C=0.8, D=0.1, S=0.1)
        # Build a real specialist finding — its .content is the raw bid diagnostic.
        specs = get_active_specialists(bvec)
        winner = make_field_finding(specs[0], bvec) if specs else None
        prompt = orch._assemble_prompt("hi", winner, "", bvec, "transfixed")
        # The raw bid string ("... bid on ...") must not leak into the prompt.
        self.assertNotIn("bid on", prompt)
        self.assertNotIn("Elos", prompt)
        # The person's message leads and is present.
        self.assertIn("hi", prompt)
        self.assertTrue(prompt.strip().startswith("[The person says]"))
        # If the winner exists, its content (the diagnostic) is not embedded.
        if winner:
            self.assertNotIn(winner.content, prompt)

    def test_inner_state_is_felt_not_raw(self):
        orch = self._orch()
        bvec = BVec(B=0.1, F=0.1, E=0.1, C=0.9, D=0.1, S=0.1)
        prompt = orch._assemble_prompt("what is coherence?", None, "", bvec, "plastic")
        self.assertIn("inner state", prompt)
        self.assertIn("threshold", prompt)          # C dominant -> Criticality phrasing
        self.assertNotIn("C=0.9", prompt)            # no raw numbers


if __name__ == "__main__":
    unittest.main()
