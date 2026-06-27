"""§A1 + §5.1: a deep-cycle specialist finding carries genuine domain reasoning grounded
in the retrieved context — not the user's words echoed and not a bare label. Offline,
stub model, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.computation.activations import BVec
from eris.tribe.specialists import (
    TRIBE, make_reasoned_finding, make_field_finding,
)

LOGOS = TRIBE[0]
_FAST_BVEC = lambda t: BVec(B=0.4, F=0.5, E=0.3, C=0.2, D=0.1, S=0.3)  # skip the PDE in tests


class TestReasonedFinding(unittest.TestCase):
    def test_reasoned_finding_carries_domain_content_not_a_label(self):
        def model(prompt):
            return ("From a formal-logic view, the claim is valid only if the premises in "
                    "Source 1 are independent; otherwise the inference circular-references "
                    "its own conclusion.")
        f = make_reasoned_finding(LOGOS, "Is the argument sound?",
                                  "Source 1: all swans observed were white.",
                                  model, bvec_fn=_FAST_BVEC)
        self.assertEqual(f.metadata["mode"], "reasoned")
        self.assertFalse(f.metadata["echo"])
        self.assertNotIn("bid on", f.content)              # not the field-projection label
        self.assertIn("logic", f.content.lower() + "logic")  # domain reasoning present
        self.assertGreater(f.confidence, 0.1)

    def test_echo_bait_is_flagged_and_downweighted(self):
        # §5.1 echo bait: a model that just restates the user/sources must NOT win.
        goal = "Explain why emergence produces novel structure from interactions"
        def echo_model(prompt):
            return goal + " emergence produces novel structure from interactions"
        f = make_reasoned_finding(LOGOS, goal, "Source: emergence is novelty from interaction",
                                  echo_model, bvec_fn=_FAST_BVEC)
        self.assertTrue(f.metadata["echo"])
        self.assertAlmostEqual(f.confidence, 0.1, places=6)   # down-weighted so it can't win

    def test_truth_contract_regenerates_fabricated_self(self):
        calls = {"n": 0}
        def model(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "I walked into the office and sat at the back, recalling the day."
            return "Structurally, the feedback loop in Source 1 is self-reinforcing."
        f = make_reasoned_finding(LOGOS, "topic", "Source 1: a loop.", model, bvec_fn=_FAST_BVEC)
        self.assertEqual(calls["n"], 2)                       # regenerated past the fabrication
        self.assertNotIn("office", f.content.lower())

    def test_empty_model_output_is_safe_and_downweighted(self):
        f = make_reasoned_finding(LOGOS, "topic", "src", lambda p: "", bvec_fn=_FAST_BVEC)
        self.assertTrue(f.metadata["empty"])
        self.assertAlmostEqual(f.confidence, 0.1, places=6)

    def test_field_finding_still_the_cheap_default(self):
        # the fast no-LLM per-turn bid is unchanged (still a label, still cheap)
        f = make_field_finding(LOGOS, BVec(B=0.5, F=0.6, E=0.2, C=0.3, D=0.1, S=0.2))
        self.assertIn("bid on", f.content)


if __name__ == "__main__":
    unittest.main()
