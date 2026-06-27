"""§5: adversarial cases for the restored hive + working memory. Offline, deterministic.
  1. Echo bait        — findings carry domain reasoning, not the user's words.
  2. Transfixion bait — repetitive input never runaway-concatenates; the frame stays bounded.
  3. Held-topic       — a topic she already holds routes to introspect, not the web.
  4. Grounding        — an unciteable claim in a canonized entry is stripped.
  5. Cost ceiling     — the active set is capped and reasoning stays on the injected (local) model.
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.computation.activations import BVec
from eris.tribe.specialists import (
    TRIBE, make_reasoned_finding, get_active_specialists, CrossAttentionHub,
)
from eris.tribe.research import run_two_cycle_research, _ground_citations
from eris.executive.workspace import SharedCognitiveWorkspace, MoEGate
from eris.metacognition.topic_router import route_topic

_FAST = lambda t: BVec(B=0.4, F=0.5, E=0.3, C=0.3, D=0.2, S=0.3)
GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


class TestAdversarialHive(unittest.TestCase):
    # 1 ── echo bait ─────────────────────────────────────────────────────────
    def test_echo_bait_carries_reasoning_not_user_words(self):
        goal = "tell me about resonance and coupling in fields"
        echo = make_reasoned_finding(TRIBE[0], goal, "src", lambda p: goal, bvec_fn=_FAST)
        reasoned = make_reasoned_finding(
            TRIBE[0], goal, "Source: coupled oscillators phase-lock above a threshold.",
            lambda p: "Above the locking threshold the oscillators synchronize; below it they drift.",
            bvec_fn=_FAST)
        self.assertTrue(echo.metadata["echo"]); self.assertAlmostEqual(echo.confidence, 0.1, 6)
        self.assertFalse(reasoned.metadata["echo"])

    # 2 ── transfixion bait: no runaway concatenation, bounded frame ──────────
    def test_transfixion_input_never_runaway_concatenates(self):
        ws = SharedCognitiveWorkspace()
        ws.set_goal("loop", GOAL)
        for _ in range(300):                      # repetitive flood
            ws.broadcast("same thought", BVec(E=0.5), source="x", coherence=0.3)
        self.assertLessEqual(len(ws.history), 100)         # deque bound holds
        self.assertEqual(len(ws.working_set(k=3)["broadcasts"]), 3)  # frame stays bounded
        # the MoEGate still returns a single winner under the flood (transfixion override path)
        gate = MoEGate(); gate.set_goal(GOAL)
        from eris.tribe.specialists import make_field_finding
        findings = [make_field_finding(TRIBE[i % len(TRIBE)], GOAL) for i in range(20)]
        w = gate.select_winner(findings)
        self.assertIsNotNone(w)

    # 3 ── held-topic routes to introspect, not web ──────────────────────────
    def test_held_topic_introspects_not_web(self):
        class _Mem:
            def retrieve(self, query_embedding=None, top_k=5):
                r = type("R", (), {})(); r.embedding = np.array([1.0, 0.0, 0.0]); return [r]
        out = route_topic("quantum entanglement", _Mem(),
                          embed=lambda q: np.array([1.0, 0.0, 0.0]))   # perfect coverage
        self.assertEqual(out["action"], "introspect")

    # 4 ── grounding: uncited/unresolved claim is stripped at canonize ───────
    def test_unciteable_claim_stripped(self):
        out, n = _ground_citations("Real [s:0]. Fabricated [s:77].", n_sources=1)
        self.assertEqual(n, 1)
        self.assertNotIn("[s:77]", out); self.assertIn("[s:0]", out)

    # 5 ── cost ceiling: active cap + reasoning stays on the injected model ──
    def test_cost_ceiling_cap_and_local_only(self):
        self.assertLessEqual(len(get_active_specialists(GOAL, max_k=5)), 5)
        calls = {"n": 0}
        def only_model(prompt):
            calls["n"] += 1                       # the ONLY model the engine may call
            return "Domain analysis [s:0]."
        res = run_two_cycle_research("topic", retriever=lambda q: ["a source"],
                                     model=only_model, specialists=TRIBE[:3], goal_bvec=GOAL)
        self.assertEqual(res.n_active, 3)         # capped set honoured
        self.assertGreater(calls["n"], 0)         # all reasoning went through the injected model


if __name__ == "__main__":
    unittest.main()
