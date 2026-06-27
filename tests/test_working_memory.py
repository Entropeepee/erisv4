"""§B1/§B2: the workspace exposes a bounded, structured working-memory frame, and
goal-conditioned retrieval ranks by coherence-gain toward the active goal (not raw
similarity) and returns a bounded, structured top-k (never a concatenated string)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.computation.activations import BVec
from eris.executive.workspace import SharedCognitiveWorkspace
from eris.executive.working_memory import goal_conditioned_context, coherence_gain


class _Rec:
    def __init__(self, bvec, text):
        self.bvec = bvec; self.text = text


class TestWorkingMemory(unittest.TestCase):
    def test_working_set_is_bounded_and_structured(self):
        ws = SharedCognitiveWorkspace()
        ws.set_goal("understand emergence", BVec(E=0.8, C=0.5))
        for i in range(6):
            ws.broadcast(f"thought {i}", BVec(E=0.5, F=0.3), source=f"s{i}", coherence=0.4)
        frame = ws.working_set(k=3)
        self.assertEqual(frame["goal"], "understand emergence")
        self.assertEqual(len(frame["broadcasts"]), 3)             # bounded to k
        self.assertEqual(frame["broadcasts"][-1]["source"], "s5")  # most recent
        self.assertIsInstance(frame["broadcasts"], list)          # structured, not a string

    def test_goal_conditioned_ranks_by_coherence_gain(self):
        goal = BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3)
        aligned = _Rec(BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3), "aligned")
        opposed = _Rec(BVec(B=0.1, F=0.1, E=0.8, C=0.8, D=0.7, S=0.1), "opposed")
        out = goal_conditioned_context([opposed, aligned], goal, k=2)
        self.assertEqual(out[0]["record"].text, "aligned")         # coherence-gain ordering
        self.assertIsInstance(out, list)
        self.assertIn("coherence_gain", out[0])

    def test_bounded_topk(self):
        goal = BVec(B=0.5, F=0.5, E=0.5, C=0.5, D=0.5, S=0.5)
        cands = [_Rec(BVec(B=0.5, F=0.5, E=0.5, C=0.5, D=0.5, S=0.5), str(i)) for i in range(20)]
        self.assertEqual(len(goal_conditioned_context(cands, goal, k=5)), 5)

    def test_coherence_gain_sign(self):
        goal = BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3)
        self.assertGreater(coherence_gain(goal, goal),
                           coherence_gain(BVec(E=0.9, C=0.9, D=0.8), goal))


if __name__ == "__main__":
    unittest.main()
