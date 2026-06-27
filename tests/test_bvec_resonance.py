"""§B3/§0.2/§0.3: the per-domain cos+sin coupling metric is shared by the MoEGate, the
cross-attention hub, and specialist activation — one field metric, not three cosines."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.computation.activations import BVec, bvec_resonance, bvec_resonance_2d
from eris.executive.workspace import MoEGate
from eris.tribe.specialists import (
    SpecialistFinding, CrossAttentionHub, Specialist, get_active_specialists,
)


class TestBvecResonance(unittest.TestCase):
    def test_matches_moegate_score_bid(self):
        # factoring is behavior-preserving (the gate's correct 2D core is untouched)
        rng = np.random.RandomState(0)
        for _ in range(50):
            a = BVec(*rng.rand(6)); g = BVec(*rng.rand(6))
            gate = MoEGate(); gate.set_goal(g)
            self.assertAlmostEqual(gate.score_bid(SpecialistFinding("x", "", a)),
                                   bvec_resonance(a, g), places=5)

    def test_resonance_is_gpu_safe_uses_to_numpy(self):
        # Regression guard (GPU-only bug, invisible to CPU tests): davidian_weight returns
        # an xp array (CuPy on GPU); converting it with np.asarray() raises "implicit
        # conversion". The helper MUST use to_numpy(.get()). Assert at the source level,
        # since CPU's to_numpy falls back to np.asarray and can't reproduce the failure.
        import inspect
        from eris.computation import activations
        src = inspect.getsource(activations.bvec_resonance)
        self.assertIn("to_numpy(davidian_weight", src)
        self.assertNotIn("np.asarray(davidian_weight", src)

    def test_constructive_beats_destructive(self):
        goal = BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3)
        aligned = BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3)     # same → constructive
        opposed = BVec(B=0.1, F=0.1, E=0.8, C=0.8, D=0.7, S=0.1)     # different → destructive
        self.assertGreater(bvec_resonance(aligned, goal), bvec_resonance(opposed, goal))

    def test_hub_query_uses_resonance(self):
        hub = CrossAttentionHub()
        near = SpecialistFinding("a", "near", BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3))
        far = SpecialistFinding("b", "far", BVec(B=0.1, F=0.1, E=0.8, C=0.8, D=0.7, S=0.1))
        hub.post(near); hub.post(far)
        top = hub.query(BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3), top_k=1)
        self.assertEqual(top[0].specialist_id, "a")

    def test_specialist_activation_runs_on_resonance(self):
        # strong-alignment goal activates the matching specialist (no exception, boolean)
        s = Specialist("logos", "Logos", "logic", "",
                       sensitivity_bvec=BVec(B=0.6, F=0.7, E=0.3, C=0.2, D=0.1, S=0.3))
        out = s.should_activate(BVec(B=0.6, F=0.7, E=0.3, C=0.2, D=0.1, S=0.3))
        self.assertIsInstance(out, bool)

    def test_active_set_is_capped_and_never_empty(self):
        # §2 cost cap: tribe-relative top-K, never all eleven, never zero — even on a
        # saturated goal bvec where every resonance is negative.
        saturated = BVec(B=0.0, F=1.0, E=0.07, C=1.0, D=0.05, S=0.0)
        active = get_active_specialists(saturated, max_k=5)
        self.assertGreaterEqual(len(active), 1)
        self.assertLessEqual(len(active), 5)
        self.assertLess(len(active), 11)
        # tighter cap honoured
        self.assertLessEqual(len(get_active_specialists(saturated, max_k=3)), 3)


class TestBvecResonance2D(unittest.TestCase):
    def test_keeps_both_channels_magnitude_geq_abs_net(self):
        # the 2D form must NOT discard the sine: magnitude √(κ²+λ²) ≥ |net elastic−plastic|
        rng = np.random.RandomState(1)
        for _ in range(50):
            a = BVec(*rng.rand(6)); g = BVec(*rng.rand(6))
            r2 = bvec_resonance_2d(a, g)
            self.assertGreaterEqual(r2["magnitude"] + 1e-9, abs(bvec_resonance(a, g)))

    def test_torsion_channel_nonzero_on_mismatch(self):
        # a partially-aligned neighbor carries signal in the λ/sine (torsion) channel
        goal = BVec(B=0.8, F=0.7, E=0.2, C=0.2, D=0.1, S=0.3)
        mixed = BVec(B=0.8, F=0.1, E=0.2, C=0.8, D=0.1, S=0.3)   # half aligned, half opposed
        self.assertGreater(abs(bvec_resonance_2d(mixed, goal)["R_sin"]), 0.0)

    def test_is_gpu_safe_uses_to_numpy(self):
        import inspect
        from eris.computation import activations
        src = inspect.getsource(activations.bvec_resonance_2d)
        self.assertIn("to_numpy(davidian_weight", src)
        self.assertNotIn("np.asarray(davidian_weight", src)


class TestResonanceRerank(unittest.TestCase):
    def test_reranks_by_resonance_magnitude_keeping_torsion(self):
        from eris.tribe.research import resonance_rerank
        goal = BVec(B=0.9, F=0.8, E=0.1, C=0.1, D=0.1, S=0.2)
        # map each text to a bvec deterministically (no PDE in the test)
        table = {
            "aligned": BVec(B=0.9, F=0.8, E=0.1, C=0.1, D=0.1, S=0.2),    # strong resonance
            "weak":    BVec(B=0.1, F=0.1, E=0.9, C=0.9, D=0.8, S=0.1),    # poor resonance
        }
        texts = ["weak", "aligned"]            # incoming order puts the poor match first
        out = resonance_rerank(goal, texts, blend=1.0, bvec_of=lambda t: table[t])
        self.assertEqual(out[0], "aligned")    # resonance promotes the real match

    def test_blend_zero_preserves_incoming_order(self):
        from eris.tribe.research import resonance_rerank
        goal = BVec(B=0.5, F=0.5, E=0.5, C=0.5, D=0.5, S=0.5)
        texts = ["first", "second", "third"]
        out = resonance_rerank(goal, texts, blend=0.0,
                               bvec_of=lambda t: BVec(*([0.5] * 6)))
        self.assertEqual(out, texts)

    def test_empty_and_missing_bvec_are_safe(self):
        from eris.tribe.research import resonance_rerank
        goal = BVec(*([0.5] * 6))
        self.assertEqual(resonance_rerank(goal, []), [])
        def _boom(t):
            raise ValueError("no bvec")
        out = resonance_rerank(goal, ["a", "b"], bvec_of=_boom)   # falls back to incoming order
        self.assertEqual(out, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
