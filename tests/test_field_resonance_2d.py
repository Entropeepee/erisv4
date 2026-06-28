"""§B3/§0.5: field_resonance keeps BOTH channels. The scalar cosine form is preserved
for back-compat; the 2D form surfaces the signed sine/torsion channel that the cosine
integral discards — proving the sine carries non-redundant structure (fixes §0.5)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.retrieval.field_interference import (
    field_resonance, field_resonance_2d, FieldInterferenceRetriever, resonance_vs_cosine,
)


class _Rec:
    def __init__(self, phi, theta, emb=None, title=""):
        self.phi_snapshot = phi; self.theta_snapshot = theta
        self.embedding = emb; self.metadata = {"title": title}; self.text = title


class _Mem:
    def __init__(self, recs):
        self.ltm = type("L", (), {"_records": recs})()


class TestResonance2D(unittest.TestCase):
    def setUp(self):
        self.phi = np.ones((8, 8))
        self.theta0 = np.zeros((8, 8))

    def test_scalar_is_backward_compatible(self):
        # field_resonance still returns the cosine-only scalar R_cos.
        r = field_resonance(self.phi, self.theta0, self.phi, self.theta0)
        self.assertAlmostEqual(r, field_resonance_2d(self.phi, self.theta0,
                                                     self.phi, self.theta0)["R_cos"], places=9)
        self.assertIsInstance(r, float)

    def test_sine_channel_carries_what_cosine_discards(self):
        # A constant phase offset of π/2 → cosine ≈ 0 (the cosine integral says "unrelated")
        # but the SINE/torsion channel is large: the fields ARE coupled, just out of phase.
        theta_q = np.full((8, 8), np.pi / 2)
        r2 = field_resonance_2d(self.phi, theta_q, self.phi, self.theta0)
        self.assertLess(abs(r2["R_cos"]), 1e-6)            # cosine blind
        self.assertGreater(abs(r2["R_sin"]), 0.5)          # torsion sees the coupling
        self.assertGreater(r2["magnitude"], 0.5)
        self.assertAlmostEqual(abs(r2["mixing_angle"]), np.pi / 2, places=3)  # fully torsion

    def test_identical_fields_are_pure_curvature(self):
        r2 = field_resonance_2d(self.phi, self.theta0, self.phi, self.theta0)
        self.assertGreater(r2["R_cos"], 0.5)
        self.assertAlmostEqual(r2["R_sin"], 0.0, places=6)   # no torsion when aligned
        self.assertAlmostEqual(r2["mixing_angle"], 0.0, places=6)

    def test_magnitude_dominates_scalar(self):
        theta_q = np.full((8, 8), 1.0)
        r2 = field_resonance_2d(self.phi, theta_q, self.phi, self.theta0)
        self.assertGreaterEqual(r2["magnitude"] + 1e-9, abs(r2["R_cos"]))

    def test_retriever_ranks_by_magnitude_finds_torsion_neighbor(self):
        # A torsion-coupled record (π/2 offset: big R_sin, ~0 R_cos) outranks a weakly
        # cosine-aligned one when ranking by magnitude — the neighbor cosine would miss.
        torsion_rec = _Rec(self.phi, np.full((8, 8), np.pi / 2), title="torsion")
        weak_rec = _Rec(self.phi * 0.2, np.zeros((8, 8)), title="weak-cos")
        mem = _Mem([weak_rec, torsion_rec])
        top = FieldInterferenceRetriever(mem).retrieve(self.phi, self.theta0, k=1)
        self.assertEqual(top[0][1].metadata["title"], "torsion")

    def test_resonance_vs_cosine_reports_torsion(self):
        recs = [_Rec(self.phi, np.full((8, 8), np.pi / 2), emb=np.array([1.0, 0.0]), title="a"),
                _Rec(self.phi * 0.5, np.zeros((8, 8)), emb=np.array([0.0, 1.0]), title="b")]
        mem = _Mem(recs)
        out = resonance_vs_cosine(mem, [{"phi": self.phi, "theta": self.theta0,
                                         "embedding": np.array([1.0, 0.0]), "title": "p"}])
        self.assertIn("mean_torsion_magnitude", out)
        self.assertGreaterEqual(out["mean_torsion_magnitude"], 0.0)


class TestFieldResonanceRerank(unittest.TestCase):
    """field_resonance_rerank ranks retrieved chunks by the genuine phase-based field resonance
    (signed sin Δθ torsion), not bvec/cosine. Fields injected via field_of — no PDE."""
    def setUp(self):
        self.phi = np.ones((8, 8)); self.zero = np.zeros((8, 8))
        self.fields = {
            "aligned": (self.phi, self.zero),                       # same phase → strong R_cos
            "torsion": (self.phi, np.full((8, 8), np.pi / 2)),      # π/2 offset → strong R_sin
            "weak":    (self.phi * 0.05, self.zero),               # tiny φ → ~0 magnitude
        }

    def _rerank(self, texts, blend=1.0):
        from eris.tribe.research import field_resonance_rerank
        return field_resonance_rerank((self.phi, self.zero), texts, blend=blend,
                                      field_of=lambda t: self.fields[t])

    def test_promotes_field_resonant_chunk(self):
        self.assertEqual(self._rerank(["weak", "aligned"])[0], "aligned")

    def test_torsion_coupled_chunk_beats_weak(self):
        # a π/2-out-of-phase but high-φ chunk resonates through the SIGNED sine channel and
        # outranks a weak in-phase one — the whole point of phase-based field resonance
        self.assertEqual(self._rerank(["weak", "torsion"])[0], "torsion")

    def test_blend_zero_preserves_incoming_order(self):
        self.assertEqual(self._rerank(["weak", "aligned", "torsion"], blend=0.0),
                         ["weak", "aligned", "torsion"])

    def test_empty_and_bad_field_are_safe(self):
        from eris.tribe.research import field_resonance_rerank
        self.assertEqual(field_resonance_rerank((self.phi, self.zero), []), [])
        def _boom(t):
            raise ValueError("no field")
        out = field_resonance_rerank((self.phi, self.zero), ["a", "b"], field_of=_boom)
        self.assertEqual(out, ["a", "b"])              # falls back to incoming order


class TestTextToFieldEvolved(unittest.TestCase):
    def test_returns_two_same_shape_fields(self):
        from eris.tribe.specialists import _text_to_field
        phi, theta = _text_to_field("boundary limited exchange critical dynamics", field_size=16)
        self.assertEqual(phi.shape, (16, 16))
        self.assertEqual(theta.shape, (16, 16))
        self.assertEqual(str(phi.dtype), "float32")


if __name__ == "__main__":
    unittest.main()
