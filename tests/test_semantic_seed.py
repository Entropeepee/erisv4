"""Phase-2 Step 4 — modular semantic-seed substrate + the 2×2 ablation harness.

Offline / deterministic: ERIS_EMBEDDINGS=off → get_embedding is the deterministic hashed fallback,
FractalField has a fixed seed, so every projector/descriptor/cell is reproducible. These tests pin
the PLUMBING (shapes, determinism, non-degeneracy, the 2×2 runs) — NOT which arm wins (that's the
empirical result when run with bge-m3 on the GPU box).
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import math
import unittest
import warnings

import numpy as np

from eris.config import to_numpy
from eris.field.seeding import (
    field_lambda, PlaneWaveProjector, RandomFourierProjector,
    BVecDescriptor, LambdaDescriptor, seed_and_evolve, signature,
    run_pairing, run_ablation_2x2,
)

# tiny labelled set: 3 meaning-groups, 2 surface forms each
TEXTS = [
    "the cat sat on the mat", "a feline rested upon the rug",          # group 0
    "stock prices fell sharply today", "markets dropped hard this session",  # group 1
    "photosynthesis converts light to sugar", "plants turn sunlight into glucose",  # group 2
]
GROUPS = [0, 0, 1, 1, 2, 2]


class TestLambdaChannel(unittest.TestCase):
    def test_lambda_is_finite_scalar_in_unit_range(self):
        rng = np.random.default_rng(0)
        phi = rng.uniform(0, 1, (32, 32)).astype(np.float32)
        theta = rng.uniform(0, 2 * math.pi, (32, 32)).astype(np.float32)
        lam = field_lambda(phi, theta)
        self.assertTrue(math.isfinite(lam))
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)

    def test_lambda_zero_when_gradients_parallel(self):
        # θ ≡ φ → ∇θ ∥ ∇φ → cross product ≈ 0 → λ ≈ 0 (the perpendicular channel is empty)
        x = np.linspace(0, 1, 32, dtype=np.float32)
        phi = np.tile(x, (32, 1))
        theta = phi.copy()
        self.assertLess(field_lambda(phi, theta), 0.05)


class TestSeedProjectors(unittest.TestCase):
    def test_planewave_shapes_and_bounds(self):
        phi, theta = PlaneWaveProjector().seed_text("a complex thought", size=32)
        self.assertEqual(phi.shape, (32, 32))
        self.assertTrue((phi >= 0).all() and (phi <= 1).all())
        self.assertTrue((theta >= 0).all() and (theta < 2 * math.pi + 1e-3).all())

    def test_planewave_deterministic(self):
        a = PlaneWaveProjector().seed_text("hello world", size=32)[0]
        b = PlaneWaveProjector().seed_text("hello world", size=32)[0]
        np.testing.assert_array_equal(a, b)

    def test_random_fourier_frozen_and_valid(self):
        p = RandomFourierProjector(n_modes=32, seed=7)
        phi1, _ = p.seed_text("emergence", size=32)
        phi2, _ = RandomFourierProjector(n_modes=32, seed=7).seed_text("emergence", size=32)
        np.testing.assert_allclose(phi1, phi2, atol=1e-6)         # frozen projection → reproducible
        self.assertEqual(phi1.shape, (32, 32))
        self.assertTrue((phi1 >= 0).all() and (phi1 <= 1).all())

    def test_random_fourier_dc_pinned_to_offset(self):
        # DC is NOT driven by the embedding: phi mean ≈ dc_offset
        p = RandomFourierProjector(n_modes=16, dc_offset=0.12)
        phi, _ = p.seed_text("anything at all", size=32)
        self.assertAlmostEqual(float(phi.mean()), 0.12, delta=0.03)

    def test_different_meanings_give_different_seeds(self):
        a = PlaneWaveProjector().seed_text("the cat sat on the mat", size=32)[0]
        b = PlaneWaveProjector().seed_text("stock prices fell sharply", size=32)[0]
        self.assertGreater(float(np.abs(a - b).sum()), 0.01)


class TestDescriptors(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.field = seed_and_evolve(PlaneWaveProjector(), "structured emergence", size=32, steps=8)

    def test_bvec_is_six(self):
        self.assertEqual(BVecDescriptor().describe(self.field).shape, (6,))

    def test_lambda_descriptor_is_seven_and_extends_bvec(self):
        v = LambdaDescriptor().describe(self.field)
        self.assertEqual(v.shape, (7,))
        np.testing.assert_allclose(v[:6], BVecDescriptor().describe(self.field), atol=1e-9)

    def test_lambda_not_degenerate_with_F_or_C(self):
        v = LambdaDescriptor().describe(self.field)
        F, C, lam = v[1], v[3], v[6]                              # B F E C D S λ
        self.assertGreater(abs(lam - F), 1e-4)
        self.assertGreater(abs(lam - C), 1e-4)


class TestHarness(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_pairing_runs_and_scores_finite(self):
        r = run_pairing(PlaneWaveProjector(), BVecDescriptor(), TEXTS, GROUPS, size=32, steps=8)
        self.assertEqual(r["signatures"].shape, (6, 6))
        self.assertTrue(math.isfinite(r["separation"]))

    def test_2x2_runs_all_four_cells_deterministically(self):
        a = run_ablation_2x2(TEXTS, GROUPS, size=32, steps=6)
        self.assertEqual(len(a["cells"]), 4)
        for k, v in a["cells"].items():
            self.assertTrue(math.isfinite(v), f"{k} = {v}")
        # determinism: same inputs → same 2×2
        b = run_ablation_2x2(TEXTS, GROUPS, size=32, steps=6)
        self.assertEqual(a["cells"], b["cells"])
        # the headline diagonal cells exist
        self.assertIn("planewave × bvec", a["cells"])
        self.assertIn("randomfourier × bvec+lambda", a["cells"])


class TestSeedWiring(unittest.TestCase):
    def test_encode_text_embedding_seam(self):
        from eris.field.pde import encode_text
        from eris.knowledge.embeddings import get_embedding
        default = to_numpy(encode_text("a sentence", size=32)[0])
        real = to_numpy(encode_text("a sentence", size=32, embedding=get_embedding("a sentence"))[0])
        self.assertGreater(float(np.abs(default - real).sum()), 0.01)   # the seam actually changes the seed

    def test_use_frt_flag_is_wired(self):
        from eris.field.pde import FractalField
        f1 = FractalField(size=32); f1.seed_from_text("hello world", use_frt=False)
        f2 = FractalField(size=32); f2.seed_from_text("hello world", use_frt=True)
        diff = float(np.abs(to_numpy(f1.phi) - to_numpy(f2.phi)).sum())
        self.assertGreater(diff, 0.01, "use_frt=True must take a different seed path (it was ignored)")


if __name__ == "__main__":
    unittest.main()
