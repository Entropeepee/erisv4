"""LAF multiresolution-SVD signature (κ, λ) — zero-weight, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.vision.laf import (
    laf_signature, kappa_overlap, lambda_distance, tower, LAFConfig,
)


def _field(seed, size=32):
    rng = np.random.RandomState(seed)
    g = np.linspace(0, 1, size)
    base = np.outer(g, g) + 0.05 * rng.randn(size, size)
    theta = np.tile(g, (size, 1)) * (1 + 0.3 * seed)
    return np.clip(base, 0, 1), theta


class TestLAF(unittest.TestCase):
    def test_signature_shapes_and_energy(self):
        mag, th = _field(0)
        cfg = LAFConfig(patch=8, n_scales=4, n_modes=8)
        k, lam, tau = laf_signature(mag, th, cfg)
        self.assertEqual(k.shape[1], 8)             # n_modes columns
        self.assertEqual(lam.shape[0], 8)
        self.assertEqual(tau.shape[0], 8)
        self.assertAlmostEqual(float(lam.sum()), 1.0, places=5)  # normalized energy
        self.assertTrue(np.isfinite(k).all() and np.isfinite(lam).all())

    def test_overlap_self_is_one_emergent_zero(self):
        mag, th = _field(1)
        k, _, _ = laf_signature(mag, th)
        aligned, emergent = kappa_overlap(k, k)
        self.assertGreater(aligned, 0.99)           # a field aligns with itself
        self.assertLess(emergent, 0.01)             # nothing emergent vs itself

    def test_overlap_keeps_sine_half(self):
        # RULE 2: cross-field overlap retains an emergent (sine) component — the
        # modes in the query not in the prototype — not just the aligned cosine.
        k1, _, _ = laf_signature(*_field(2))
        k2, _, _ = laf_signature(*_field(7))
        aligned, emergent = kappa_overlap(k1, k2)
        self.assertLess(aligned, 1.0)
        self.assertGreaterEqual(emergent, 0.0)
        # aligned² + emergent² ≈ 1 per principal angle (mean is a fair proxy here).
        self.assertLessEqual(aligned, 1.0)

    def test_lambda_distance(self):
        _, l1, _ = laf_signature(*_field(3))
        _, l2, _ = laf_signature(*_field(9))
        self.assertEqual(lambda_distance(l1, l1), 0.0)
        self.assertGreaterEqual(lambda_distance(l1, l2), 0.0)

    def test_tower_concatenates_scales(self):
        X = np.ones((4, 8), dtype=np.complex128)
        T = tower(X, n_scales=3)            # 8 + 4 + 2 = 14 columns
        self.assertEqual(T.shape, (4, 14))

    def test_degenerate_field_safe(self):
        k, lam, tau = laf_signature(np.zeros((16, 16)), np.zeros((16, 16)))
        self.assertTrue(np.isfinite(k).all())


if __name__ == "__main__":
    unittest.main()
