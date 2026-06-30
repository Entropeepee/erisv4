"""Phase-2: torus topology + soft Boundary (the reflecting-edge → torus reversal).

- The field is a finite TORUS: periodic stencils wrap, no Dirichlet wall (no zeroed border).
- AMPLITUDE bound is a SOFT ceiling (softplus, C¹, no kink → τ stays clean) + a hard non-negativity
  floor; φ stays strictly inside (0, B_max) so the IBT/ZBT barriers never reach their poles.
- The Boundary BVec domain is a value-based membrane measure (φ − mean φ), not the grid edge.
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import warnings
import unittest

import numpy as np

from eris.config import to_numpy
from eris.field.pde import FractalField, _soft_clamp, _lap


class TestSoftClamp(unittest.TestCase):
    def test_in_band_is_near_identity(self):
        x = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32)
        out = to_numpy(_soft_clamp(x, 0.0, 0.9999))
        np.testing.assert_allclose(out, x, atol=0.02)

    def test_bounded_into_floor_and_ceiling(self):
        hi = 0.9999                                   # = B_max − 1e-4, safely below the IBT pole (B_max+δ)
        x = np.array([-5.0, -0.1, 0.0, 0.5, 0.9999, 2.0, 50.0], dtype=np.float32)
        out = to_numpy(_soft_clamp(x, 0.0, hi))
        self.assertTrue((out >= 0.0).all(), f"floor violated: {out}")
        self.assertTrue((out <= hi + 1e-6).all(), f"ceiling violated: {out}")
        self.assertLess(float(out.max()), 1.0)        # strictly below B_max → IBT denominator stays positive

    def test_ceiling_is_smooth_no_hard_kink(self):
        # crossing the ceiling, the derivative is continuous (a hard clip would jump to exactly 0)
        x = np.linspace(0.97, 1.03, 200).astype(np.float32)
        out = to_numpy(_soft_clamp(x, 0.0, 0.9999))
        d = np.diff(out)
        self.assertTrue((d >= -1e-6).all(), "monotone non-decreasing")
        self.assertLess(np.max(np.abs(np.diff(d))), 0.02, "second difference small → no kink")


class TestTorusTopology(unittest.TestCase):
    def test_periodic_laplacian_couples_opposite_edges(self):
        # a single hot cell on the LEFT edge: the periodic _lap reaches its left neighbour, which is
        # the RIGHT edge (wrap) — proof the topology is a torus, not a walled box.
        a = np.zeros((8, 8), dtype=np.float32)
        a[4, 0] = 1.0
        lap = to_numpy(_lap(a))
        self.assertNotEqual(lap[4, -1], 0.0, "left edge did not couple to the right edge (no wrap)")

    def test_no_zeroed_border_after_steps(self):
        f = FractalField(size=32)
        f.seed_from_text("a thought that lives at the edge")
        for _ in range(40):
            f.step()
        phi = to_numpy(f.phi)
        self.assertGreater(phi[0, :].max(), 0.0)     # border evolves like the interior
        self.assertGreater(phi[:, -1].max(), 0.0)

    def test_field_stays_finite_and_bounded_over_a_long_run(self):
        warnings.simplefilter("ignore")
        f = FractalField(size=48)
        f.seed_from_text("coherence, torsion, decay — round and round the torus")
        for _ in range(200):
            f.step()
        phi, theta, tau = to_numpy(f.phi), to_numpy(f.theta), to_numpy(f.tau)
        self.assertTrue(np.isfinite(phi).all() and np.isfinite(theta).all() and np.isfinite(tau).all())
        self.assertGreaterEqual(phi.min(), 0.0)
        self.assertLess(phi.max(), f.p.B_max)


if __name__ == "__main__":
    unittest.main()
