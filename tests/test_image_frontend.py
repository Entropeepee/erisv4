"""§2: image frontend (gradient spinor), torsion (RULE 3 guard), two-channel
coupling (RULE 2), GLNCS debias. Offline, synthetic, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.knowledge.frontends import (
    ImageFrontend, image_density_phase, torsion,
)
from eris.vision.coupling import field_coupling, coupling_score, FieldDebias


class TestImageFrontend(unittest.TestCase):
    def test_to_field_shape_and_range(self):
        img = (np.random.RandomState(0).rand(40, 50, 3) * 255).astype(np.uint8)
        mag, theta = ImageFrontend().to_field(img, size=32)
        self.assertEqual(mag.shape, (32, 32))
        self.assertEqual(theta.shape, (32, 32))
        self.assertTrue(np.isfinite(mag).all() and np.isfinite(theta).all())
        self.assertLessEqual(mag.max(), 1.0 + 1e-5)      # mag = √ρ, ρ∈[0,1]
        self.assertGreaterEqual(mag.min(), 0.0)

    def test_density_phase_ranges(self):
        g = np.random.RandomState(1).rand(24, 24)
        rho, th = image_density_phase(g)
        self.assertAlmostEqual(float(rho.min()), 0.0, places=5)
        self.assertAlmostEqual(float(rho.max()), 1.0, places=5)
        self.assertTrue((th >= -np.pi - 1e-6).all() and (th <= np.pi + 1e-6).all())


def _twist(perpendicular, seed, size=24):
    # θ is IDENTICAL across both classes (ramp in x), so the Laplacian of θ — which
    # never sees ρ — is blind to the class. The classes differ ONLY in how ρ twists
    # against θ (∥ vs ⊥), which only ∇ρ×∇θ can detect.
    rng = np.random.RandomState(seed)
    g = np.linspace(0, 2 * np.pi, size)
    theta = np.tile(g, (size, 1)) + 0.02 * rng.randn(size, size)       # θ ramps in x
    rho = (np.tile(g[:, None] / (2 * np.pi), (1, size)) if perpendicular  # ρ ramps in y (⊥)
           else np.tile(g / (2 * np.pi), (size, 1)))                   # ρ ramps in x (∥)
    rho = rho + 0.02 * rng.randn(size, size)
    return rho, theta


def _laplacian(theta):
    return (np.gradient(np.gradient(theta, axis=0), axis=0)
            + np.gradient(np.gradient(theta, axis=1), axis=1))


class TestTorsionRule3Guard(unittest.TestCase):
    def test_cross_product_separates_more_than_laplacian(self):
        # RULE 3: τ=∇ρ×∇θ must discriminate a ρ-θ twist better than the Laplacian
        # ∇²θ the Chimera prototype used. Two classes: aligned ∇ρ∥∇θ (low torsion)
        # vs perpendicular ∇ρ⊥∇θ (high torsion). The cross product separates them;
        # the Laplacian of a (near-)linear phase ramp is ~0 for BOTH → can't.
        def cross_stat(r, t):
            return float(np.mean(np.abs(torsion(r, t))))

        def lap_stat(r, t):
            return float(np.mean(np.abs(_laplacian(t) / (r + 1e-6))))

        A_cross = [cross_stat(*_twist(False, s)) for s in range(6)]
        B_cross = [cross_stat(*_twist(True, s + 100)) for s in range(6)]
        A_lap = [lap_stat(*_twist(False, s)) for s in range(6)]
        B_lap = [lap_stat(*_twist(True, s + 100)) for s in range(6)]

        def disc(a, b):                  # class gap relative to typical magnitude
            return abs(np.mean(a) - np.mean(b)) / (np.mean(np.abs(a + b)) + 1e-9)

        disc_cross, disc_lap = disc(A_cross, B_cross), disc(A_lap, B_lap)
        # The cross product is a MARKEDLY stronger discriminator (the 28× finding
        # encoded as a direction, not a magic number) — fails if anyone swaps in
        # the Laplacian, which can't see ρ at all.
        self.assertGreater(disc_cross, disc_lap)
        self.assertGreater(disc_cross, 3.0 * disc_lap)


class TestTwoChannelCoupling(unittest.TestCase):
    def test_identical_fields_resonate_no_tension(self):
        mag = np.random.RandomState(2).rand(20, 20) + 0.5
        th = np.random.RandomState(3).rand(20, 20) * np.pi
        e, p, sine = field_coupling(mag, th, mag, th)
        self.assertGreater(e, 0.0)                # in-phase → elastic
        self.assertLess(p, 1e-9)                  # Δθ=0 → sin²=0 → no plastic
        self.assertLess(sine, 1e-9)

    def test_quarter_phase_offset_creates_tension(self):
        mag = np.ones((16, 16))
        th = np.zeros((16, 16))
        e, p, sine = field_coupling(mag, th, mag, th + np.pi / 2)   # Δθ=90°
        self.assertLess(e, 1e-6)                  # cos²(90°)=0 → no elastic
        self.assertGreater(p, 0.5)                # sin²(90°)=1 → all plastic
        self.assertGreater(sine, 0.9)             # mean|sinΔθ|≈1
        self.assertLess(coupling_score(mag, th, mag, th + np.pi / 2), 0.0)  # net negative


class TestFieldDebias(unittest.TestCase):
    def test_strips_shared_nuisance_direction(self):
        size = 16
        rng = np.random.RandomState(0)
        v = rng.randn(size * size); v /= np.linalg.norm(v)     # shared nuisance dir
        mags = [(0.1 * rng.randn(size * size) + 4.0 * rng.randn() * v).reshape(size, size)
                for _ in range(16)]
        deb = FieldDebias(size, bias_fraction=0.1).fit(mags)
        before = abs(mags[0].ravel() @ v)
        after = abs(deb.apply(mags[0]).ravel() @ v)
        self.assertLess(after, 0.3 * before)      # cross-class nuisance removed


if __name__ == "__main__":
    unittest.main()
