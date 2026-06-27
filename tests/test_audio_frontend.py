"""§2: AudioFrontend — torch-free STFT → (mag=√ρ, θ), commensurable with image/word
fields, GLNCS floor removal, and the RULE-3 torsion guard (∇ρ×∇θ, not Laplacian).
Offline, synthetic (tones/chirps/noise), deterministic — no audio files, no scipy."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.knowledge.frontends import (
    AudioFrontend, ImageFrontend, audio_density_phase, torsion,
)
from eris.vision.coupling import coupling_score, FieldDebias


SR = 16000


def _tone(freq, dur=1.0, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def _chirp(f0, f1, dur=1.0, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    k = (f1 - f0) / dur
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


class TestAudioFrontend(unittest.TestCase):
    def test_shape_finite_and_commensurable_with_image(self):
        size = 32
        mag, theta = AudioFrontend().to_field(_tone(440), size=size)
        self.assertEqual(mag.shape, (size, size))
        self.assertEqual(theta.shape, (size, size))
        self.assertTrue(np.isfinite(mag).all() and np.isfinite(theta).all())
        self.assertGreaterEqual(float(mag.min()), 0.0)
        self.assertLessEqual(float(mag.max()), 1.0 + 1e-6)        # √ρ in [0,1]
        # same grid + normalization as an image field ⇒ cross-modal comparison valid
        imag, ith = ImageFrontend().to_field(np.random.RandomState(0).rand(20, 20), size=size)
        self.assertEqual(mag.shape, imag.shape)

    def test_density_phase_is_minmax_and_angle(self):
        rho, th = audio_density_phase(_tone(440))
        self.assertAlmostEqual(float(rho.min()), 0.0, places=6)
        self.assertLessEqual(float(rho.max()), 1.0 + 1e-9)
        self.assertLessEqual(float(np.abs(th).max()), np.pi + 1e-6)   # θ = angle ∈ [-π,π]

    def test_identical_tones_couple_higher_than_tone_vs_chirp(self):
        af = AudioFrontend()
        m1, t1 = af.to_field(_tone(440), size=32)
        m2, t2 = af.to_field(_tone(440), size=32)        # identical signal
        mc, tc = af.to_field(_chirp(200, 4000), size=32)  # sweeping, different structure
        same = coupling_score(m1, t1, m2, t2)
        diff = coupling_score(m1, t1, mc, tc)
        self.assertGreater(same, diff)

    def test_glncs_removes_injected_broadband_floor(self):
        # A broadband floor that VARIES in level across samples is the dominant
        # cross-class variance direction GLNCS should project out. (A constant floor
        # lives in the mean, which calibrate() centers away — so we vary its gain.)
        af = AudioFrontend()
        base = [af.to_field(_tone(f), size=24)[0].astype(np.float64)
                for f in (300, 600, 900, 1200)]
        F = np.random.RandomState(1).rand(24, 24)            # fixed broadband floor
        gains = [0.5, 1.5, 2.5, 3.5]
        noisy = [b + g * F for b, g in zip(base, gains)]
        deb = FieldDebias(24).fit(noisy)
        cleaned = deb.apply(noisy[0])
        fhat = F.ravel() / (np.linalg.norm(F.ravel()) + 1e-12)
        proj_noisy = abs(float(noisy[0].ravel() @ fhat))
        proj_clean = abs(float(cleaned.ravel() @ fhat))
        self.assertLess(proj_clean, 0.5 * proj_noisy)        # floor direction suppressed
        self.assertEqual(cleaned.shape, (24, 24))

    def test_rule3_torsion_is_cross_product_not_laplacian(self):
        # Two audio-like fields whose PHASE is identical (a pure x-ramp) and that differ
        # only in ρ-orientation (∥ vs ⊥ the θ-gradient). The Chimera ∇²θ "torsion" depends
        # ONLY on θ, so it is identical for both classes (genuinely blind); the #41
        # cross-product ∇ρ×∇θ uses ρ and so discriminates them (RULE 3).
        size = 24
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        theta = (xx / size) * 6.0                       # identical phase ramp both classes
        rho_par = xx / size                              # ρ varies along x (∥ θ-gradient)
        rho_perp = yy / size                             # ρ varies along y (⊥ θ-gradient)

        def _lap_torsion(rho, th):
            # the Laplacian-of-phase candidate ∇²θ — note it ignores ρ entirely
            return (np.gradient(np.gradient(th, axis=1), axis=1)
                    + np.gradient(np.gradient(th, axis=0), axis=0))

        def _disc(a, b):
            return abs(a.mean() - b.mean()) / (abs(a.mean()) + abs(b.mean()) + 1e-9)

        tau_par, tau_perp = torsion(rho_par, theta), torsion(rho_perp, theta)
        lap_par, lap_perp = _lap_torsion(rho_par, theta), _lap_torsion(rho_perp, theta)
        disc_cross = _disc(tau_par, tau_perp)            # cross-product SEES ρ-orientation
        disc_lap = _disc(lap_par, lap_perp)              # Laplacian is BLIND (θ-only)
        self.assertLess(disc_lap, 0.05)                  # ∇²θ cannot tell the classes apart
        self.assertGreater(disc_cross, 0.3)              # ∇ρ×∇θ clearly can
        self.assertGreater(disc_cross, 5.0 * disc_lap)   # and by a wide margin


if __name__ == "__main__":
    unittest.main()
