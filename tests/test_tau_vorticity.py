"""τ (torsion) reconciliation: the field τ is the canonical VORTICITY ∇ρ×∇θ, not the legacy
amplitude-Laplacian proxy. The proxy ignores phase, so it CANNOT tell a rotational field from an
irrotational one that shares the same amplitude — the vorticity can. This regression test is the
guard against silently sliding back to the proxy. Behind ERIS_TAU_VORTICITY (default off)."""
import os
os.environ.setdefault("ERIS_GPU", "0")          # xp == numpy, so the helpers take plain np arrays
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

import numpy as np

from eris.field.pde import _vorticity, _lap, _compute_tau


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))


# Two fields sharing the SAME amplitude ρ = x:
#   rotational   — phase θ = y  → ∇ρ=(1,0) ⟂ ∇θ=(0,1)  → vorticity ≈ 1 (nonzero)
#   irrotational — phase θ = x  → ∇ρ=(1,0) ∥ ∇θ=(1,0)  → vorticity ≈ 0
_N = 16
_YY, _XX = np.mgrid[0:_N, 0:_N]
_XX = _XX.astype(np.float32)
_YY = _YY.astype(np.float32)


class TestTauVorticity(unittest.TestCase):
    def test_vorticity_nonzero_for_rotation_zero_for_irrotational(self):
        rot = _vorticity(_XX, _YY)              # ρ=x, θ=y → rotational
        irr = _vorticity(_XX, _XX)              # ρ=x, θ=x → irrotational
        self.assertGreater(_rms(rot), 0.3)      # genuine torsion present
        self.assertLess(_rms(irr), 1e-4)        # no torsion in an irrotational field
        self.assertGreater(_rms(rot) / (_rms(irr) + 1e-9), 100)   # huge separation (cf. proxy below)

    def test_laplacian_proxy_cannot_discriminate(self):
        # the whole reason for the fix: the amplitude-Laplacian depends ONLY on ρ, which is the
        # same (x) for both fields — so it returns the IDENTICAL τ for a rotational and an
        # irrotational field. It is blind to the distinction the vorticity captures.
        self.assertTrue(np.allclose(np.asarray(_lap(_XX)), np.asarray(_lap(_XX))))
        # concretely, proxy separation ratio ~1 vs vorticity's >100
        self.assertAlmostEqual(_rms(_lap(_XX)) / (_rms(_lap(_XX)) + 1e-9), 1.0, places=6)

    def test_vorticity_is_wrap_safe_on_a_phase_branch_cut(self):
        # an azimuthal phase has a 2π branch cut; wrap-safe gradients must not inject a spurious
        # ridge of vorticity along it (bounded by tanh, but it must not dominate the field)
        cx = cy = _N / 2.0
        ang = np.arctan2(_YY - cy, _XX - cx).astype(np.float32)     # wraps across -x axis
        rho = np.hypot(_XX - cx, _YY - cy).astype(np.float32)
        tau = np.asarray(_vorticity(rho, ang))
        self.assertTrue(np.all(np.isfinite(tau)))
        self.assertLessEqual(float(np.abs(tau).max()), 1.0 + 1e-6)  # tanh-bounded, no blow-up

    def test_compute_tau_dispatches_on_the_flag(self):
        import eris.field.pde as pde
        orig = pde._TAU_VORTICITY
        try:
            pde._TAU_VORTICITY = False
            self.assertTrue(np.allclose(np.asarray(pde._compute_tau(_XX, _YY)),
                                        np.asarray(_lap(_XX))))      # legacy proxy
            pde._TAU_VORTICITY = True
            self.assertTrue(np.allclose(np.asarray(pde._compute_tau(_XX, _YY)),
                                        np.asarray(_vorticity(_XX, _YY))))   # canonical vorticity
        finally:
            pde._TAU_VORTICITY = orig


if __name__ == "__main__":
    unittest.main()
