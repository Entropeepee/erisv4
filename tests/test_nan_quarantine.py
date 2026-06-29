"""Phase-2 Batch 1 — NaN/Inf quarantine (Codex #9).

A non-finite φ/θ/τ must NEVER reach BVec / gates / routing / memory. The guard catches it at the
two gateways — compute_bvec_from_field (the BVec boundary) and FractalField.step (field integrity).
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import math
import unittest

import numpy as np

from eris.config import to_numpy
from eris.computation.activations import compute_bvec_from_field, quarantine_nonfinite


def _all_finite_bvec(bv) -> bool:
    return all(math.isfinite(v) for v in (bv.B, bv.F, bv.E, bv.C, bv.D, bv.S))


class TestQuarantineHelper(unittest.TestCase):
    def test_replaces_nan_and_inf(self):
        a = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
        clean, n_bad = quarantine_nonfinite(a, 0.0, "x")
        self.assertEqual(n_bad, 3)
        self.assertTrue(np.all(np.isfinite(to_numpy(clean))))
        self.assertEqual(float(to_numpy(clean)[0, 0]), 1.0)   # finite value untouched

    def test_clean_array_unchanged(self):
        a = np.ones((4, 4), dtype=np.float32)
        clean, n_bad = quarantine_nonfinite(a, 0.0, "x")
        self.assertEqual(n_bad, 0)


class TestBVecGateway(unittest.TestCase):
    def test_nan_theta_does_not_reach_bvec(self):
        n = 16
        phi = np.full((n, n), 0.3, dtype=np.float32)
        theta = np.zeros((n, n), dtype=np.float32)
        tau = np.zeros((n, n), dtype=np.float32)
        phi_prev = np.full((n, n), 0.25, dtype=np.float32)
        theta[0, 0] = np.nan                                  # inject the corruption
        bv = compute_bvec_from_field(phi, theta, tau, phi_prev)
        self.assertTrue(_all_finite_bvec(bv), f"BVec leaked non-finite: {bv}")

    def test_inf_tau_does_not_reach_bvec(self):
        n = 16
        phi = np.full((n, n), 0.3, dtype=np.float32)
        tau = np.zeros((n, n), dtype=np.float32)
        tau[5, 5] = np.inf
        bv = compute_bvec_from_field(phi, np.zeros((n, n), np.float32), tau,
                                     np.full((n, n), 0.2, np.float32))
        self.assertTrue(_all_finite_bvec(bv))


class TestFieldStepIntegrity(unittest.TestCase):
    def test_injected_nan_is_cleaned_by_step_and_bvec_stays_finite(self):
        from eris.field.pde import FractalField
        f = FractalField(size=32)
        f.run(2)                                              # warm it up
        # inject corruption straight into the live field, then step
        f.theta[0, 0] = np.nan
        f.phi[1, 1] = np.inf
        f.step()
        self.assertTrue(np.all(np.isfinite(to_numpy(f.phi))), "phi carried NaN/inf past the step")
        self.assertTrue(np.all(np.isfinite(to_numpy(f.theta))))
        self.assertTrue(np.all(np.isfinite(to_numpy(f.tau))))
        bv = f.compute_bvec()
        self.assertTrue(_all_finite_bvec(bv))


if __name__ == "__main__":
    unittest.main()
