"""Phase-2 Batch 1 — DCR correctness (Codex #3 + #4).

#3: the field integral must be the RAW DCR integral (amplitude-preserving), not energy-normalized.
#4: a shape-mismatched field must be resampled onto a COMMON grid (not flatten-truncated to the
    wrong spatial region), and θ — a PHASE — must be aggregated CIRCULARLY, never arithmetically.
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import math
import unittest

import numpy as np

from eris.memory.interference import _field_integral
from eris.retrieval.field_interference import (
    field_resonance, circular_mean, resample_field, _common)


class TestRawFieldIntegral(unittest.TestCase):
    def test_amplitude_is_preserved(self):
        # identical θ; the raw integral must scale with φ² — φ=1 → 1, φ=2 → 4 (NOT both 1).
        theta = np.zeros((8, 8), dtype=np.float64)
        r1 = _field_integral(np.ones((8, 8)), theta, np.ones((8, 8)), theta)
        r2 = _field_integral(np.full((8, 8), 2.0), theta, np.full((8, 8), 2.0), theta)
        self.assertAlmostEqual(r1.total, 1.0, places=6)
        self.assertAlmostEqual(r2.total, 4.0, places=6)
        self.assertTrue(r1.used_field_integral)

    def test_antiphase_is_negative(self):
        a = np.zeros((8, 8)); b = np.full((8, 8), math.pi)      # cos(π) = −1
        r = _field_integral(np.ones((8, 8)), a, np.ones((8, 8)), b)
        self.assertAlmostEqual(r.total, -1.0, places=6)
        self.assertEqual(r.regime, "conflicting")

    def test_regime_stays_amplitude_invariant(self):
        # a weak but perfectly-aligned pair is still 'resonant' (regime normalized, total raw)
        theta = np.zeros((8, 8))
        r = _field_integral(np.full((8, 8), 0.01), theta, np.full((8, 8), 0.01), theta)
        self.assertEqual(r.regime, "resonant")
        self.assertAlmostEqual(r.total, 0.0001, places=8)      # raw, tiny


class TestCircularMean(unittest.TestCase):
    def test_branch_cut_block(self):
        # [0.01, 2π−0.01, 0.01, 2π−0.01] → circular mean ≈ 0, NOT π
        block = [0.01, 2 * math.pi - 0.01, 0.01, 2 * math.pi - 0.01]
        self.assertAlmostEqual(abs(circular_mean(block)), 0.0, places=3)
        # arithmetic mean would be ≈ π — prove we are NOT doing that
        self.assertAlmostEqual(float(np.mean(block)), math.pi, places=2)

    def test_resample_theta_is_circular(self):
        # a 2×2 θ block straddling the branch cut → downsample to 1×1 ≈ 0
        block = np.array([[0.01, 2 * math.pi - 0.01],
                          [0.01, 2 * math.pi - 0.01]])
        out = resample_field(block, (1, 1), circular=True)
        self.assertAlmostEqual(abs(float(out[0, 0])), 0.0, places=3)


class TestShapeSafeResonance(unittest.TestCase):
    def _signal_lower_right(self, n, k):
        """An n×n φ field with a constant signal only in the lower-right k×k corner."""
        f = np.zeros((n, n), dtype=np.float64)
        f[n - k:, n - k:] = 1.0
        return f

    def test_mismatched_grids_resample_to_common_region(self):
        # query: 128×128 with signal in the lower-right 96×96; stored = its own 32×32 downsample.
        q_phi = self._signal_lower_right(128, 96)
        q_theta = np.zeros((128, 128))
        s_phi = resample_field(q_phi, (32, 32))            # the stored snapshot
        s_theta = np.zeros((32, 32))

        # resonance of the 128 query vs its 32 downsample ≈ self-resonance of the downsample
        cross = field_resonance(q_phi, q_theta, s_phi, s_theta)
        selfr = field_resonance(s_phi, s_theta, s_phi, s_theta)
        self.assertGreater(cross, 0.3)                      # real signal, not washed out
        self.assertAlmostEqual(cross, selfr, places=6)      # same region compared

    def test_common_does_not_truncate_to_wrong_region(self):
        # the OLD flatten-truncate compared the first 1024 cells (top rows = empty) → ~0.
        q_phi = self._signal_lower_right(128, 96)
        s_phi = resample_field(q_phi, (32, 32))
        pq, tq, ps, ts = _common(q_phi, np.zeros((128, 128)), s_phi, np.zeros((32, 32)))
        self.assertEqual(pq.shape, ps.shape)                # brought to a common grid
        self.assertEqual(pq.shape, (32, 32))
        self.assertGreater(float(np.mean(pq * ps)), 0.1)    # signal survived the resample


if __name__ == "__main__":
    unittest.main()
