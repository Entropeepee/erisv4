"""§3: zero-weight classifier — deterministic centroids, two-channel score +
Hill-Power, SGT unknown gate, blade-control zero-contribution, and the κ+Λ
separability receipt (RULE 2: separation comes from the tension channel; cos-only
can barely tell the classes apart)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.experiments.zero_weight_vision import (
    VisionConfig, Signature, ClassPrototype, ZeroWeightClassifier,
    compute_signature, build_prototype, raw_score,
)
from eris.vision.coupling import field_coupling, UnknownGate


def _stripe_image(orient, seed, size=24):
    rng = np.random.RandomState(seed)
    base = np.zeros((size, size))
    if orient == "h":
        base[::3, :] = 1.0
    else:
        base[:, ::3] = 1.0
    img = (base + 0.05 * rng.randn(size, size))
    return np.clip(img, 0, 1)


class TestZeroWeightClassifier(unittest.TestCase):
    def test_prototype_is_deterministic_no_fitting(self):
        cfg = VisionConfig(size=24)
        sigs = [compute_signature(_stripe_image("h", s), cfg) for s in range(4)]
        p1 = build_prototype("h", sigs, cfg)
        p2 = build_prototype("h", sigs, cfg)
        np.testing.assert_allclose(p1.mag, p2.mag)
        np.testing.assert_allclose(p1.lam, p2.lam)

    def test_separates_two_classes_above_random(self):
        cfg = VisionConfig(size=24)
        train = {"h": [compute_signature(_stripe_image("h", s), cfg) for s in range(5)],
                 "v": [compute_signature(_stripe_image("v", s + 50), cfg) for s in range(5)]}
        clf = ZeroWeightClassifier(cfg).fit(train)
        correct = 0
        for s in range(20):
            label = "h" if s % 2 == 0 else "v"
            img = _stripe_image(label, s + 200)
            pred, _ = clf.predict(compute_signature(img, cfg))
            correct += (pred == label)
        self.assertGreater(correct, 13)            # well above 10/20 chance

    def test_blade_control_zero_contribution(self):
        cfg = VisionConfig(size=24)
        train = {"h": [compute_signature(_stripe_image("h", s), cfg) for s in range(3)],
                 "v": [compute_signature(_stripe_image("v", s + 50), cfg) for s in range(3)]}
        clf = ZeroWeightClassifier(cfg).fit(train)
        clf.deactivate("h")
        q = compute_signature(_stripe_image("h", 999), cfg)
        pred, scores = clf.predict(q)
        self.assertNotIn("h", scores)              # deactivated → never scored
        self.assertNotEqual(pred, "h")             # → never predicted (geometric absence)

    def test_unknown_gate(self):
        g = UnknownGate(threshold_sigma=1.0, warmup=6)
        for _ in range(10):
            self.assertTrue(g.is_known(1.0))       # establish floor near 1.0
        self.assertFalse(g.is_known(-5.0))         # far below floor → unknown
        self.assertTrue(g.is_known(1.0))           # back at floor → known

    def test_kappa_lambda_separability_rule2(self):
        # Two classes with (near-)IDENTICAL elastic coupling to a query but very
        # different PLASTIC (tension). Both-channels separates them by a wide
        # margin; cos-only can barely tell them apart → the discrimination lives
        # in the sine/tension half (fails if anyone drops it).
        size = 8
        n = size * size
        ones = np.ones((size, size))
        q = Signature(mag=ones.copy(), theta=np.zeros((size, size)),
                      kappa=np.zeros((1, 1)), lam=np.zeros(1), tau_stats=np.zeros(3))
        # A: fully aligned (θ=0), magnitude on 36% of pixels → elastic≈0.6, plastic 0.
        magA = np.zeros(n); magA[:int(0.36 * n)] = 1.0
        A = ClassPrototype("A", magA.reshape(size, size), np.zeros((size, size)),
                           np.zeros((1, 1)), np.zeros(1), np.zeros(3))
        # B: full magnitude, 40% of pixels at Δθ=90° → elastic≈0.6, plastic≈0.4.
        thB = np.zeros(n); thB[:int(0.40 * n)] = np.pi / 2
        B = ClassPrototype("B", ones.copy(), thB.reshape(size, size),
                           np.zeros((1, 1)), np.zeros(1), np.zeros(3))
        eA, pA, _ = field_coupling(q.mag, q.theta, A.mag, A.theta)
        eB, pB, _ = field_coupling(q.mag, q.theta, B.mag, B.theta)
        self.assertLess(abs(eA - eB), 0.06)        # same ALIGNED channel
        self.assertGreater(pB - pA, 0.2)           # different TENSION channel

        cfg_both = VisionConfig(w_field=1.0, w_kappa=0.0, w_lambda=0.0, two_channel=True)
        cfg_cos = VisionConfig(w_field=1.0, w_kappa=0.0, w_lambda=0.0, two_channel=False)
        margin_both = abs(raw_score(q, A, cfg_both) - raw_score(q, B, cfg_both))
        margin_cos = abs(raw_score(q, A, cfg_cos) - raw_score(q, B, cfg_cos))
        self.assertGreater(margin_both, 5 * margin_cos)   # tension carries the signal
        # And both-channels prefers the low-tension class.
        clf = ZeroWeightClassifier(cfg_both); clf.protos = {"A": A, "B": B}
        pred, _ = clf.predict(q)
        self.assertEqual(pred, "A")


if __name__ == "__main__":
    unittest.main()
