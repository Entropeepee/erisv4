"""§3: zero-weight AUDIO classifier — deterministic prototypes, two-channel score,
SGT unknown gate, blade-control, and the κ+Λ two-channel receipt — all reusing the #41
field machinery through AudioFrontend. Offline, synthetic tones, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.experiments.zero_weight_audio import (
    AudioConfig, compute_audio_signature, mfcc_features, logmel_features,
)
from eris.experiments.zero_weight_vision import (
    Signature, ClassPrototype, ZeroWeightClassifier, build_prototype, raw_score,
)
from eris.vision.coupling import field_coupling, UnknownGate

SR = 16000


def _band(kind, seed, dur=0.5, sr=SR):
    """Synthetic class signal: 'low' = a few tones in 200–500 Hz, 'high' = 2.5–4 kHz,
    with per-seed jitter + light noise so prototypes are non-degenerate."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freqs = (rng.uniform(200, 500, 3) if kind == "low" else rng.uniform(2500, 4000, 3))
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return x + 0.05 * rng.randn(t.size)


class TestZeroWeightAudio(unittest.TestCase):
    def test_prototype_is_deterministic_no_fitting(self):
        cfg = AudioConfig(size=24)
        sigs = [compute_audio_signature(_band("low", s), cfg) for s in range(4)]
        p1 = build_prototype("low", sigs, cfg)
        p2 = build_prototype("low", sigs, cfg)
        np.testing.assert_allclose(p1.mag, p2.mag)
        np.testing.assert_allclose(p1.lam, p2.lam)

    def test_separates_two_classes_above_random(self):
        cfg = AudioConfig(size=24)
        train = {"low": [compute_audio_signature(_band("low", s), cfg) for s in range(5)],
                 "high": [compute_audio_signature(_band("high", s + 50), cfg) for s in range(5)]}
        clf = ZeroWeightClassifier(cfg).fit(train)
        correct = 0
        for s in range(20):
            label = "low" if s % 2 == 0 else "high"
            sig = compute_audio_signature(_band(label, s + 200), cfg)
            pred, _ = clf.predict(sig)
            correct += (pred == label)
        self.assertGreater(correct, 13)            # well above 10/20 chance

    def test_blade_control_zero_contribution(self):
        cfg = AudioConfig(size=24)
        train = {"low": [compute_audio_signature(_band("low", s), cfg) for s in range(3)],
                 "high": [compute_audio_signature(_band("high", s + 50), cfg) for s in range(3)]}
        clf = ZeroWeightClassifier(cfg).fit(train)
        clf.deactivate("low")
        q = compute_audio_signature(_band("low", 999), cfg)
        pred, scores = clf.predict(q)
        self.assertNotIn("low", scores)            # deactivated → never scored
        self.assertNotEqual(pred, "low")

    def test_unknown_gate(self):
        g = UnknownGate(threshold_sigma=1.0, warmup=6)
        for _ in range(10):
            self.assertTrue(g.is_known(1.0))
        self.assertFalse(g.is_known(-5.0))
        self.assertTrue(g.is_known(1.0))

    def test_kappa_lambda_separability_rule2(self):
        # Modality-agnostic receipt: two classes with ~identical ELASTIC coupling but
        # very different PLASTIC (tension). Both-channels separates by a wide margin;
        # cos-only can barely tell them apart → the signal lives in the sine half.
        size = 8
        n = size * size
        ones = np.ones((size, size))
        q = Signature(mag=ones.copy(), theta=np.zeros((size, size)),
                      kappa=np.zeros((1, 1)), lam=np.zeros(1), tau_stats=np.zeros(3))
        magA = np.zeros(n); magA[:int(0.36 * n)] = 1.0
        A = ClassPrototype("A", magA.reshape(size, size), np.zeros((size, size)),
                           np.zeros((1, 1)), np.zeros(1), np.zeros(3))
        thB = np.zeros(n); thB[:int(0.40 * n)] = np.pi / 2
        B = ClassPrototype("B", ones.copy(), thB.reshape(size, size),
                           np.zeros((1, 1)), np.zeros(1), np.zeros(3))
        eA, pA, _ = field_coupling(q.mag, q.theta, A.mag, A.theta)
        eB, pB, _ = field_coupling(q.mag, q.theta, B.mag, B.theta)
        self.assertLess(abs(eA - eB), 0.06)
        self.assertGreater(pB - pA, 0.2)
        cfg_both = AudioConfig(w_field=1.0, w_kappa=0.0, w_lambda=0.0, two_channel=True)
        cfg_cos = AudioConfig(w_field=1.0, w_kappa=0.0, w_lambda=0.0, two_channel=False)
        margin_both = abs(raw_score(q, A, cfg_both) - raw_score(q, B, cfg_both))
        margin_cos = abs(raw_score(q, A, cfg_cos) - raw_score(q, B, cfg_cos))
        self.assertGreater(margin_both, 5 * margin_cos)

    def test_ablation_features_are_finite_fixed_length(self):
        x = _band("low", 7)
        self.assertTrue(np.isfinite(mfcc_features(x)).all())
        self.assertTrue(np.isfinite(logmel_features(x)).all())
        # fixed length regardless of clip duration (mean+std aggregation)
        self.assertEqual(mfcc_features(_band("low", 1, dur=0.3)).shape,
                         mfcc_features(_band("low", 2, dur=0.7)).shape)


if __name__ == "__main__":
    unittest.main()
