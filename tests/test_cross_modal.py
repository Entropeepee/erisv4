"""§4: cross-modal resonance — coupling, 3-way ablation, permutation null. Offline,
synthetic, deterministic. An identical-field match couples above the shuffled null."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.experiments.zero_weight_vision import Signature, VisionConfig
from eris.experiments.cross_modal import (
    cross_modal_score, classify, run_ablation, permutation_null, effect_size,
    word_signature,
)


def _sig(seed, size=12):
    rng = np.random.RandomState(seed)
    mag = np.clip(rng.rand(size, size), 0, 1)
    theta = (rng.rand(size, size) - 0.5) * 2 * np.pi
    return Signature(mag=mag, theta=theta, kappa=np.zeros((1, 1)),
                     lam=np.zeros(1), tau_stats=np.zeros(3))


class TestCrossModal(unittest.TestCase):
    def setUp(self):
        # Two distinct "word" fields; images are (noisy) copies of their word.
        self.words = {"car": _sig(1), "ship": _sig(2)}
        self.imgs, self.gold = [], []
        for i in range(20):
            label = "car" if i % 2 == 0 else "ship"
            w = self.words[label]
            rng = np.random.RandomState(100 + i)
            img = Signature(mag=np.clip(w.mag + 0.05 * rng.randn(*w.mag.shape), 0, 1),
                            theta=w.theta + 0.05 * rng.randn(*w.theta.shape),
                            kappa=np.zeros((1, 1)), lam=np.zeros(1), tau_stats=np.zeros(3))
            self.imgs.append(img); self.gold.append(label)

    def test_identical_field_couples_highest_to_itself(self):
        w = self.words["car"]
        self.assertGreater(cross_modal_score(w, self.words["car"], "both"),
                           cross_modal_score(w, self.words["ship"], "both"))

    def test_classify_matches_above_chance(self):
        preds = [classify(s, self.words, "both") for s in self.imgs]
        acc = np.mean([p == g for p, g in zip(preds, self.gold)])
        self.assertGreater(acc, 0.6)               # well above 0.5 chance

    def test_three_way_ablation_well_formed(self):
        ab = run_ablation(self.imgs, self.gold, self.words)
        for mode in ("aligned", "tension", "both"):
            self.assertIn("accuracy", ab[mode])
            self.assertTrue(0.0 <= ab[mode]["accuracy"] <= 1.0)
        self.assertAlmostEqual(ab["chance"], 0.5)

    def test_permutation_null_and_effect_size(self):
        preds = [classify(s, self.words, "both") for s in self.imgs]
        real, p = permutation_null(preds, self.gold, n_perm=500, seed=0)
        self.assertGreater(real, 0.6)
        self.assertLess(p, 0.05)                   # match couples above the shuffled null
        self.assertGreater(effect_size(self.imgs, self.gold, self.words), 0.0)

    def test_word_signature_real_text_path(self):
        # The real FRT/PDE text path produces a same-size (mag, θ) field + (κ, λ).
        cfg = VisionConfig(size=16)
        sig = word_signature("automobile", cfg)
        self.assertEqual(sig.mag.shape, (16, 16))
        self.assertEqual(sig.theta.shape, (16, 16))
        self.assertTrue(np.isfinite(sig.mag).all())


if __name__ == "__main__":
    unittest.main()
