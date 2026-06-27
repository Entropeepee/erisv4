"""§5: harness — within/cross phases, report writing, resume cache — end-to-end on
a tiny synthetic dataset (no network)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import json
import tempfile
import unittest
import numpy as np

from eris.experiments.zero_weight_vision import VisionConfig
from eris.experiments.vision_harness import (
    run_within, run_cross, write_report, load_dataset, SignatureCache,
)


def _make_dataset(root, n=8, size=24):
    """Two visually distinct classes (horizontal vs vertical stripes) as .npy."""
    for label, orient in (("car", "h"), ("ship", "v")):
        cdir = os.path.join(root, label)
        os.makedirs(cdir, exist_ok=True)
        for i in range(n):
            rng = np.random.RandomState(hash((label, i)) % 2**32)
            base = np.zeros((size, size, 3))
            if orient == "h":
                base[::3, :, :] = 1.0
            else:
                base[:, ::3, :] = 1.0
            img = np.clip(base + 0.05 * rng.randn(size, size, 3), 0, 1) * 255
            np.save(os.path.join(cdir, f"{i:03d}.npy"), img.astype(np.uint8))


class TestHarness(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        _make_dataset(self.root)
        self.cfg = VisionConfig(size=24)

    def test_load_dataset(self):
        imgs, labels = load_dataset(self.root)
        self.assertEqual(len(imgs), 16)
        self.assertEqual(set(labels), {"car", "ship"})

    def test_run_within_beats_random(self):
        res = run_within(self.root, self.cfg)
        self.assertIn("zero_weight_accuracy", res)
        self.assertGreaterEqual(res["zero_weight_accuracy"], res["random_baseline"])
        self.assertTrue(0.0 <= res["unknown_rate"] <= 1.0)

    def test_run_cross_well_formed(self):
        res = run_cross(self.root, self.cfg)
        self.assertIn("ablation", res)
        for m in ("aligned", "tension", "both"):
            self.assertTrue(0.0 <= res["ablation"][m] <= 1.0)
        self.assertTrue(0.0 <= res["permutation_p"] <= 1.0)
        self.assertIn("keeps_lambda_helps", res)

    def test_write_report(self):
        out = tempfile.mkdtemp()
        path = write_report({"within": {"zero_weight_accuracy": 0.7}}, out_dir=out)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            self.assertEqual(json.load(f)["within"]["zero_weight_accuracy"], 0.7)

    def test_signature_cache_resume(self):
        p = os.path.join(tempfile.mkdtemp(), "cache.txt")
        c = SignatureCache(p)
        self.assertFalse(c.has("abc"))
        c.mark("abc")
        self.assertTrue(SignatureCache(p).has("abc"))   # reloads marker from disk


if __name__ == "__main__":
    unittest.main()
