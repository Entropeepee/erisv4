"""§3b/§5: audio + grounding harness — within-modal run, grounding run (all pairs +
zero-shot + transitivity), report IO, dataset load. Offline, synthetic, fast config."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import json
import tempfile
import unittest
import numpy as np

from eris.experiments.zero_weight_vision import VisionConfig as FieldConfig
from eris.experiments.grounding import GroundingConfig
from eris.experiments.audio_harness import (
    run_audio_within, run_grounding, write_report, load_audio_dataset, _by_class,
)

SR = 16000


def _band(kind, seed, dur=0.4):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freqs = (rng.uniform(200, 500, 3) if kind == "low" else rng.uniform(2500, 4000, 3))
    return sum(np.sin(2 * np.pi * f * t) for f in freqs) + 0.05 * rng.randn(t.size)


def _field(seed, shift):
    rng = np.random.RandomState(seed)
    mag = np.clip(np.roll(rng.rand(16, 16), shift, axis=0), 0, 1)
    theta = (rng.rand(16, 16) - 0.5) * 2 * np.pi
    return mag, theta


class TestAudioHarness(unittest.TestCase):
    def test_run_audio_within_beats_random(self):
        cfg = FieldConfig(size=24)
        samples = {"low": [_band("low", s) for s in range(6)],
                   "high": [_band("high", s + 50) for s in range(6)]}
        res = run_audio_within(samples, cfg)
        self.assertIn("zero_weight_accuracy", res)
        self.assertGreaterEqual(res["zero_weight_accuracy"], res["random_baseline"])
        self.assertTrue(0.0 <= res["unknown_rate"] <= 1.0)

    def test_run_grounding_well_formed(self):
        gcfg = GroundingConfig(descriptor_grid=4, Ns=(0, 4), n_repeat=4, seed=0)
        # three concepts, distinct synthetic fields per modality
        cls = ("a", "b", "c")
        audio = {c: [_field(hash((c, "au", i)) % 999, sh) for i in range(4)]
                 for c, sh in zip(cls, (0, 5, 10))}
        image = {c: [_field(hash((c, "im", i)) % 999, sh) for i in range(4)]
                 for c, sh in zip(cls, (1, 6, 11))}
        word = {c: [_field(hash((c, "wd", i)) % 999, sh) for i in range(4)]
                for c, sh in zip(cls, (2, 7, 12))}
        res = run_grounding(audio, image, word, gcfg)
        self.assertIn("audio->word", res["pairs"])
        self.assertIn("image->word", res["pairs"])
        for name, pr in res["pairs"].items():
            self.assertIn("unitary", pr)
            self.assertIn("shuffled", pr)
            self.assertIn("cosine_fit", pr)
            self.assertTrue(0.0 <= pr["ceiling_src"] <= 1.0)
        for name, zs in res["zero_shot"].items():
            self.assertTrue(0.0 <= zs["acc"] <= 1.0)
            self.assertTrue(0.0 <= zs["p"] <= 1.0)
        self.assertIn("transitivity", res)

    def test_run_audio_within_guards_degenerate_data(self):
        cfg = FieldConfig(size=24)
        self.assertIn("error", run_audio_within({}, cfg))                 # empty
        self.assertIn("error", run_audio_within({"low": [_band("low", 0)]}, cfg))  # 1 class

    def test_grounding_report_includes_lowlevel_probe(self):
        gcfg = GroundingConfig(descriptor_grid=4, Ns=(0, 4), n_repeat=3, seed=0)
        cls = ("a", "b")
        audio = {c: [_field(hash((c, "au", i)) % 999, sh) for i in range(4)]
                 for c, sh in zip(cls, (0, 6))}
        word = {c: [_field(hash((c, "wd", i)) % 999, sh) for i in range(4)]
                for c, sh in zip(cls, (2, 8))}
        res = run_grounding(audio, {}, word, gcfg)
        zs = res["zero_shot"]["audio->word"]
        self.assertIn("lowlevel_probe", zs)
        self.assertIn("residual_mean", zs["lowlevel_probe"])

    def test_write_report_roundtrip(self):
        out = tempfile.mkdtemp()
        path = write_report({"audio_within": {"zero_weight_accuracy": 0.8}}, out_dir=out)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            self.assertEqual(json.load(f)["audio_within"]["zero_weight_accuracy"], 0.8)

    def test_load_audio_dataset(self):
        root = tempfile.mkdtemp()
        for label in ("dog", "cat"):
            d = os.path.join(root, label); os.makedirs(d)
            for i in range(3):
                np.save(os.path.join(d, f"{i}.npy"), _band("low", i))
        xs, ys = load_audio_dataset(root)
        self.assertEqual(len(xs), 6)
        self.assertEqual(set(ys), {"dog", "cat"})
        self.assertEqual(set(_by_class(xs, ys)), {"dog", "cat"})


if __name__ == "__main__":
    unittest.main()
