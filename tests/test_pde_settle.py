"""Stage-2 convergence early-stop: FractalField.run_settled suspends once the field reaches its
attractor (coherence change falls below a relative tol), with a hard min-steps floor — so the
retrieval rerank PDE doesn't always pay the full 50 steps. Offline, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.field.pde import FractalField


class TestRunSettled(unittest.TestCase):
    def test_stops_within_bounds(self):
        f = FractalField(size=16, seed=42)
        f.seed_from_text("boundary limited exchange critical dynamics")
        n = f.run_settled(max_steps=50, min_steps=8, check_every=4)
        self.assertGreaterEqual(n, 8)          # never under the floor
        self.assertLessEqual(n, 50)            # never over the cap

    def test_min_steps_floor_respected(self):
        # even with a trivially-loose tol that would suspend immediately, min_steps holds
        f = FractalField(size=16, seed=42)
        f.seed_from_text("hello world")
        n = f.run_settled(max_steps=50, min_steps=12, check_every=4, tol=1e9)
        self.assertGreaterEqual(n, 12)

    def test_settle_actually_saves_steps_on_a_settling_field(self):
        # a field that settles should stop before the cap (the whole point of the early-stop)
        f = FractalField(size=16, seed=42)
        f.seed_from_text("a calm steady phrase that settles quickly")
        n = f.run_settled(max_steps=200, min_steps=8, check_every=4, tol=0.02)
        self.assertLess(n, 200)                # genuinely terminated early

    def test_to_field_evolved_settle_matches_shape(self):
        from eris.tribe.specialists import _text_to_field
        phi, theta = _text_to_field("criticality boundary coupling", field_size=16)
        self.assertEqual(phi.shape, (16, 16))
        self.assertEqual(theta.shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
