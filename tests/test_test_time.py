"""Tests for the test-time-compute layer (roadmap 0.3)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import unittest

from eris.interface.test_time import self_consistent_generate, budget_forced_generate


class _Resp:
    def __init__(self, text):
        self.text = text
        self.provider = "fake"
        self.model = "fake-1"


class _Same:
    """Deterministic backend: always the same answer (trivially converged)."""
    def is_available(self):
        return True

    async def generate(self, prompt="", system="", max_tokens=8192, temperature=0.7):
        return _Resp("The answer is 42, and here is a stable explanation.")


class _Varied:
    """Backend returning maximally different answers each call (never converges)."""
    def __init__(self):
        self.i = 0
        self.texts = [
            "alpha beta gamma delta one two three",
            "ocean mountain river desert tundra reef",
            "quantum lattice phonon boson fermion gauge",
            "saffron cardamom turmeric paprika cumin clove",
            "Helsinki Nairobi Quito Reykjavik Ulaanbaatar",
            "violin cello oboe timpani celesta theremin",
            "granite basalt obsidian gneiss schist marble",
            "nebula quasar pulsar magnetar blazar comet",
        ]

    def is_available(self):
        return True

    async def generate(self, prompt="", system="", max_tokens=8192, temperature=0.7):
        t = self.texts[self.i % len(self.texts)]
        self.i += 1
        return _Resp(t)


class TestSelfConsistency(unittest.TestCase):
    def test_converged_answers_stop_at_min(self):
        resp, n = asyncio.run(self_consistent_generate(
            _Same(), "q", min_samples=3, max_samples=8))
        self.assertIsNotNone(resp)
        self.assertEqual(n, 3)                      # trivially converged -> floor only
        self.assertIn("42", resp.text)

    def test_divergent_answers_do_not_early_stop(self):
        resp, n = asyncio.run(self_consistent_generate(
            _Varied(), "q", min_samples=3, max_samples=8))
        self.assertIsNotNone(resp)
        self.assertGreater(n, 3)                    # never converged -> past the floor

    def test_returns_medoid_membership(self):
        backend = _Varied()
        resp, n = asyncio.run(self_consistent_generate(
            backend, "q", min_samples=3, max_samples=5))
        # The consensus must be one of the actually-sampled texts.
        self.assertIn(resp.text, backend.texts)

    def test_no_backend_returns_none(self):
        class _Empty:
            def is_available(self): return True
            async def generate(self, *a, **k): return _Resp("")
        resp, n = asyncio.run(self_consistent_generate(
            _Empty(), "q", min_samples=2, max_samples=4))
        self.assertIsNone(resp)
        self.assertEqual(n, 0)


class TestBudgetForcing(unittest.TestCase):
    def test_extends_until_budget(self):
        class _Short:
            def __init__(self): self.calls = 0
            def is_available(self): return True
            async def generate(self, prompt="", system="", max_tokens=8192, temperature=0.7):
                self.calls += 1
                return _Resp("short.")
        b = _Short()
        resp, calls = asyncio.run(budget_forced_generate(
            b, "q", min_thinking_tokens=1000, max_extensions=2))
        self.assertEqual(calls, 3)                  # 1 initial + 2 "Wait" extensions
        self.assertGreater(len(resp.text), len("short."))

    def test_no_forcing_when_budget_zero(self):
        class _One:
            def __init__(self): self.calls = 0
            def is_available(self): return True
            async def generate(self, prompt="", system="", max_tokens=8192, temperature=0.7):
                self.calls += 1
                return _Resp("done")
        b = _One()
        resp, calls = asyncio.run(budget_forced_generate(b, "q", min_thinking_tokens=0))
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
