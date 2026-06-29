"""Autonomous-loop cost guard: the idle dream loop's condense/refine step calls PAID cloud Claude
(ask_expert.ask). With ANTHROPIC_API_KEY set (for the interactive deep-mediator), the unattended
loop would spend money every crawl cycle. This is the TRACE David asked for — it monkeypatches the
paid call and asserts it does NOT fire by default (proving the call is gated, not just that a flag
exists), then that it fires when enabled and stops at the ceiling."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest


class _StubLoop:
    """Minimal self for _claude_condense_and_refine — only the attrs it touches."""
    _focus = None
    def _regime_feeling(self, regime, dominant_domains=None):
        return "calm"
    def _clean_topic_line(self, text):
        return (text or "").strip()


class _FakeAns:
    answer = "The key insight is X.\nNEXT: a next topic"


class TestDreamCostGuard(unittest.TestCase):
    def setUp(self):
        from eris.knowledge import ask_expert
        import eris.metacognition.dreaming as d
        self.ask_expert = ask_expert
        self.d = d
        self._orig_avail = ask_expert.is_available
        self._orig_ask = ask_expert.ask
        self.calls = {"n": 0}
        ask_expert.is_available = lambda: True            # pretend the key IS present (the risk case)

        def _fake_ask(*a, **k):
            self.calls["n"] += 1                          # TRACE the paid call
            return _FakeAns()
        ask_expert.ask = _fake_ask
        d._dream_cloud_calls = 0
        for k in ("ERIS_DREAM_CLOUD", "ERIS_DREAM_CLOUD_MAX"):
            os.environ.pop(k, None)

    def tearDown(self):
        self.ask_expert.is_available = self._orig_avail
        self.ask_expert.ask = self._orig_ask
        self.d._dream_cloud_calls = 0
        for k in ("ERIS_DREAM_CLOUD", "ERIS_DREAM_CLOUD_MAX"):
            os.environ.pop(k, None)

    def _call(self):
        from eris.metacognition.dreaming import DreamingLoop
        return DreamingLoop._claude_condense_and_refine(_StubLoop(), "topic", "elastic", True, "text")

    def test_TRACE_paid_path_does_not_fire_by_default(self):
        out = self._call()
        # THE TRACE: even with the key "available", the paid ask_expert.ask must NOT be invoked.
        self.assertEqual(self.calls["n"], 0, "paid ask_expert.ask FIRED by default — cost guard failed")
        self.assertEqual(out, (None, None, False))        # the dormant (no-spend) return path

    def test_enabled_fires_when_opted_in(self):
        os.environ["ERIS_DREAM_CLOUD"] = "1"
        out = self._call()
        self.assertEqual(self.calls["n"], 1)              # now the paid call fires
        self.assertTrue(out[2])                           # used_claude=True

    def test_ceiling_caps_spend(self):
        os.environ["ERIS_DREAM_CLOUD"] = "1"
        os.environ["ERIS_DREAM_CLOUD_MAX"] = "2"
        self.d._dream_cloud_calls = 0
        for _ in range(5):
            self._call()
        self.assertEqual(self.calls["n"], 2)              # capped at 2 paid calls, not 5


if __name__ == "__main__":
    unittest.main()
