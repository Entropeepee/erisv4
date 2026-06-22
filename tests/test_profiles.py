"""Tests for the mode/profile selector (per-request Fast/Deep + custom)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import json
import tempfile
import unittest

from eris.interface.profiles import Profile, ProfileStore, builtin_profiles


class _Rec:
    """Records the max_tokens each generate() call receives."""
    def __init__(self, text="Stable answer."):
        self.calls = []
        self._text = text

    def is_available(self):
        return True

    async def generate(self, prompt="", system="", max_tokens=8192, temperature=0.7):
        self.calls.append({"max_tokens": max_tokens, "temperature": temperature})
        class R:
            pass
        r = R()
        r.text = self._text
        r.provider = "rec"
        r.model = "m"
        r.reasoning = ""
        return r


class TestProfileStore(unittest.TestCase):
    def test_builtin_fallback_when_no_file(self):
        s = ProfileStore(tempfile.mkdtemp())
        ids = [p.id for p in s.list()]
        self.assertIn("fast", ids)
        self.assertIn("deep", ids)
        self.assertEqual(s.default().id, "fast")

    def test_loads_from_json(self):
        d = tempfile.mkdtemp()
        json.dump([{"id": "zippy", "label": "Zippy", "default": True,
                    "max_tokens": 512, "ttc": False}],
                  open(os.path.join(d, "profiles.json"), "w"))
        s = ProfileStore(d)
        self.assertEqual([p.id for p in s.list()], ["zippy"])
        self.assertEqual(s.default().id, "zippy")
        self.assertEqual(s.get("zippy").max_tokens, 512)

    def test_malformed_file_falls_back(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "profiles.json"), "w").write("{ this is not json")
        s = ProfileStore(d)
        self.assertEqual(s.default().id, "fast")        # graceful fallback

    def test_get_unknown_returns_default(self):
        s = ProfileStore(tempfile.mkdtemp())
        self.assertEqual(s.get("nope").id, "fast")

    def test_first_becomes_default_if_none_marked(self):
        d = tempfile.mkdtemp()
        json.dump([{"id": "a"}, {"id": "b"}],
                  open(os.path.join(d, "profiles.json"), "w"))
        s = ProfileStore(d)
        self.assertEqual(s.default().id, "a")


class TestProcessHonorsProfile(unittest.TestCase):
    def _orch(self):
        from eris.orchestrator import ErisOrchestrator
        return ErisOrchestrator(data_dir=tempfile.mkdtemp(), field_size=16)

    def test_fast_profile_short_and_single_call(self):
        orch = self._orch()
        rec = _Rec()
        orch.mediator._backends = [rec]
        fast = Profile(id="fast", field_steps=30, max_tokens=1024, ttc=False)
        asyncio.run(orch.process("hi", profile=fast))
        self.assertEqual(orch.counters.pde_steps, 30)        # field_steps honored
        self.assertEqual(orch.counters.llm_samples, 1)       # no TTC -> one call
        self.assertEqual(rec.calls[0]["max_tokens"], 1024)   # token budget honored

    def test_deep_profile_multisample_and_bigger_budget(self):
        orch = self._orch()
        rec = _Rec()
        orch.mediator._backends = [rec]
        deep = next(p for p in builtin_profiles() if p.id == "deep")
        asyncio.run(orch.process("hi", profile=deep))
        self.assertEqual(orch.counters.pde_steps, 50)        # deep field_steps
        self.assertGreater(orch.counters.llm_samples, 1)     # TTC sampled
        self.assertLessEqual(orch.counters.llm_samples, deep.ttc_max_samples)
        self.assertTrue(all(c["max_tokens"] == 4096 for c in rec.calls))

    def test_no_profile_is_ambient_unchanged(self):
        # process() with no profile reproduces current global defaults.
        orch = self._orch()
        rec = _Rec()
        orch.mediator._backends = [rec]
        asyncio.run(orch.process("hi"))
        from eris.config import CONFIG
        self.assertEqual(orch.counters.pde_steps, CONFIG.pde_steps_per_input)
        self.assertEqual(orch.counters.llm_samples, 1)


if __name__ == "__main__":
    unittest.main()
