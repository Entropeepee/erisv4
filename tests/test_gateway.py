"""Contractor Layer §7/§8/§10 — the gateway wiring: tier backends, failover cascade, semantic
cache, and the Agent-SDK key-bypass guard. All stubbed; no network, no real keys."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import unittest

from eris.interface.mediator import LLMBackend, LLMResponse
from eris.interface.gateway import CachingBackend, ClaudeAgentSDKBackend, ContractorGateway


class _StubBackend(LLMBackend):
    def __init__(self, name, *, text="ok", fail=False, available=True):
        self.name = name
        self.model = name
        self.text = text
        self.fail = fail
        self._available = available
        self.calls = 0

    async def generate(self, prompt, system="", max_tokens=8192, temperature=0.7):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} 429 simulated")
        return LLMResponse(text=self.text, provider=self.name, model=self.name, latency_ms=1.0)

    def is_available(self):
        return self._available


class _Cfg:
    gateway_base_url = "http://localhost:4000/v1"
    gateway_api_key = "sk-litellm-local"
    tier_free = "free-pool"
    tier_cheap = "cheap-paid"
    tier_synth = "synth"


def _run(coro):
    return asyncio.run(coro)


class TestCachingBackend(unittest.TestCase):
    def test_second_identical_prompt_served_from_cache(self):
        inner = _StubBackend("free", text="answer")
        cached = CachingBackend(inner)
        r1 = _run(cached.generate("What is BLECD?", system="s"))
        r2 = _run(cached.generate("  what   IS   blecd? ", system="S"))   # normalized-identical
        self.assertEqual(r1.text, r2.text)
        self.assertEqual(inner.calls, 1)        # NO second upstream call
        self.assertEqual(cached.hits, 1)
        self.assertEqual(cached.misses, 1)


class TestFailover(unittest.TestCase):
    def test_429_on_free_advances_to_cheap(self):
        free = _StubBackend("gateway-free", fail=True)
        cheap = _StubBackend("gateway-cheap", text="from-cheap")
        gw = ContractorGateway(
            config=_Cfg(),
            backend_factory=lambda group, name: free if "free" in name else cheap)
        m = gw.open_mediator()
        resp = _run(m.generate("hello"))
        self.assertEqual(resp.text, "from-cheap")   # failover fired
        self.assertEqual(free.calls, 1)
        self.assertEqual(cheap.calls, 1)


class TestTiers(unittest.TestCase):
    def test_gateway_off_has_no_free_cheap_but_synth_exists(self):
        class Off(_Cfg):
            gateway_base_url = ""
        gw = ContractorGateway(config=Off(), synth_factory=lambda: _StubBackend("claude-agent-sdk"))
        self.assertFalse(gw.enabled)
        self.assertIsNone(gw.tier("free"))
        self.assertIsNone(gw.tier("cheap"))
        self.assertIsNotNone(gw.tier("synth"))      # synth independent of the gateway
        self.assertEqual(len(gw.open_mediator()._backends), 0)

    def test_tier_backends_are_non_local(self):
        from eris.interface.sovereignty import is_local_backend
        gw = ContractorGateway(config=_Cfg(),
                               backend_factory=lambda g, n: _StubBackend(n))
        self.assertFalse(is_local_backend(gw.tier("free")))
        self.assertFalse(is_local_backend(gw.tier("cheap")))


class TestKeyBypassGuard(unittest.TestCase):
    def test_synth_unavailable_and_raises_when_api_key_set(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-should-not-be-used"
        try:
            b = ClaudeAgentSDKBackend()
            self.assertTrue(ClaudeAgentSDKBackend.key_bypass_risk())
            self.assertFalse(b.is_available())          # refuses → won't bill pay-go
            with self.assertRaises(RuntimeError):
                _run(b.generate("synthesize"))          # loud refusal, not silent spend
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_synth_availability_tracks_sdk_when_no_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        b = ClaudeAgentSDKBackend()
        # SDK isn't installed in CI → unavailable, but for the RIGHT reason (no key bypass)
        self.assertFalse(ClaudeAgentSDKBackend.key_bypass_risk())
        self.assertIsInstance(b.is_available(), bool)


if __name__ == "__main__":
    unittest.main()
