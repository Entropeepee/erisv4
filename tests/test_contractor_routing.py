"""Contractor Layer §8/§10 — the routing seam and its sovereignty enforcement, end to end
through the ContractorRouter and the Hermes contractor. Stubbed backends; no network/keys."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import unittest

from eris.interface.mediator import LLMMediator, LLMBackend, LLMResponse
from eris.interface.sovereignty import Sensitivity, SovereigntyError
from eris.interface.contractor import ContractorRouter
from eris.interface.gateway import ContractorGateway


class _Stub(LLMBackend):
    def __init__(self, name, *, fail=False, available=True):
        self.name = name; self.model = name; self.fail = fail; self._a = available; self.calls = 0
    async def generate(self, prompt, system="", max_tokens=8192, temperature=0.7):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} down")
        return LLMResponse(text=f"{self.name}:ok", provider=self.name, model=self.name, latency_ms=1.0)
    def is_available(self):
        return self._a


class _Cfg:
    gateway_base_url = "http://localhost:4000/v1"; gateway_api_key = "sk-litellm-local"
    tier_free = "free-pool"; tier_cheap = "cheap-paid"; tier_synth = "synth"


def _local_mediator():
    m = LLMMediator(); m.add_backend(_Stub("ollama")); return m


def _gateway(free=None, cheap=None, synth=None):
    free = free or _Stub("gateway-free"); cheap = cheap or _Stub("gateway-cheap")
    synth = synth or _Stub("claude-agent-sdk")
    f = lambda group, name: free if "free" in name else cheap
    return ContractorGateway(config=_Cfg(), backend_factory=f, synth_factory=lambda: synth)


def _run(c): return asyncio.run(c)


class TestRouting(unittest.TestCase):
    def test_decision_to_tier_mapping(self):
        r = ContractorRouter(_gateway(), _local_mediator())
        self.assertEqual(r.tier_for_decision("CONTINUE"), "local")
        self.assertEqual(r.tier_for_decision("SWITCH"), "free")
        self.assertEqual(r.tier_for_decision("ESCALATE"), "cheap")

    def test_open_free_resolves_to_gateway(self):
        r = ContractorRouter(_gateway(), _local_mediator())
        resp = _run(r.generate(Sensitivity.OPEN, "free", "hi"))
        self.assertEqual(resp.text, "gateway-free:ok")

    def test_open_falls_back_to_local_when_tier_unavailable(self):
        gw = _gateway(free=_Stub("gateway-free", available=False))
        r = ContractorRouter(gw, _local_mediator())
        resp = _run(r.generate(Sensitivity.OPEN, "free", "hi"))
        self.assertEqual(resp.text, "ollama:ok")          # graceful local fallback

    def test_open_generate_failure_falls_back_to_local(self):
        gw = _gateway(free=_Stub("gateway-free", fail=True))
        r = ContractorRouter(gw, _local_mediator())
        resp = _run(r.generate(Sensitivity.OPEN, "free", "hi"))
        self.assertEqual(resp.text, "ollama:ok")

    def test_sovereign_always_resolves_local(self):
        r = ContractorRouter(_gateway(), _local_mediator())
        self.assertEqual(r.resolve(Sensitivity.SOVEREIGN, "free").name, "ollama")
        resp = _run(r.generate(Sensitivity.SOVEREIGN, "synth", "secret"))
        self.assertEqual(resp.text, "ollama:ok")

    def test_sovereign_with_no_local_backend_fails_closed(self):
        r = ContractorRouter(_gateway(), LLMMediator())   # empty local mediator
        with self.assertRaises(SovereigntyError):
            r.resolve(Sensitivity.SOVEREIGN, "local")

    def test_sovereign_never_falls_back_to_cloud_on_local_failure(self):
        m = LLMMediator(); m.add_backend(_Stub("ollama", fail=True))
        r = ContractorRouter(_gateway(), m)
        with self.assertRaises(RuntimeError):             # local failed → raises, NOT cloud
            _run(r.generate(Sensitivity.SOVEREIGN, "local", "secret"))

    def test_cost_log_counts_tiers_and_paid(self):
        costs = {}
        r = ContractorRouter(_gateway(), _local_mediator(), cost_log=costs)
        _run(r.generate(Sensitivity.OPEN, "free", "a"))
        _run(r.generate(Sensitivity.OPEN, "cheap", "b"))
        self.assertEqual(costs.get("gateway-free"), 1)
        self.assertEqual(costs.get("gateway-cheap"), 1)
        self.assertEqual(costs.get("_paid_calls"), 1)     # cheap is paid, free is not


class _HCfg:
    hermes_base_url = "http://127.0.0.1:8642"; hermes_api_key = "bearer-xyz"


class TestHermes(unittest.TestCase):
    def _hermes(self, cfg=None, captured=None):
        from eris.interface.hermes import HermesContractor
        def poster(url, json, headers, timeout):
            if captured is not None:
                captured.update(url=url, headers=headers, json=json)
            return {"run_id": "r1", "status": "queued"}
        return HermesContractor(config=cfg or _HCfg(), poster=poster)

    def test_open_run_posts_with_bearer_on_loopback(self):
        cap = {}
        out = self._hermes(captured=cap).run("research X", sensitivity="open")
        self.assertEqual(out["status"], "queued")
        self.assertEqual(cap["url"], "http://127.0.0.1:8642/v1/runs")
        self.assertEqual(cap["headers"]["Authorization"], "Bearer bearer-xyz")

    def test_sovereign_task_is_refused(self):
        with self.assertRaises(SovereigntyError):
            self._hermes().run("secret IP work", sensitivity="sovereign")

    def test_non_loopback_is_refused(self):
        from eris.interface.hermes import HermesNotConfiguredError
        class Remote(_HCfg):
            hermes_base_url = "http://10.0.0.9:8642"
        with self.assertRaises(HermesNotConfiguredError):
            self._hermes(cfg=Remote()).run("research", sensitivity="open")

    def test_disabled_when_unconfigured(self):
        from eris.interface.hermes import HermesContractor, HermesNotConfiguredError
        class Off(_HCfg):
            hermes_base_url = ""; hermes_api_key = ""
        h = HermesContractor(config=Off())
        self.assertFalse(h.enabled)
        with self.assertRaises(HermesNotConfiguredError):
            h.run("x", sensitivity="open")


if __name__ == "__main__":
    unittest.main()
