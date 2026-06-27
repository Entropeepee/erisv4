"""Contractor Layer §10.4/§10.6 wired into the hive: with the gateway ON, OPEN research routes
specialist reasoning to the free tier and (only with ERIS_HIVE_SYNTH_CLOUD=1) synthesis to the
synth tier; a SOVEREIGN research run stays entirely local and never touches the gateway.
Offline: stub backends + stub gateway, one stored doc for retrieval."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import tempfile
import unittest

import numpy as np

from eris.computation.activations import BVec
from eris.memory.tiers import MemoryRecord
from eris.interface.mediator import LLMMediator, LLMBackend, LLMResponse
from eris.interface.gateway import ContractorGateway


class _Stub(LLMBackend):
    def __init__(self, name):
        self.name = name; self.model = name; self.calls = 0
    async def generate(self, prompt, system="", max_tokens=8192, temperature=0.7):
        self.calls += 1
        return LLMResponse(text="Grounded finding [s:0].", provider=self.name,
                           model=self.name, latency_ms=1.0)
    def is_available(self):
        return True


class _Cfg:
    gateway_base_url = "http://localhost:4000/v1"; gateway_api_key = "sk-litellm-local"
    tier_free = "free-pool"; tier_cheap = "cheap-paid"; tier_synth = "synth"


class TestHiveContractorWiring(unittest.TestCase):
    def _orch(self):
        from eris.orchestrator import ErisOrchestrator
        orch = ErisOrchestrator(field_size=16, data_dir=tempfile.mkdtemp())
        # local mediator = a stub ollama; gateway = stub free/cheap/synth
        m = LLMMediator(); m.add_backend(_Stub("ollama")); orch.mediator = m
        self.free = _Stub("gateway-free"); self.synth = _Stub("claude-agent-sdk")
        orch.gateway = ContractorGateway(
            config=_Cfg(),
            backend_factory=lambda g, n: self.free,
            synth_factory=lambda: self.synth)
        rec = MemoryRecord(text="The BLECD framework defines boundary coupling.",
                           bvec=BVec(B=0.5, F=0.5, E=0.4, C=0.5, D=0.2, S=0.3),
                           embedding=np.ones(8, dtype=np.float32),
                           source="reading:Abstract", metadata={"title": "Abstract"})
        orch.memory.ltm.store(rec)
        return orch

    def test_open_routes_reasoning_to_free_tier(self):
        orch = self._orch()
        out = asyncio.run(orch.hive_research("BLECD coupling", scope="doc",
                                             document="BLECD", sensitivity="open"))
        self.assertEqual(out["sensitivity"], "open")
        self.assertGreater(self.free.calls, 0)            # specialists used the gateway free tier
        self.assertIn("gateway-free", out["tier_calls"])

    def test_sovereign_stays_local_never_gateway(self):
        orch = self._orch()
        out = asyncio.run(orch.hive_research("BLECD coupling", scope="doc",
                                             document="BLECD", sensitivity="sovereign"))
        self.assertEqual(out["sensitivity"], "sovereign")
        self.assertEqual(self.free.calls, 0)              # gateway NEVER touched
        self.assertEqual(self.synth.calls, 0)
        self.assertNotIn("gateway-free", out["tier_calls"])
        self.assertIn("ollama", out["tier_calls"])        # local only

    def test_synth_flag_gates_synth_tier(self):
        from eris.config import CONFIG
        orch = self._orch()
        # OFF (default) → synth tier never used
        asyncio.run(orch.hive_research("BLECD coupling", scope="doc", document="BLECD",
                                       sensitivity="open"))
        self.assertEqual(self.synth.calls, 0)
        # ON → synthesis/canonize route to the synth tier
        old = CONFIG.hive_synth_cloud
        CONFIG.hive_synth_cloud = True
        try:
            asyncio.run(orch.hive_research("BLECD coupling", scope="doc", document="BLECD",
                                           sensitivity="open"))
            self.assertGreater(self.synth.calls, 0)
        finally:
            CONFIG.hive_synth_cloud = old


if __name__ == "__main__":
    unittest.main()
