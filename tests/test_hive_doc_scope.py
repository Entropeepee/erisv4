"""Regression: the named-document / scope=doc retrieval path must not crash and must return
the doc's chunks. A dataclass-__eq__-on-numpy-arrays bug in _rag's `h not in lead` membership
silently emptied EVERY scope=doc run (lead non-empty → array truth value → ValueError).
Offline: model + embeddings stubbed, no network."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import tempfile
import unittest

import numpy as np

from eris.computation.activations import BVec
from eris.memory.tiers import MemoryRecord
from eris.interface.mediator import LLMResponse


class _StubMediator:
    """Returns a canned grounded response so the hive runs without Ollama."""
    _backends = []
    async def generate(self, prompt="", system="", max_tokens=8192, temperature=0.7):
        return LLMResponse(text="Grounded finding [s:0].", provider="stub",
                           model="stub", latency_ms=1.0)
    async def ensemble(self, *a, **k):
        return []


class TestHiveDocScope(unittest.TestCase):
    def _orch_with_doc(self):
        from eris.orchestrator import ErisOrchestrator
        orch = ErisOrchestrator(field_size=16, data_dir=tempfile.mkdtemp())
        orch.mediator = _StubMediator()
        orch.deep_mediator = _StubMediator()
        # a document chunk whose BODY (not title) carries the name — like an ingested PDF
        emb = np.ones(8, dtype=np.float32)
        rec = MemoryRecord(
            text="Abstract\n\nThe BLECD framework describes resonance coupling between "
                 "criticality and boundary domains.",
            bvec=BVec(B=0.5, F=0.5, E=0.4, C=0.5, D=0.2, S=0.3), embedding=emb,
            source="reading:Abstract", metadata={"title": "Abstract"})
        orch.memory.ltm.store(rec)
        return orch

    def test_scope_doc_retrieves_named_document_without_crashing(self):
        orch = self._orch_with_doc()
        summary = asyncio.run(orch.hive_research("BLECD resonance coupling",
                                                 scope="doc", document="BLECD"))
        self.assertNotIn("error", summary)
        self.assertGreater(summary["n_sources"], 0)        # retrieval worked (no silent 0)
        self.assertTrue(any("BLECD" in s for s in summary["sources"]))

    def test_document_lead_in_memory_scope_does_not_crash(self):
        # same hazard fires whenever lead is non-empty — also exercise scope=memory + document
        orch = self._orch_with_doc()
        summary = asyncio.run(orch.hive_research("BLECD resonance coupling",
                                                 scope="memory", document="BLECD"))
        self.assertNotIn("error", summary)
        self.assertGreater(summary["n_sources"], 0)


if __name__ == "__main__":
    unittest.main()
