"""Turnkey Eris arm — the parts testable WITHOUT a served model: the real-token meter (wraps
backend.generate and sums LLMResponse.tokens_used), the scratch-memory wipe (so item N+1 can't
retrieve item N's passage), and the live-store safety guard. The full ask() path needs Ollama and
is exercised by running the benchmark, not here."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import tempfile
import unittest

from eris.computation.activations import BVec
from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.memory.thought_stream import ThoughtStream
from eris.experiments.benchmarks.eris_arm import _TokenMeter, _wipe_memory, make_eris_arm

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


class _Resp:
    def __init__(self, t): self.tokens_used = t


class _Backend:
    def __init__(self, toks): self._toks = toks
    async def generate(self, *a, **kw): return _Resp(self._toks)


class _Mediator:
    def __init__(self, *backends): self._backends = list(backends)


class TestTokenMeter(unittest.TestCase):
    def test_sums_real_tokens_across_backends(self):
        med = _Mediator(_Backend(30), _Backend(12))
        meter = _TokenMeter().attach(med)
        # call each wrapped backend once
        for b in med._backends:
            asyncio.run(b.generate("p"))
        self.assertEqual(meter.tokens, 42)
        self.assertEqual(meter.calls, 2)

    def test_reset_zeroes_counters(self):
        med = _Mediator(_Backend(10))
        meter = _TokenMeter().attach(med)
        asyncio.run(med._backends[0].generate("p"))
        meter.reset()
        self.assertEqual(meter.tokens, 0)
        self.assertEqual(meter.calls, 0)

    def test_attach_is_idempotent_no_double_count(self):
        med = _Mediator(_Backend(5))
        meter = _TokenMeter().attach(med).attach(med)        # wrap twice
        asyncio.run(med._backends[0].generate("p"))
        self.assertEqual(meter.tokens, 5)                    # counted once, not 10

    def test_preserves_response_passthrough(self):
        med = _Mediator(_Backend(7))
        _TokenMeter().attach(med)
        resp = asyncio.run(med._backends[0].generate("p"))
        self.assertEqual(resp.tokens_used, 7)                # the real response still flows back


class TestWipeMemory(unittest.TestCase):
    class _Orch:
        def __init__(self, data_dir):
            self.memory = MemorySystem(data_dir=data_dir)
            self.thought_stream = ThoughtStream(path=os.path.join(data_dir, "t.jsonl"))

    def test_wipe_clears_every_tier_and_thoughtstream(self):
        orch = self._Orch(tempfile.mkdtemp())
        orch.memory.mtm.store(MemoryRecord(text="item N passage", bvec=GOAL, source="reading:bench"))
        orch.memory.ltm.store(MemoryRecord(text="ltm thing", bvec=GOAL, source="synthesis:x"))
        orch.memory.stm.store(MemoryRecord(text="a turn", bvec=GOAL, source="conversation"))
        from eris.memory.thought_stream import link_and_store
        link_and_store(orch.thought_stream, topic="t", regime="plastic", text="a thought")
        _wipe_memory(orch)
        self.assertEqual(orch.memory.mtm.size, 0)
        self.assertEqual(len(orch.memory.ltm._records), 0)
        self.assertEqual(len(orch.memory.stm.get_all()), 0)
        self.assertEqual(orch.thought_stream.size(), 0)

    def test_wipe_isolates_items_no_cross_retrieval(self):
        # after a wipe, the previous item's ingested passage is gone from the doc pool
        orch = self._Orch(tempfile.mkdtemp())
        orch.memory.mtm.store(MemoryRecord(text="SECRET_ITEM_N", bvec=GOAL, source="reading:bench"))
        _wipe_memory(orch)
        self.assertFalse(any("SECRET_ITEM_N" in r.text for r in orch.memory.mtm._records))


class TestSafetyGuard(unittest.TestCase):
    def test_refuses_to_benchmark_against_live_store(self):
        with self.assertRaises(ValueError):
            make_eris_arm(data_dir="eris_data")


if __name__ == "__main__":
    unittest.main()
