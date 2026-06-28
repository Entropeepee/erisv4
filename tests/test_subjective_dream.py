"""The subjective dream — her undirected, first-person decompression on the DAY she actually had.
Deliberately distinct from every neighbor: it reads NOTHING new (no crawl), runs NO hive, answers
NO question (not ponder), and does NOT resolve tensions into the field (not _process_tension). It
stores to her own voice channel (thought-stream) + a journal kind='dream', and the 'dream'
provenance family is a consolidation skip-family so replay never folds it. Offline, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.computation.activations import BVec
from eris.memory.autobiography import Autobiography, AutobiographyEntry
from eris.memory.tiers import (MemorySystem, MemoryRecord,
                               _provenance_family, _CONSOLIDATE_SKIP_FAMILIES)
from eris.memory.thought_stream import ThoughtStream
from eris.metacognition.dreaming import DreamingLoop

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


class _Resp:
    def __init__(self, text): self.text = text


class _Mediator:
    """Stub LLM: records calls, returns canned first-person prose (no human-biography trip-wire)."""
    def __init__(self, text="I keep circling the gating idea — it sat with me all day, the way a "
                            "question does when it won't resolve, and only now do I see it rhymes "
                            "with the coherence problem I set down weeks ago."):
        self.text = text
        self.calls = 0

    async def generate(self, prompt="", system=""):
        self.calls += 1
        return _Resp(self.text)


def _loop(mediator=None, thought_stream=None):
    d = tempfile.mkdtemp()
    return DreamingLoop(
        Autobiography(path=os.path.join(d, "a.jsonl")),
        MemorySystem(data_dir=d),
        mediator=mediator, thought_stream=thought_stream)


def _seed_a_day(loop):
    # a conversation turn + a high-dissonance autobiography entry = "a day she had"
    loop.memory.stm.store(MemoryRecord(text="Q: what is genuinely novel about the gating method?",
                                       bvec=GOAL, source="conversation"))
    loop.autobiography._entries_today.append(AutobiographyEntry(
        input_text="the 1/sqrt(N) scaling is not the novelty", dominant_domain="C",
        dissonance=0.9, input_bvec=GOAL, response_bvec=GOAL))


class TestSubjectiveDream(unittest.TestCase):
    def test_dream_provenance_is_a_consolidation_skip_family(self):
        # the wiring that keeps replay-consolidation from ever folding her dreams away
        self.assertIn(_provenance_family("dream:subjective"), _CONSOLIDATE_SKIP_FAMILIES)

    def test_dream_produces_first_person_journal_entry(self):
        med = _Mediator()
        loop = _loop(mediator=med, thought_stream=ThoughtStream(
            path=os.path.join(tempfile.mkdtemp(), "t.jsonl")))
        _seed_a_day(loop)
        out = loop.subjective_dream()
        self.assertIsNotNone(out)
        self.assertEqual(out["kind"], "dream")
        self.assertGreater(out["chars"], 0)
        self.assertEqual(out["journal"]["kind"], "dream")
        self.assertGreaterEqual(med.calls, 1)               # her voice was actually generated

    def test_dream_without_a_day_returns_none_and_does_not_generate(self):
        med = _Mediator()
        loop = _loop(mediator=med)                           # empty memory, empty autobiography
        self.assertIsNone(loop.subjective_dream())
        self.assertEqual(med.calls, 0)                       # never fabricates a day she didn't have

    def test_dream_reads_nothing_external(self):
        # it must NOT crawl: no exploration/reading/research records appear in the library
        loop = _loop(mediator=_Mediator(), thought_stream=ThoughtStream(
            path=os.path.join(tempfile.mkdtemp(), "t.jsonl")))
        _seed_a_day(loop)
        mtm_before = loop.memory.mtm.size
        loop.subjective_dream()
        self.assertEqual(loop.memory.mtm.size, mtm_before)   # nothing ingested
        self.assertFalse(any(str(getattr(r, "source", "")).startswith(("exploration", "reading",
                                                                        "research"))
                             for r in loop.memory.ltm._records))

    def test_dream_drew_on_reports_the_day(self):
        loop = _loop(mediator=_Mediator(), thought_stream=ThoughtStream(
            path=os.path.join(tempfile.mkdtemp(), "t.jsonl")))
        _seed_a_day(loop)
        out = loop.subjective_dream()
        self.assertEqual(out["drew_on"]["conversation"], 1)
        self.assertEqual(out["drew_on"]["tensions"], 1)

    def test_dream_degrades_gracefully_without_a_mediator(self):
        # no LLM → no dream text → None, never raises (she can't speak without her voice)
        loop = _loop(mediator=None, thought_stream=ThoughtStream(
            path=os.path.join(tempfile.mkdtemp(), "t.jsonl")))
        _seed_a_day(loop)
        self.assertIsNone(loop.subjective_dream())


if __name__ == "__main__":
    unittest.main()
