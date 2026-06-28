"""Memory REPLAY (replay_consolidate / consolidate_records): the second half of sleep, distinct
from consolidate()'s tier-promotion. It folds near-duplicate LIBRARY traces of the SAME provenance
into one REINFORCED record — without deleting content, without crossing provenance families, and
without ever touching her subjective voice (reflection / dream / ponder) or the thought-stream.
Offline, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.computation.activations import BVec
from eris.memory.tiers import (
    MemorySystem, MemoryRecord,
    consolidate_records, _provenance_family,
)

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)

# A realistic patent abstract, ingested twice under different temp filenames (the junk class).
_ABS = ("NON PROVISIONAL PATENT APPLICATION STATISTICAL GATING OF CONTROL SIGNAL DRIFT IN "
        "QUANTUM AND PRECISION MEASUREMENT SYSTEMS ABSTRACT A method system and integrated "
        "circuit for controlling quantum or precision measurement systems by statistically "
        "gating control signal drift to reduce unnecessary physical actuation prevent noise "
        "induced degradation and reduce dynamic power consumption using a dual path control "
        "architecture and a continuous non linear gate function that creates a quiet zone")


def _rec(text, source, **kw):
    return MemoryRecord(text=text, bvec=GOAL, source=source, **kw)


class TestProvenanceFamily(unittest.TestCase):
    def test_strips_final_variable_segment(self):
        self.assertEqual(_provenance_family("reading:tmpA.docx"), "reading")
        self.assertEqual(_provenance_family("exploration:https://uspto.gov"), "exploration")  # URL-safe
        self.assertEqual(_provenance_family("study:qa:gating"), "study")
        self.assertEqual(_provenance_family("deepread:doc7:leaf"), "deepread")
        self.assertEqual(_provenance_family("reflection"), "reflection")     # no colon → itself


class TestConsolidateRecords(unittest.TestCase):
    def test_folds_reingested_duplicate_into_one(self):
        recs = [_rec(_ABS, "reading:tmp7rqflxdi.docx"),
                _rec(_ABS, "reading:tmp2m7vss4v.docx")]   # same file, different temp names
        kept, merged = consolidate_records(recs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(merged, 1)

    def test_survivor_is_reinforced_not_forgotten(self):
        a = _rec(_ABS, "reading:tmpA.docx", access_count=2, timestamp=100.0)
        b = _rec(_ABS, "reading:tmpB.docx", access_count=3, timestamp=500.0)
        kept, merged = consolidate_records([a, b])
        self.assertEqual(len(kept), 1)
        self.assertGreaterEqual(kept[0].access_count, 2 + 3)    # summed reinforcement
        self.assertEqual(kept[0].timestamp, 500.0)             # freshest kept → won't prune away

    def test_never_merges_across_provenance_families(self):
        # identical text, DIFFERENT families → must stay separate (a reading is not an exploration)
        recs = [_rec(_ABS, "reading:tmpA.docx"), _rec(_ABS, "exploration:https://x")]
        kept, merged = consolidate_records(recs)
        self.assertEqual(len(kept), 2)
        self.assertEqual(merged, 0)

    def test_never_merges_subjective_voice(self):
        # two near-identical reflections must BOTH survive — her thinking is never flattened
        text = ("I keep returning to the idea that coherence is the thing that matters most here "
                "and it colors how I read everything else in this material.")
        kept, merged = consolidate_records([_rec(text, "reflection"), _rec(text, "reflection")])
        self.assertEqual(len(kept), 2)
        self.assertEqual(merged, 0)

    def test_distinct_chunks_of_same_doc_are_kept(self):
        # different sections of ONE document share a family but are NOT near-duplicates → kept
        recs = [_rec("Section 5.2 System Architecture: five interconnected modules in a loop.",
                     "reading:sgt.docx"),
                _rec("Section 5.3 Distinction from Prior Art: the threshold derives from physics.",
                     "reading:sgt.docx")]
        kept, merged = consolidate_records(recs)
        self.assertEqual(len(kept), 2)
        self.assertEqual(merged, 0)

    def test_empty_and_singleton_are_noops(self):
        self.assertEqual(consolidate_records([]), ([], 0))
        one = [_rec(_ABS, "reading:a.docx")]
        kept, merged = consolidate_records(one)
        self.assertEqual(len(kept), 1)
        self.assertEqual(merged, 0)


class TestReplayConsolidateOnMemorySystem(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        self.mem = MemorySystem(data_dir=self.data_dir)

    def test_replay_collapses_mtm_and_ltm_duplicates_and_persists(self):
        self.mem.mtm.store(_rec(_ABS, "reading:tmpA.docx"))
        self.mem.mtm.store(_rec(_ABS, "reading:tmpB.docx"))
        self.mem.ltm.store(_rec(_ABS, "research:srcX"))
        self.mem.ltm.store(_rec(_ABS, "research:srcY"))
        out = self.mem.replay_consolidate()
        self.assertEqual(out["mtm_merged"], 1)
        self.assertEqual(out["ltm_merged"], 1)
        self.assertEqual(self.mem.mtm.size, 1)
        self.assertEqual(len(self.mem.ltm._records), 1)
        # persisted: a fresh MemorySystem on the same dir sees the collapsed store
        reloaded = MemorySystem(data_dir=self.data_dir)
        self.assertEqual(reloaded.mtm.size, 1)

    def test_replay_leaves_stm_untouched(self):
        self.mem.stm.store(_rec("a conversational turn", "conversation"))
        self.mem.stm.store(_rec("a conversational turn", "conversation"))
        before = len(self.mem.stm.get_all())
        self.mem.replay_consolidate()
        self.assertEqual(len(self.mem.stm.get_all()), before)    # ephemeral turns never folded

    def test_replay_is_noop_when_nothing_duplicates(self):
        self.mem.mtm.store(_rec("a unique passage about coupled oscillators", "reading:a.docx"))
        self.mem.mtm.store(_rec("an unrelated passage about Fisher information", "reading:b.docx"))
        out = self.mem.replay_consolidate()
        self.assertEqual(out, {"mtm_merged": 0, "ltm_merged": 0})
        self.assertEqual(self.mem.mtm.size, 2)


if __name__ == "__main__":
    unittest.main()
