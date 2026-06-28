"""Consolidation v2 — hive synthesis write-back. A canonized, citation-grounded synthesis is
stored as a FIRST-CLASS memory that OUTRANKS the raw chunks it summarized, so retrieval returns
what she LEARNED, not a re-derivation. The 'synthesis' family is foldable (a re-run reinforces).
Offline, deterministic — explicit embeddings since the semantic model is off."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

import numpy as np

from eris.computation.activations import BVec
from eris.memory.tiers import (MemorySystem, MemoryRecord,
                               consolidate_records, _provenance_family)

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


def _emb(*v):
    return np.array(v, dtype=np.float32)


class TestWriteBackSynthesis(unittest.TestCase):
    def setUp(self):
        self.mem = MemorySystem(data_dir=tempfile.mkdtemp())

    def test_writeback_stores_consolidated_first_class_record(self):
        rec = self.mem.write_back_synthesis("statistical gating novelty",
                                            "The novelty is the integration, not the 1/sqrt(N) scaling.",
                                            embedding=_emb(1.0, 0.0), bvec=GOAL, n_sources=6)
        self.assertEqual(_provenance_family(rec.source), "synthesis")
        self.assertTrue(rec.metadata["consolidated"])
        self.assertEqual(rec.metadata["kind"], "synthesis")
        self.assertEqual(rec.metadata["grounds"], 6)
        self.assertIn(rec, self.mem.ltm._records)

    def test_synthesis_outranks_raw_chunks_on_same_topic(self):
        # raw source chunks (decayed slightly) point near the query; the synthesis points AT it and
        # carries the consolidated salience lift → it must come back first.
        q = _emb(1.0, 0.0)
        self.mem.ltm.store(MemoryRecord(text="raw chunk one about gating and drift",
                                        bvec=GOAL, embedding=_emb(0.92, 0.39),
                                        source="reading:sgt.docx", metadata={"title": "sgt"}))
        self.mem.ltm.store(MemoryRecord(text="raw chunk two about gating and drift",
                                        bvec=GOAL, embedding=_emb(0.90, 0.43),
                                        source="reading:sgt.docx", metadata={"title": "sgt"}))
        self.mem.write_back_synthesis("sgt gating",
                                      "LEARNED: the integration is the novelty, not 1/sqrt(N).",
                                      embedding=q, bvec=GOAL, n_sources=2)
        hits = self.mem.retrieve(query_bvec=GOAL, query_embedding=q, top_k=3)
        self.assertIn("LEARNED", hits[0].text)               # her conclusion ranks first
        self.assertTrue(hits[0].metadata.get("consolidated"))

    def test_consolidated_flag_lifts_score_over_identical_unflagged(self):
        # isolate the salience lift: two records, same embedding, one consolidated → it wins
        q = _emb(1.0, 0.0)
        self.mem.ltm.store(MemoryRecord(text="plain", bvec=GOAL, embedding=q, source="reading:x"))
        self.mem.write_back_synthesis("topic", "consolidated", embedding=q, bvec=GOAL)
        hits = self.mem.retrieve(query_bvec=GOAL, query_embedding=q, top_k=2)
        self.assertEqual(hits[0].text, "consolidated")

    def test_duplicate_syntheses_fold_and_reinforce(self):
        # a re-run on the same topic yields a near-identical synthesis → replay folds them into one
        text = ("The genuine novelty of the method is the integration of the soft gate with the "
                "dual-path shared-accumulator architecture, not the well-known 1/sqrt(N) scaling.")
        a = self.mem.write_back_synthesis("sgt", text, embedding=_emb(1.0, 0.0), bvec=GOAL)
        b = self.mem.write_back_synthesis("sgt", text, embedding=_emb(1.0, 0.0), bvec=GOAL)
        kept, merged = consolidate_records(self.mem.ltm._records)
        self.assertEqual(merged, 1)                          # foldable: reinforced, not duplicated
        self.assertEqual(len(kept), 1)

    def test_distinct_topic_syntheses_are_not_folded(self):
        self.mem.write_back_synthesis("gating", "Gating: the novelty is the integration.",
                                      embedding=_emb(1.0, 0.0), bvec=GOAL)
        self.mem.write_back_synthesis("kuramoto", "Kuramoto: coupled oscillators synchronize "
                                      "above a critical coupling strength threshold.",
                                      embedding=_emb(0.0, 1.0), bvec=GOAL)
        kept, merged = consolidate_records(self.mem.ltm._records)
        self.assertEqual(merged, 0)                          # different conclusions, both kept
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
