"""documents_matching must find a named document by its NAME appearing in the chunk BODY,
not only in the title/filename — PDFs are ingested with per-section-heading titles, so an
acronym like 'BLECD' lives in the text. But it must NOT pull in plain conversation that
merely mentions the name. Offline, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.computation.activations import BVec
from eris.memory.tiers import MemorySystem, MemoryRecord

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


def _doc(text, title="", source="reading:section"):
    return MemoryRecord(text=text, bvec=GOAL, source=source,
                        metadata=({"title": title} if title else {}))


class TestNamedDocMatch(unittest.TestCase):
    def setUp(self):
        self.mem = MemorySystem(data_dir=tempfile.mkdtemp())

    def test_matches_name_in_body_not_just_title(self):
        # a PDF chunk whose section-heading title is "Abstract" but whose BODY says BLECD
        self.mem.ltm.store(_doc("Abstract\n\nThe BLECD framework treats criticality as a domain.",
                                title="Abstract", source="reading:Abstract"))
        hits = self.mem.documents_matching("BLECD")
        self.assertEqual(len(hits), 1)
        self.assertIn("BLECD", hits[0].text)

    def test_still_matches_by_title_filename(self):
        self.mem.ltm.store(_doc("Some body text.", title="BLECD_v2.0.pdf",
                                source="reading:BLECD_v2.0.pdf"))
        self.assertEqual(len(self.mem.documents_matching("BLECD")), 1)

    def test_does_not_pull_plain_conversation_mentioning_name(self):
        # a chat turn that mentions BLECD is NOT a document → must be ignored
        self.mem.stm.store(MemoryRecord(text="We talked about BLECD yesterday.",
                                        bvec=GOAL, source="conversation"))
        self.assertEqual(self.mem.documents_matching("BLECD"), [])

    def test_short_acronym_name_matches(self):
        # the SGT bug: a 3-char acronym doc name was dropped by the old 4-char token floor
        self.mem.ltm.store(_doc("Statistical Gating Technology\n\nThe SGT method gates on a "
                                "running statistical outlier.", title="sgtpatent",
                                source="reading:sgtpatent"))
        hits = self.mem.documents_matching("SGT")
        self.assertEqual(len(hits), 1)
        # filename substring also works (sgt ∈ sgtpatent)
        self.assertEqual(len(self.mem.documents_matching("sgtpatent")), 1)

    def test_ranks_doc_chunks_by_query_embedding(self):
        # the scope=doc fix: a doc's chunks are ranked by relevance to the QUESTION, so the
        # conceptual chunk surfaces above tail-end code — not just whichever chunk matched the name
        import numpy as np
        concept = MemoryRecord(
            text="SGT Summary: the novelty is the soft gate, not the 1/sqrt(N) scaling.",
            bvec=GOAL, embedding=np.array([1.0, 0.0], dtype=np.float32),
            source="reading:Summary", metadata={"title": "Summary"})
        code = MemoryRecord(
            text="SGT reference code: acc += k_w*err; resid = buf - (a + b*idx)",
            bvec=GOAL, embedding=np.array([0.0, 1.0], dtype=np.float32),
            source="reading:Code", metadata={"title": "Code"})
        self.mem.ltm.store(code); self.mem.ltm.store(concept)   # store code first (storage order)
        q_emb = np.array([1.0, 0.0], dtype=np.float32)          # points at the concept chunk
        hits = self.mem.documents_matching("SGT", max_chunks=2, query_embedding=q_emb)
        self.assertEqual(len(hits), 2)
        self.assertIn("novelty", hits[0].text)                 # concept ranked first, not the code

    def test_common_short_word_does_not_overmatch(self):
        # lowering the floor must NOT make "the" match every document
        self.mem.ltm.store(_doc("The contents of an unrelated paper.", source="reading:other"))
        self.assertEqual(self.mem.documents_matching("the"), [])

    def test_multiword_name_requires_all_tokens_in_body(self):
        self.mem.ltm.store(_doc("This paper covers resonance coupling in detail.",
                                source="reading:sec"))
        self.mem.ltm.store(_doc("This paper covers only resonance, nothing else.",
                                source="reading:sec"))
        hits = self.mem.documents_matching("resonance coupling")
        self.assertEqual(len(hits), 1)
        self.assertIn("coupling", hits[0].text)


if __name__ == "__main__":
    unittest.main()
