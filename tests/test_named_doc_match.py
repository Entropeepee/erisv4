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

    def test_list_documents_groups_and_flags_kind(self):
        # backend for the future UI document picker + store diagnostics
        self.mem.ltm.store(_doc("SGT abstract chunk one.", source="reading:sgtpatent.docx"))
        self.mem.ltm.store(_doc("SGT distinction chunk two.", source="reading:sgtpatent.docx"))
        self.mem.mtm.store(MemoryRecord(text="[Self-study: sgt] web boilerplate", bvec=GOAL,
                                        source="exploration:https://uspto.gov",
                                        metadata={"title": "sgt study"}))
        docs = self.mem.list_documents()
        by_src = {d["source"]: d for d in docs}
        self.assertEqual(by_src["reading:sgtpatent.docx"]["chunks"], 2)
        self.assertEqual(by_src["reading:sgtpatent.docx"]["kind"], "file")
        self.assertEqual(by_src["exploration:https://uspto.gov"]["kind"], "note")
        self.assertEqual(docs[0]["source"], "reading:sgtpatent.docx")   # biggest first

    def test_common_short_word_does_not_overmatch(self):
        # lowering the floor must NOT make "the" match every document
        self.mem.ltm.store(_doc("The contents of an unrelated paper.", source="reading:other"))
        self.assertEqual(self.mem.documents_matching("the"), [])

    def test_multiword_name_requires_all_tokens_in_body(self):
        # distinct documents (distinct sources): only the one whose body has BOTH tokens is the
        # seed, and we return its source-group
        self.mem.ltm.store(_doc("This paper covers resonance coupling in detail.",
                                source="reading:docA"))
        self.mem.ltm.store(_doc("This paper covers only resonance, nothing else.",
                                source="reading:docB"))
        hits = self.mem.documents_matching("resonance coupling")
        self.assertEqual(len(hits), 1)
        self.assertIn("coupling", hits[0].text)

    def test_pulls_whole_document_including_non_name_sections(self):
        # the SGT fix: a section that does NOT contain the name (Summary/Distinction) must still
        # be returned because it shares the document's source-group with a name-bearing chunk
        src = "reading:sgtpatent.docx"
        self.mem.ltm.store(_doc("ABSTRACT … the SGT method statistically gates drift.", source=src))
        self.mem.ltm.store(_doc("DISTINCTION FROM PRIOR ART: the architecture itself is the "
                                "novelty; the 1/sqrt(N) scaling is not.", source=src))   # no "SGT"
        hits = self.mem.documents_matching("SGT")
        self.assertEqual(len(hits), 2)
        self.assertTrue(any("DISTINCTION" in h.text for h in hits))   # the body section surfaced

    def test_prefers_real_file_over_web_note_contamination(self):
        # a reading: file beats an exploration: self-study note that merely MENTIONS the name
        self.mem.ltm.store(_doc("SGT patent: statistical gating of control drift.",
                                source="reading:sgtpatent.docx"))
        self.mem.mtm.store(MemoryRecord(
            text="[Self-study: fixed it. please read sgtpatent] USPTO PATENTSCOPE boilerplate",
            bvec=GOAL, source="exploration:https://uspto.gov", metadata={"title": "sgt study"}))
        hits = self.mem.documents_matching("SGT")
        self.assertTrue(all("reading:" in str(h.source) for h in hits))   # no exploration note
        self.assertFalse(any("USPTO" in h.text for h in hits))

    def test_collapses_duplicate_ingests(self):
        # the same file ingested twice under different temp names → one logical result
        body = ("tmp{0}.docx NON PROVISIONAL PATENT APPLICATION STATISTICAL GATING OF CONTROL "
                "SIGNAL DRIFT IN QUANTUM AND PRECISION MEASUREMENT SYSTEMS ABSTRACT A method "
                "system and integrated circuit for controlling quantum or precision measurement "
                "systems by statistically gating control signal drift to reduce unnecessary "
                "physical actuation prevent noise induced degradation and reduce dynamic power "
                "consumption using a dual path control architecture and a continuous non linear "
                "gate function that creates a quiet zone")
        self.mem.ltm.store(_doc(body.format("7rqflxdi"), source="reading:tmp7rqflxdi.docx"))
        self.mem.ltm.store(_doc(body.format("2m7vss4v"), source="reading:tmp2m7vss4v.docx"))
        hits = self.mem.documents_matching("gating")
        self.assertEqual(len(hits), 1)        # near-duplicate ingests collapsed to one

    def test_distinct_claims_are_not_collapsed(self):
        # the review's concern: a patent's Claim 1 and dependent Claim 2 share boilerplate but
        # are DISTINCT — they must NOT be deduped away (length gate + 0.92 threshold protect them)
        src = "reading:sgtpatent.docx"
        self.mem.ltm.store(_doc(
            "Claim 1. A system for statistically gating control signal drift, comprising a "
            "processor configured to compute a statistical noise threshold and apply a soft "
            "non-linear gate that suppresses corrections within a quiet zone.", source=src))
        self.mem.ltm.store(_doc(
            "Claim 2. The system of claim 1, wherein the soft non-linear gate is a logistic "
            "function whose width is derived from a running detrended-variance outlier estimate "
            "updated every sample period across binomial and Poissonian models.", source=src))
        hits = self.mem.documents_matching("SGT gating")
        self.assertEqual(len(hits), 2)        # both claims kept


if __name__ == "__main__":
    unittest.main()
