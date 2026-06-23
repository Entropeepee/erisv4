"""Structure-aware contextual chunking — the highest-ROI ingestion upgrade."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.knowledge.chunking import structured_chunks


class TestStructuredChunking(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(structured_chunks(""), [])
        self.assertEqual(structured_chunks("   \n  "), [])

    def test_contextual_header_carries_section_path(self):
        doc = ("# Collapse Theory\n\n"
               "## Path A\n\nA slow always-on integrator that accumulates drift.\n\n"
               "## Path B\n\nA gated corrector that rejects the noise subspace.")
        chunks = structured_chunks(doc, title="SGT Patent")
        # Every chunk is prefixed with Title › Section so a bare fragment is findable.
        a = [c for c in chunks if "accumulates drift" in c][0]
        b = [c for c in chunks if "noise subspace" in c][0]
        self.assertIn("SGT Patent › Collapse Theory › Path A", a)
        self.assertIn("SGT Patent › Collapse Theory › Path B", b)

    def test_title_only_header_when_no_headings(self):
        chunks = structured_chunks("Just a flat paragraph of prose.", title="Notes")
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("Notes\n\n"))

    def test_no_header_when_no_title_no_heading(self):
        chunks = structured_chunks("plain text, nothing else")
        self.assertEqual(chunks, ["plain text, nothing else"])

    def test_long_section_splits_under_budget(self):
        para = ". ".join(f"sentence number {i} about the topic" for i in range(200)) + "."
        chunks = structured_chunks(para, title="T", target_chars=500, overlap_chars=0)
        self.assertGreater(len(chunks), 1)
        # Body (sans header) stays near the budget — no giant chunk.
        for c in chunks:
            body = c.split("\n\n", 1)[-1]
            self.assertLessEqual(len(body), 700)        # target + slack

    def test_overlap_carries_context_across_boundary(self):
        paras = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(4))
        chunks = structured_chunks(paras, title="T", target_chars=350, overlap_chars=80)
        self.assertGreater(len(chunks), 1)
        # The 2nd+ chunks carry a leading ellipsis tail from the prior chunk.
        self.assertTrue(any(c.split("\n\n", 1)[-1].startswith("…") for c in chunks[1:]))

    def test_covers_all_content(self):
        doc = "# H\n\nalpha beta\n\ngamma delta\n\nepsilon zeta"
        joined = " ".join(structured_chunks(doc, title="T"))
        for w in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"):
            self.assertIn(w, joined)


class TestIngestChunkerSelection(unittest.TestCase):
    def test_default_is_structured_with_header(self):
        from eris.knowledge.web_reader import _chunk_for_ingest
        out = _chunk_for_ingest("## Sec\n\nbody text here", "DocTitle")
        self.assertTrue(any("DocTitle › Sec" in c for c in out))

    def test_legacy_flag_uses_naive(self):
        from eris.config import CONFIG
        from eris.knowledge import web_reader
        old = CONFIG.chunker
        CONFIG.chunker = "legacy"
        try:
            out = web_reader._chunk_for_ingest("## Sec\n\nbody text here", "DocTitle")
            self.assertFalse(any("DocTitle › Sec" in c for c in out))  # no header
        finally:
            CONFIG.chunker = old


if __name__ == "__main__":
    unittest.main()
