"""Tests for the deep-read (RAPTOR map-reduce) pipeline."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.memory.tiers import MemorySystem
from eris.knowledge import deep_read as dr
from eris.knowledge.deep_read import deep_read, DeepReadConfig, _segment_chunks


class _CountingSummarizer:
    """Fake summarizer: records calls; deep flag tracked."""
    def __init__(self):
        self.calls = []

    def __call__(self, text, deep):
        self.calls.append({"len": len(text), "deep": deep})
        return ("DEEP: " if deep else "sum: ") + text[:20].replace("\n", " ")


def _mem(tmp):
    return MemorySystem(data_dir=os.path.join(tmp, "memory"))


class TestChunking(unittest.TestCase):
    def test_python_chunks_by_structure(self):
        code = ('"""mod"""\nimport os\n\n'
                'def a():\n    return 1\n\n'
                'class B:\n    def m(self):\n        return 2\n')
        chunks = _segment_chunks("x.py", code, DeepReadConfig(chunk_chars=40))
        joined = "".join(chunks)
        self.assertIn("def a", joined)
        self.assertIn("class B", joined)
        self.assertTrue(len(chunks) >= 1)

    def test_paragraph_chunks_dont_cut_mid_para(self):
        text = ("Para one is here." * 5) + "\n\n" + ("Para two is here." * 5)
        chunks = _segment_chunks("x.md", text, DeepReadConfig(chunk_chars=90))
        self.assertGreaterEqual(len(chunks), 2)


class TestDeepRead(unittest.TestCase):
    def test_builds_tree_and_stores_leaves_and_summaries(self):
        tmp = tempfile.mkdtemp()
        mem = _mem(tmp)
        # A source big enough to force several chunks + a reduce level.
        text = "\n\n".join(f"Section {i}: " + ("content " * 40) for i in range(20))
        summ = _CountingSummarizer()
        res = deep_read(mem, summ, text, data_dir=tmp,
                        cfg=DeepReadConfig(chunk_chars=300, group_size=4))
        self.assertGreater(res["n_chunks"], 4)
        self.assertTrue(res["synthesis"].startswith("DEEP: "))   # final synth is deep
        self.assertGreaterEqual(res["levels"], 2)
        # The final synthesis pass used deep=True exactly once (the synth).
        self.assertEqual(sum(1 for c in summ.calls if c["deep"]), 1)
        # Memory holds BOTH leaf chunks and summary nodes.
        recs = mem.all_records()
        kinds = {r.metadata.get("kind") for r in recs}
        self.assertIn("chunk", kinds)        # retrievable leaf detail
        self.assertIn("summary", kinds)      # gestalt nodes

    def test_resume_skips_completed_chunks(self):
        tmp = tempfile.mkdtemp()
        mem = _mem(tmp)
        text = "\n\n".join(f"Block {i} " + ("x " * 30) for i in range(8))
        cfg = DeepReadConfig(chunk_chars=200, group_size=4)
        s1 = _CountingSummarizer()
        deep_read(mem, s1, text, data_dir=tmp, cfg=cfg)
        first_map_calls = sum(1 for c in s1.calls if not c["deep"])
        # Re-run on the SAME source: it's complete -> cached, zero new calls.
        s2 = _CountingSummarizer()
        res2 = deep_read(mem, s2, text, data_dir=tmp, cfg=cfg)
        self.assertTrue(res2.get("cached"))
        self.assertEqual(len(s2.calls), 0)
        self.assertGreater(first_map_calls, 0)

    def test_resume_after_partial(self):
        tmp = tempfile.mkdtemp()
        mem = _mem(tmp)
        text = "\n\n".join(f"Item {i} " + ("y " * 30) for i in range(6))
        cfg = DeepReadConfig(chunk_chars=200, group_size=4)

        # Summarizer that dies after 2 chunks to simulate a kill mid-run.
        class _Dies:
            def __init__(self): self.n = 0
            def __call__(self, t, deep):
                self.n += 1
                if self.n > 2:
                    raise RuntimeError("killed")
                return "sum"
        try:
            deep_read(mem, _Dies(), text, data_dir=tmp, cfg=cfg)
        except RuntimeError:
            pass
        # Resume with a good summarizer: it must NOT re-summarize the first 2.
        s2 = _CountingSummarizer()
        res = deep_read(mem, s2, text, data_dir=tmp, cfg=cfg)
        self.assertTrue(res["resumed"])
        self.assertTrue(res["synthesis"])
        # fewer map calls than total chunks, since 2 were already done
        map_calls = sum(1 for c in s2.calls if not c["deep"])
        self.assertLess(map_calls, res["n_chunks"])

    def test_ltm_source(self):
        tmp = tempfile.mkdtemp()
        mem = _mem(tmp)
        from eris.knowledge.embeddings import get_embedding
        for i in range(5):
            mem.store_text(f"I learned fact number {i} about coherence.",
                           embedding=get_embedding(f"fact {i}"), source="test")
        res = deep_read(mem, _CountingSummarizer(), "ltm", data_dir=tmp,
                        cfg=DeepReadConfig(chunk_chars=500))
        self.assertGreater(res["n_chunks"], 0)
        self.assertTrue(res["synthesis"])


if __name__ == "__main__":
    unittest.main()
