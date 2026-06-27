"""§2/§6: end-to-end retrieval shadow on a tiny in-memory corpus with
deterministic embeddings — both paths run, the FLOOR drives in shadow, and one
arbiter-scored divergence row lands."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.computation.activations import BVec
from eris.knowledge.embeddings import get_embedding
from eris.dual.retrieval import build_retrieval_dualpath
from eris.dual.path import Mode
from eris.dual.arbiter import Arbiter
from eris.dual.divergence_log import DivergenceLog
from eris.dual import report


CORPUS = [
    ("Kuramoto coupling drives synchronization of oscillators above a critical K.", "k1"),
    ("Self-organized criticality describes systems poised at a phase transition.", "soc"),
    ("Predictive coding casts perception as hierarchical Bayesian inference.", "pc"),
    ("The Reynolds operator averages over a symmetry group to build invariants.", "rey"),
]


def _mem():
    m = MemorySystem(data_dir=tempfile.mkdtemp())
    for text, sha in CORPUS:
        m.mtm.store(MemoryRecord(text=text, bvec=BVec(), embedding=get_embedding(text),
                                 source=f"reading:{sha}", metadata={"sha256": sha,
                                                                    "title": sha}))
    return m


class TestRetrievalShadow(unittest.TestCase):
    def test_shadow_runs_both_floor_drives_logs_one_row(self):
        m = _mem()
        path = os.path.join(tempfile.mkdtemp(), "dual", "divergence.jsonl")
        log = DivergenceLog(path)
        dp = build_retrieval_dualpath(m, mode=Mode.SHADOW, arbiter=Arbiter(), logger=log)
        q = "what drives Kuramoto synchronization?"
        res = dp.run(q, query_bvec=BVec(), query_embedding=get_embedding(q), gold="sha:k1")
        self.assertTrue(res.records)                 # floor returned an answer
        rows = [r for r in log.rows() if "verdict" in r]
        self.assertEqual(len(rows), 1)               # exactly one divergence row
        row = rows[0]
        # Both paths were scored, with arbiter sub-scores on each.
        self.assertIn("success", row["trad"]["arbiter"])
        self.assertIn("success", row["novel"]["arbiter"])
        self.assertIn("aligned_ids", row["novel"])   # novel channels carried through
        self.assertIn(row["verdict"], ("novel_wins", "trad_wins", "tie", "both_miss"))

    def test_traditional_only_logs_nothing(self):
        m = _mem()
        log = DivergenceLog(os.path.join(tempfile.mkdtemp(), "d.jsonl"))
        dp = build_retrieval_dualpath(m, mode=Mode.TRADITIONAL_ONLY, logger=log)
        dp.run("anything", query_bvec=BVec(), query_embedding=get_embedding("anything"))
        self.assertEqual(log.rows(), [])             # floor-only: no shadow logging

    def test_novel_only_returns_resonant_set(self):
        m = _mem()
        dp = build_retrieval_dualpath(m, mode=Mode.NOVEL_ONLY)
        res = dp.run("predictive coding", query_bvec=BVec(),
                     query_embedding=get_embedding("predictive coding"))
        self.assertIsNotNone(res.aligned)            # resonant channels present

    def test_report_summarizes_log(self):
        m = _mem()
        path = os.path.join(tempfile.mkdtemp(), "divergence.jsonl")
        log = DivergenceLog(path)
        dp = build_retrieval_dualpath(m, mode=Mode.SHADOW, arbiter=Arbiter(), logger=log)
        for q, gold in [("Kuramoto synchronization", "sha:k1"),
                        ("self-organized criticality", "sha:soc")]:
            dp.run(q, query_bvec=BVec(), query_embedding=get_embedding(q), gold=gold)
        s = report.summarize(path)
        self.assertEqual(s["turns"], 2)
        self.assertIn("mean_success", s)


if __name__ == "__main__":
    unittest.main()
